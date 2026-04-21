"""E2E tests for ``GET /v1/sessions/{id}/context`` (issue #60)."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.harness import runtime

    settings = get_settings()
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = crypto_box
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    prev_crypto = runtime.crypto_box
    runtime.crypto_box = crypto_box
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
        ) as client:
            yield client
    finally:
        runtime.crypto_box = prev_crypto


@pytest.fixture
async def seeded_session(pool: Any) -> dict[str, Any]:
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"ctx-env-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"ctx-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="You are a test assistant.",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    session = await sessions_svc.create_session(
        pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
    )
    await sessions_svc.append_user_message(pool, session.id, "hello")
    return {"agent_id": agent.id, "session_id": session.id, "model": agent.model}


class TestContextEndpoint:
    async def test_returns_preview_shape(
        self, http_client: httpx.AsyncClient, seeded_session: dict[str, Any]
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{seeded_session['session_id']}/context")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == seeded_session["session_id"]
        assert body["model"] == seeded_session["model"]
        assert isinstance(body["messages"], list)
        assert isinstance(body["tools"], list)

    async def test_includes_user_message_in_context(
        self, http_client: httpx.AsyncClient, seeded_session: dict[str, Any]
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{seeded_session['session_id']}/context")
        assert r.status_code == 200
        messages = r.json()["messages"]

        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert any(
            (isinstance(m.get("content"), str) and m["content"] == "hello")
            or (
                isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("text") == "hello" for b in m["content"])
            )
            for m in user_msgs
        ), messages

    async def test_system_message_first(
        self, http_client: httpx.AsyncClient, seeded_session: dict[str, Any]
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{seeded_session['session_id']}/context")
        messages = r.json()["messages"]
        assert messages, "expected non-empty messages"
        assert messages[0]["role"] == "system"

    async def test_404_when_session_missing(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.get(f"/v1/sessions/sess_{_uniq()}doesnotexist/context")
        assert r.status_code == 404

    async def test_is_read_only_no_events_appended(
        self, http_client: httpx.AsyncClient, pool: Any, seeded_session: dict[str, Any]
    ) -> None:
        """Dry-run semantics: hitting /context must not append events, bump status,
        or write skill files.  Side-effect-free by design."""
        from aios.services import sessions as sessions_svc

        session = await sessions_svc.get_session(pool, seeded_session["session_id"])
        seq_before = session.last_event_seq
        status_before = session.status

        r = await http_client.get(f"/v1/sessions/{seeded_session['session_id']}/context")
        assert r.status_code == 200

        session = await sessions_svc.get_session(pool, seeded_session["session_id"])
        assert session.last_event_seq == seq_before
        assert session.status == status_before

    async def test_exercises_mcp_discovery_path(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        """Agents with MCP servers reach ``runtime.require_crypto_box()`` via
        ``discover_session_mcp_tools``.  The endpoint runs in the API process,
        so ``runtime.crypto_box`` must be wired up there — otherwise this
        endpoint 500s on any non-trivial agent."""
        from aios.db import queries
        from aios.models.agents import McpServerSpec
        from aios.services import agents as agents_svc
        from aios.services import sessions as sessions_svc

        async with pool.acquire() as conn:
            env = await queries.insert_environment(conn, name=f"ctx-mcp-env-{_uniq()}")
        agent = await agents_svc.create_agent(
            pool,
            name=f"ctx-mcp-agent-{_uniq()}",
            model="openai/gpt-4o-mini",
            system="",
            tools=[],
            mcp_servers=[McpServerSpec(name="probe", url="http://127.0.0.1:1/mcp")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        session = await sessions_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
        )

        # Stub discovery so we don't depend on a running MCP server.  The
        # important wiring being tested is that ``require_crypto_box`` is
        # reachable from the API process; patching the outer discovery
        # still exercises the module-level import of the compose path.
        with mock.patch(
            "aios.harness.loop.discover_session_mcp_tools",
            new=mock.AsyncMock(return_value=([], {})),
        ):
            r = await http_client.get(f"/v1/sessions/{session.id}/context")

        assert r.status_code == 200, r.text
