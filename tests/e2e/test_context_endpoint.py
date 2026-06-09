"""E2E tests for ``GET /v1/sessions/{id}/context`` (issue #60)."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import authed_client


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
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = crypto_box
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    # The fixture skips the API lifespan (httpx ASGITransport doesn't
    # fire startup/shutdown), so mirror the lifespan's runtime
    # registrations explicitly — see ``aios.api.app.create_app``.
    prev_crypto = runtime.crypto_box
    prev_tool_provider = runtime.tool_provider
    runtime.crypto_box = crypto_box
    runtime.tool_provider = SubsystemToolProvider()
    try:
        transport = httpx.ASGITransport(app=app)
        async with authed_client(
            "http://testserver",
            aios_env["AIOS_API_KEY"],
            transport=transport,
        ) as client:
            yield client
    finally:
        runtime.crypto_box = prev_crypto
        runtime.tool_provider = prev_tool_provider


@pytest.fixture
async def seeded_session(pool: Any) -> dict[str, Any]:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"ctx-env-{_uniq()}", account_id=account_id
        )
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
        account_id=account_id,
    )
    session = await sessions_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=None,
        metadata={},
        account_id=account_id,
    )
    await sessions_svc.append_user_message(pool, session.id, "hello", account_id=account_id)
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

        # The content carries a leading ``[received=<iso>]`` envelope line
        # (uniform on every user message), so match the body as a substring.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert any(
            (isinstance(m.get("content"), str) and "hello" in m["content"])
            or (
                isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and "hello" in b.get("text", "") for b in m["content"])
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
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        session = await sessions_svc.get_session(
            pool, seeded_session["session_id"], account_id=account_id
        )
        seq_before = session.last_event_seq
        status_before = session.status

        r = await http_client.get(f"/v1/sessions/{seeded_session['session_id']}/context")
        assert r.status_code == 200

        session = await sessions_svc.get_session(
            pool, seeded_session["session_id"], account_id=account_id
        )
        assert session.last_event_seq == seq_before
        assert session.status == status_before

    async def test_exercises_mcp_discovery_path(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        """Agents with MCP servers reach ``runtime.require_crypto_box()`` via
        ``discover_session_mcp_tools``.  The endpoint runs in the API process,
        so ``runtime.crypto_box`` must be wired up there — otherwise this
        endpoint 500s on any non-trivial agent."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.models.agents import McpServerSpec
        from aios.services import agents as agents_svc
        from aios.services import sessions as sessions_svc

        async with pool.acquire() as conn:
            env = await queries.insert_environment(
                conn, name=f"ctx-mcp-env-{_uniq()}", account_id=account_id
            )
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
            account_id=account_id,
        )
        session = await sessions_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
            account_id=account_id,
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

    async def test_workspace_image_attachment_inlines_without_sandbox(
        self, http_client: httpx.AsyncClient, pool: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #630 follow-up: an image attachment under ``/workspace/...``
        inlines in the ``/context`` preview even though the API process never
        provisions a sandbox.

        ``compose_step_context`` resolves the workspace bind-mount source from
        the session row (``workspace_volume_path``), not a worker-only
        ``SandboxHandle``.  This is the cold case — no handle exists anywhere —
        so pre-fix the attachment degraded to a text marker; post-fix it
        renders as an ``image_url`` part.
        """
        import base64

        from aios.db import queries
        from aios.harness import vision
        from aios.services import agents as agents_svc
        from aios.services import sessions as sessions_svc

        account_id = "acc_test_stub"  # PR 3 scaffolding
        channel = f"echo/acct/chat-{_uniq()}"
        model = "openai/gpt-4o-mini"
        # ``supports_vision`` consults ``_VISION_OVERRIDES`` first — pin it so
        # the test doesn't ride on LiteLLM's model metadata.
        monkeypatch.setitem(vision._VISION_OVERRIDES, model, True)

        async with pool.acquire() as conn:
            env = await queries.insert_environment(
                conn, name=f"ctx-img-env-{_uniq()}", account_id=account_id
            )
        agent = await agents_svc.create_agent(
            pool,
            name=f"ctx-img-agent-{_uniq()}",
            model=model,
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        # ``focal_channel`` set so the inbound below stamps a matching
        # ``focal_channel_at_arrival`` — that equality is what unlocks
        # ``render_user_event``'s attachment branch.
        session = await sessions_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
            account_id=account_id,
            focal_channel=channel,
        )

        # Stage the image at the session's real ``workspace_volume_path``
        # (post-#409 ``<workspace_root>/<account_id>/<session_id>`` shape).
        payload = b"PNGWORKSPACEBYTES"
        workspace_path = await sessions_svc.load_session_workspace_path(
            pool, session.id, account_id=account_id
        )
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "shot.png").write_bytes(payload)

        await sessions_svc.append_user_message(
            pool,
            session.id,
            "look at this",
            {
                "channel": channel,
                "attachments": [
                    {
                        "filename": "shot.png",
                        "content_type": "image/png",
                        "size": len(payload),
                        "in_sandbox_path": "/workspace/shot.png",
                    }
                ],
            },
            account_id=account_id,
        )

        r = await http_client.get(f"/v1/sessions/{session.id}/context")
        assert r.status_code == 200, r.text
        messages = r.json()["messages"]

        image_parts = [
            part
            for m in messages
            if isinstance(m.get("content"), list)
            for part in m["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        assert len(image_parts) == 1, f"expected one inlined image; messages={messages}"
        assert base64.b64encode(payload).decode() in image_parts[0]["image_url"]["url"]
