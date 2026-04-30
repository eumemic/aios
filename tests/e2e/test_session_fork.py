"""E2E tests for ``POST /v1/sessions/{id}/fork``.

Forking copies the parent's session row and its full event log into a new
session id with fresh ``evt_`` ids but identical seqs, so the fork's next
forward step sees a context byte-identical to the parent's at fork time.
"""

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

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    with mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
        ) as client:
            yield client


@pytest.fixture
async def parent_session_id(pool: Any) -> str:
    """Create an idle session with a couple of events to clone."""
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"fork-env-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"fork-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    session = await sessions_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="parent",
        metadata={"k": "v"},
    )
    # Append a couple of events so we can verify the copy.
    await sessions_svc.append_user_message(pool, session.id, "first")
    await sessions_svc.append_user_message(pool, session.id, "second")
    # The append flips status to pending; tests that need idle reset it.
    await sessions_svc.set_session_status(pool, session.id, "idle")
    return session.id


class TestForkBasic:
    async def test_creates_new_session_id(
        self, http_client: httpx.AsyncClient, parent_session_id: str
    ) -> None:
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["id"] != parent_session_id
        assert body["id"].startswith("sess_")

    async def test_inherits_config_fields(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        parent = await sessions_svc.get_session(pool, parent_session_id)
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        assert r.status_code == 201, r.text
        fork = r.json()

        assert fork["agent_id"] == parent.agent_id
        assert fork["environment_id"] == parent.environment_id
        assert fork["agent_version"] == parent.agent_version
        assert fork["title"] == parent.title
        assert fork["metadata"] == parent.metadata
        assert fork["status"] == parent.status
        assert fork["last_event_seq"] == parent.last_event_seq

    async def test_copies_event_log_with_fresh_ids(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        parent_events = await sessions_svc.read_events(pool, parent_session_id, limit=200)
        assert len(parent_events) >= 2

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        fork_id = r.json()["id"]
        fork_events = await sessions_svc.read_events(pool, fork_id, limit=200)

        assert len(fork_events) == len(parent_events)
        for p, f in zip(parent_events, fork_events, strict=True):
            assert f.id != p.id
            assert f.id.startswith("evt_")
            assert f.seq == p.seq
            assert f.kind == p.kind
            assert f.data == p.data

    async def test_resets_cumulative_usage(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        await sessions_svc_increment_usage(pool, parent_session_id)
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        fork = r.json()
        # Parent had nonzero token counts; fork starts fresh.
        assert fork["usage"]["input_tokens"] == 0
        assert fork["usage"]["output_tokens"] == 0


class TestForkRefusal:
    async def test_refuses_running_parent(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        await sessions_svc.set_session_status(pool, parent_session_id, "running")
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        assert r.status_code == 409, r.text

    async def test_refuses_pending_parent(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        await sessions_svc.set_session_status(pool, parent_session_id, "pending")
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        assert r.status_code == 409, r.text

    async def test_allowed_when_terminated(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        await sessions_svc.set_session_status(pool, parent_session_id, "terminated")
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        assert r.status_code == 201, r.text
        assert r.json()["status"] == "terminated"

    async def test_404_on_missing_parent(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post(
            "/v1/sessions/sess_01HQR2K7VXBZ9MNPL3WYCT8FZZ/fork",
            json={},
        )
        assert r.status_code == 404, r.text


class TestForkIndependence:
    async def test_appending_to_fork_does_not_affect_parent(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        parent_events_before = await sessions_svc.read_events(pool, parent_session_id, limit=200)
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        fork_id = r.json()["id"]

        await sessions_svc.append_user_message(pool, fork_id, "fork-only")

        parent_events_after = await sessions_svc.read_events(pool, parent_session_id, limit=200)
        assert len(parent_events_after) == len(parent_events_before)

        fork_events = await sessions_svc.read_events(pool, fork_id, limit=200)
        assert len(fork_events) == len(parent_events_before) + 1
        # New event got the next seq beyond the inherited prefix.
        assert fork_events[-1].seq == parent_events_before[-1].seq + 1
        assert fork_events[-1].data["content"] == "fork-only"

    async def test_appending_to_parent_does_not_affect_fork(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        fork_id = r.json()["id"]
        fork_events_before = await sessions_svc.read_events(pool, fork_id, limit=200)

        await sessions_svc.append_user_message(pool, parent_session_id, "parent-only")

        fork_events_after = await sessions_svc.read_events(pool, fork_id, limit=200)
        assert len(fork_events_after) == len(fork_events_before)


class TestForkVaults:
    async def test_copies_vault_bindings(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        from aios.db import queries
        from aios.services import sessions as sessions_svc
        from aios.services import vaults as vaults_svc

        v1 = await vaults_svc.create_vault(pool, display_name=f"vault-{_uniq()}", metadata={})
        v2 = await vaults_svc.create_vault(pool, display_name=f"vault-{_uniq()}", metadata={})
        async with pool.acquire() as conn:
            await queries.set_session_vaults(conn, parent_session_id, [v1.id, v2.id])

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/fork", json={})
        fork = r.json()
        assert fork["vault_ids"] == [v1.id, v2.id]

        # Confirm round-trip via service too (covers the get-with-vaults shape).
        fetched = await sessions_svc.get_session(pool, fork["id"])
        assert fetched.vault_ids == [v1.id, v2.id]


async def sessions_svc_increment_usage(pool: Any, session_id: str) -> None:
    """Helper: bump parent's cumulative usage so we can verify fork resets it."""
    from aios.services import sessions as sessions_svc

    await sessions_svc.increment_usage(pool, session_id, input_tokens=42, output_tokens=7)
