"""E2E tests for ``POST /v1/sessions/{id}/clone``.

Cloneing copies the parent's session row and its full event log into a new
session id with fresh ``evt_`` ids but identical seqs, so the clone's next
forward step sees a context byte-identical to the parent's at clone time.
"""

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

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    with mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock):
        async with authed_client(
            "http://testserver",
            aios_env["AIOS_API_KEY"],
            transport=transport,
        ) as client:
            yield client


@pytest.fixture
async def parent_session_id(pool: Any) -> str:
    """Create an idle session with a couple of events to clone."""
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"clone-env-{_uniq()}", account_id=account_id
        )
    agent = await agents_svc.create_agent(
        pool,
        name=f"clone-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
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
        title="parent",
        metadata={"k": "v"},
        account_id=account_id,
    )
    # Append a user message and an assistant reply that reacts to it, so the
    # parent is idle (no unreacted stimulus — clone requires an idle parent)
    # and there are >=2 events to verify the copy.
    await sessions_svc.append_user_message(pool, session.id, "first", account_id=account_id)
    await sessions_svc.append_event(
        pool,
        session.id,
        "message",
        {"role": "assistant", "content": "ack", "reacting_to": 1},
        account_id=account_id,
    )
    return session.id


class TestCloneBasic:
    async def test_creates_new_session_id(
        self, http_client: httpx.AsyncClient, parent_session_id: str
    ) -> None:
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["id"] != parent_session_id
        assert body["id"].startswith("sess_")

    async def test_inherits_config_fields(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        parent = await sessions_svc.get_session(pool, parent_session_id, account_id=account_id)
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        assert r.status_code == 201, r.text
        clone = r.json()

        assert clone["agent_id"] == parent.agent_id
        assert clone["environment_id"] == parent.environment_id
        assert clone["agent_version"] == parent.agent_version
        assert clone["title"] == parent.title
        assert clone["metadata"] == parent.metadata
        assert clone["status"] == parent.status
        assert clone["last_event_seq"] == parent.last_event_seq

    async def test_copies_event_log_with_fresh_ids(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        parent_events = await sessions_svc.read_events(
            pool, parent_session_id, limit=200, account_id=account_id
        )
        assert len(parent_events) >= 2

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        clone_id = r.json()["id"]
        clone_events = await sessions_svc.read_events(
            pool, clone_id, limit=200, account_id=account_id
        )

        assert len(clone_events) == len(parent_events)
        for p, f in zip(parent_events, clone_events, strict=True):
            assert f.id != p.id
            assert f.id.startswith("evt_")
            assert f.seq == p.seq
            assert f.kind == p.kind
            assert f.data == p.data

    async def test_resets_cumulative_usage(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        await sessions_svc.increment_usage(
            pool, parent_session_id, input_tokens=42, output_tokens=7, account_id=account_id
        )
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        clone = r.json()
        assert clone["usage"]["input_tokens"] == 0
        assert clone["usage"]["output_tokens"] == 0


class TestCloneRefusal:
    async def test_refuses_active_parent(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        # An unreacted user message makes the parent derive ``active`` — clone
        # refuses non-idle parents (their in-flight/owed work would leave the
        # clone's expected event stream undefined).
        await sessions_svc.append_user_message(
            pool, parent_session_id, "more", account_id=account_id
        )
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        assert r.status_code == 409, r.text

    async def test_404_on_missing_parent(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post(
            "/v1/sessions/sess_01HQR2K7VXBZ9MNPL3WYCT8FZZ/clone",
            json={},
        )
        assert r.status_code == 404, r.text


class TestCloneIndependence:
    async def test_appending_to_clone_does_not_affect_parent(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        parent_events_before = await sessions_svc.read_events(
            pool, parent_session_id, limit=200, account_id=account_id
        )
        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        clone_id = r.json()["id"]

        await sessions_svc.append_user_message(pool, clone_id, "clone-only", account_id=account_id)

        parent_events_after = await sessions_svc.read_events(
            pool, parent_session_id, limit=200, account_id=account_id
        )
        assert len(parent_events_after) == len(parent_events_before)

        clone_events = await sessions_svc.read_events(
            pool, clone_id, limit=200, account_id=account_id
        )
        assert len(clone_events) == len(parent_events_before) + 1
        # New event got the next seq beyond the inherited prefix.
        assert clone_events[-1].seq == parent_events_before[-1].seq + 1
        assert clone_events[-1].data["content"] == "clone-only"

    async def test_appending_to_parent_does_not_affect_clone(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.services import sessions as sessions_svc

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        clone_id = r.json()["id"]
        clone_events_before = await sessions_svc.read_events(
            pool, clone_id, limit=200, account_id=account_id
        )

        await sessions_svc.append_user_message(
            pool, parent_session_id, "parent-only", account_id=account_id
        )

        clone_events_after = await sessions_svc.read_events(
            pool, clone_id, limit=200, account_id=account_id
        )
        assert len(clone_events_after) == len(clone_events_before)


class TestCloneVaults:
    async def test_copies_vault_bindings(
        self, http_client: httpx.AsyncClient, pool: Any, parent_session_id: str
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.db import queries
        from aios.services import sessions as sessions_svc
        from aios.services import vaults as vaults_svc

        v1 = await vaults_svc.create_vault(
            pool, display_name=f"vault-{_uniq()}", metadata={}, account_id=account_id
        )
        v2 = await vaults_svc.create_vault(
            pool, display_name=f"vault-{_uniq()}", metadata={}, account_id=account_id
        )
        async with pool.acquire() as conn:
            await queries.set_session_vaults(
                conn, parent_session_id, [v1.id, v2.id], account_id=account_id
            )

        r = await http_client.post(f"/v1/sessions/{parent_session_id}/clone", json={})
        clone = r.json()
        assert clone["vault_ids"] == [v1.id, v2.id]

        # Confirm round-trip via service too (covers the get-with-vaults shape).
        fetched = await sessions_svc.get_session(pool, clone["id"], account_id=account_id)
        assert fetched.vault_ids == [v1.id, v2.id]
