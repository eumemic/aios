"""E2E tests for the ``pending`` session status (issue #39)."""

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
async def idle_session_id(pool: Any) -> str:
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"pending-env-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"pending-agent-{_uniq()}",
        model="openai/gpt-4o-mini",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    session = await sessions_svc.create_session(
        pool, agent_id=agent.id, environment_id=env.id, title=None, metadata={}
    )
    return session.id


class TestPendingStatus:
    async def test_post_message_flips_idle_to_pending(
        self, http_client: httpx.AsyncClient, idle_session_id: str
    ) -> None:
        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "idle"

        r = await http_client.post(
            f"/v1/sessions/{idle_session_id}/messages",
            json={"content": "hi"},
        )
        assert r.status_code == 201, r.text

        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "pending"

    async def test_running_status_not_clobbered_by_concurrent_message(
        self, http_client: httpx.AsyncClient, pool: Any, idle_session_id: str
    ) -> None:
        """If the session is already ``running``, a new user message must not
        rewrite the status — the in-flight worker owns it."""
        from aios.services import sessions as sessions_svc

        await sessions_svc.set_session_status(pool, idle_session_id, "running")

        r = await http_client.post(
            f"/v1/sessions/{idle_session_id}/messages",
            json={"content": "arrived mid-turn"},
        )
        assert r.status_code == 201, r.text

        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "running"

    async def test_rescheduling_status_not_clobbered(
        self, http_client: httpx.AsyncClient, pool: Any, idle_session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        await sessions_svc.set_session_status(
            pool, idle_session_id, "rescheduling", stop_reason={"type": "rescheduling"}
        )

        r = await http_client.post(
            f"/v1/sessions/{idle_session_id}/messages",
            json={"content": "ping"},
        )
        assert r.status_code == 201, r.text

        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "rescheduling"
