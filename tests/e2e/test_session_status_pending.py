"""E2E tests for the derived ``active`` session status (issue #39): a user
message gives an idle session owed work, so it derives ``active``."""

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
async def idle_session_id(pool: Any) -> str:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn, name=f"pending-env-{_uniq()}", account_id=account_id
        )
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
    return session.id


class TestActiveStatus:
    async def test_post_message_makes_session_active(
        self, http_client: httpx.AsyncClient, idle_session_id: str
    ) -> None:
        """An unprompted user message gives the session owed work, so its
        derived status flips ``idle → active`` (#39 orchestrators need to
        distinguish a session that will run from one that's truly done)."""
        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "idle"

        r = await http_client.post(
            f"/v1/sessions/{idle_session_id}/messages",
            json={"content": "hi"},
        )
        assert r.status_code == 201, r.text

        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.json()["status"] == "active"

    async def test_armed_schedule_wake_stays_idle(
        self, http_client: httpx.AsyncClient, pool: Any, idle_session_id: str
    ) -> None:
        """Quiescent between fires: a session with an armed future wake (a
        pending scheduled task) but no in-flight or unreacted work derives
        ``idle``. The scheduled task is a row, not an event, so it adds no
        unreacted stimulus — the timer only matters when it actually fires."""
        from datetime import UTC, datetime, timedelta

        from aios.models.scheduled_tasks import ScheduledTaskCreate
        from aios.services import scheduled_tasks as scheduled_tasks_svc

        account_id = "acc_test_stub"  # PR 3 scaffolding
        await scheduled_tasks_svc.add_task(
            pool,
            idle_session_id,
            ScheduledTaskCreate(
                name=f"wake-{_uniq()}",
                fire_at=datetime.now(UTC) + timedelta(hours=1),
                command="echo hi",
                timeout_seconds=30,
            ),
            account_id=account_id,
        )

        r = await http_client.get(f"/v1/sessions/{idle_session_id}")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "idle"
