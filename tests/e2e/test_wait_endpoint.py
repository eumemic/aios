"""E2E tests for the ``GET /v1/sessions/{id}/wait`` long-poll endpoint (issue #40)."""

from __future__ import annotations

import asyncio
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
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
    ) as client:
        yield client


@pytest.fixture
async def session_id(pool: Any) -> str:
    from aios.db import queries
    from aios.services import agents as agents_svc
    from aios.services import sessions as sessions_svc

    async with pool.acquire() as conn:
        env = await queries.insert_environment(conn, name=f"wait-env-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"wait-agent-{_uniq()}",
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


class TestWaitEndpoint:
    async def test_returns_existing_events_immediately(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        await sessions_svc.append_user_message(pool, session_id, "hello")

        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after_seq": 0, "timeout": 30},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["events"]) >= 1
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "hello" for e in body["events"]
        )
        assert body["next_after"] == body["events"][-1]["seq"]
        assert body["session_status"] in {"pending", "running", "idle"}

    async def test_empty_after_timeout(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after_seq": 9999, "timeout": 1},
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["events"] == []
        assert body["next_after"] == 9999

    async def test_unblocks_on_new_event(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        from aios.services import sessions as sessions_svc

        async def wait_call() -> httpx.Response:
            return await http_client.get(
                f"/v1/sessions/{session_id}/wait",
                params={"after_seq": 0, "timeout": 10},
            )

        async def delayed_append() -> None:
            await asyncio.sleep(0.3)
            await sessions_svc.append_user_message(pool, session_id, "late hello")

        wait_task = asyncio.create_task(wait_call())
        post_task = asyncio.create_task(delayed_append())

        r, _ = await asyncio.gather(wait_task, post_task)

        assert r.status_code == 200, r.text
        body = r.json()
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "late hello"
            for e in body["events"]
        )

    async def test_rejects_timeout_above_cap(
        self, http_client: httpx.AsyncClient, session_id: str
    ) -> None:
        """Request-level validation rejects out-of-range ``timeout``."""
        r = await http_client.get(
            f"/v1/sessions/{session_id}/wait",
            params={"after_seq": 0, "timeout": 3600},
        )
        assert r.status_code == 422, r.text

    async def test_delta_notifications_do_not_short_circuit_wait(
        self, http_client: httpx.AsyncClient, pool: Any, session_id: str
    ) -> None:
        """Streaming delta pokes on the session channel must not return an empty response.

        ``_notify_delta`` in ``aios.harness.completion`` fires transient
        ``pg_notify`` payloads on ``events_<session_id>`` for every token; these
        payloads start with ``{`` and carry no DB row. A long-poll handler that
        re-reads events on every notification would return empty immediately
        and cause clients to hot-loop on any streaming session.
        """

        async def fire_delta_then_append() -> None:
            from aios.services import sessions as sessions_svc

            async with pool.acquire() as conn:
                for _ in range(3):
                    await conn.execute(
                        "SELECT pg_notify($1, $2)",
                        f"events_{session_id}",
                        '{"delta": "token"}',
                    )
                    await asyncio.sleep(0.05)
            # Real event arrives later; the wait should latch onto this,
            # not any of the deltas.
            await asyncio.sleep(0.3)
            await sessions_svc.append_user_message(pool, session_id, "real")

        async def wait_call() -> httpx.Response:
            return await http_client.get(
                f"/v1/sessions/{session_id}/wait",
                params={"after_seq": 0, "timeout": 5},
            )

        wait_task = asyncio.create_task(wait_call())
        poke_task = asyncio.create_task(fire_delta_then_append())

        r, _ = await asyncio.gather(wait_task, poke_task)

        assert r.status_code == 200, r.text
        body = r.json()
        assert any(
            e["kind"] == "message" and e["data"].get("content") == "real" for e in body["events"]
        ), body
