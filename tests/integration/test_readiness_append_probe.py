"""Readiness write probe checks against the real migrated schema."""

from __future__ import annotations

from typing import Any

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

from aios.api.routers import health
from aios.ids import ACCOUNT, AGENT, ENVIRONMENT, make_id


class _Acquire:
    def __init__(self, conn: asyncpg.Connection[Any]) -> None:
        self.conn = conn

    async def __aenter__(self) -> asyncpg.Connection[Any]:
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: asyncpg.Connection[Any]) -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


@pytest.mark.asyncio
async def test_empty_install_ready_ignores_legitimate_probe_names_and_rolls_back(
    live_conn: asyncpg.Connection[Any],
) -> None:
    """Tenant names and IDs cannot collide with rollback-only probe resources."""
    account_id = make_id(ACCOUNT)
    agent_id = make_id(AGENT)
    environment_id = make_id(ENVIRONMENT)
    await live_conn.execute(
        "INSERT INTO accounts (id, display_name) VALUES ($1, 'readiness_probe')",
        account_id,
    )
    await live_conn.execute(
        "INSERT INTO agents (id, name, model, account_id) "
        "VALUES ($1, 'readiness_probe', 'test/model', $2)",
        agent_id,
        account_id,
    )
    await live_conn.execute(
        "INSERT INTO environments (id, name, account_id) VALUES ($1, 'readiness_probe', $2)",
        environment_id,
        account_id,
    )
    count_sql = (
        "SELECT jsonb_build_object("
        "'accounts', (SELECT count(*) FROM accounts), "
        "'agents', (SELECT count(*) FROM agents), "
        "'environments', (SELECT count(*) FROM environments), "
        "'sessions', (SELECT count(*) FROM sessions), "
        "'events', (SELECT count(*) FROM events))"
    )
    before = await live_conn.fetchval(count_sql)
    app = FastAPI()
    app.include_router(health.router)
    app.state.pool = _Pool(live_conn)
    app.state.retirements_ok = True

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    after = await live_conn.fetchval(count_sql)
    assert after == before
    assert after["sessions"] == 0
