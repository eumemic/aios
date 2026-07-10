"""Exercise the image-clamp self-heal UPDATE against PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_event(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    account_id = "acc_replace_event"
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO accounts
                   (id, parent_account_id, can_mint_children, display_name)
                   VALUES ($1, NULL, TRUE, 'replace-event-test')""",
                account_id,
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=account_id, prefix="replace-event-test"
        )
        async with pool.acquire() as conn:
            event = await queries.append_event(
                conn,
                account_id=account_id,
                session_id=session.id,
                kind="message",
                data={"role": "user", "content": "before"},
            )
        yield pool, account_id, session.id, event.id
    finally:
        await pool.close()


async def test_replace_event_data_executes_scoped_update(
    pool_and_event: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    pool, account_id, session_id, event_id = pool_and_event
    replacement = {"role": "user", "content": [{"type": "text", "text": "after"}]}

    async with pool.acquire() as conn:
        assert await queries.replace_event_data(
            conn, session_id, event_id, replacement, account_id=account_id
        )
        row = await conn.fetchrow("SELECT data, kind FROM events WHERE id = $1", event_id)
        assert dict(row["data"]) == replacement
        assert row["kind"] == "message"

        assert not await queries.replace_event_data(
            conn, session_id, event_id, {"content": "wrong"}, account_id="wrong-account"
        )
        unchanged = await conn.fetchval("SELECT data FROM events WHERE id = $1", event_id)
        assert dict(unchanged) == replacement
