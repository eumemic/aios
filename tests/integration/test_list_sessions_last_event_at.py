"""Integration test: ``list_sessions`` derives ``last_event_at`` from the
session's newest event (a correlated subquery in ``_list_scoped``).

Before this, ``last_event_at`` was never populated on reads, so the console's
Sessions "Last activity" column fell back to ``created_at`` for every row.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


class TestListSessionsLastEventAt:
    async def test_last_event_at_tracks_newest_event(
        self, migrated_db_url: str, _reset_db_state: None
    ) -> None:
        pool: asyncpg.Pool[Any] = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                    "VALUES ('acc_lea', NULL, TRUE, 'last-event-at-test')"
                )
            _agent, _env, session = await seed_agent_env_session(
                pool, account_id="acc_lea", prefix="lea"
            )

            # No events yet → the subquery yields NULL → None.
            async with pool.acquire() as conn:
                sessions = await queries.list_sessions(conn, account_id="acc_lea")
            s = next(x for x in sessions if x.id == session.id)
            assert s.last_event_at is None

            # After appending events, last_event_at = the newest event's time,
            # which is strictly after the session's created_at.
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn,
                    account_id="acc_lea",
                    session_id=session.id,
                    kind="message",
                    data={"role": "user", "content": "hi"},
                )
                await queries.append_event(
                    conn,
                    account_id="acc_lea",
                    session_id=session.id,
                    kind="message",
                    data={"role": "assistant", "content": "yo"},
                )

            async with pool.acquire() as conn:
                sessions = await queries.list_sessions(conn, account_id="acc_lea")
            s = next(x for x in sessions if x.id == session.id)
            assert s.last_event_at is not None
            assert s.last_event_at >= s.created_at
        finally:
            await pool.close()
