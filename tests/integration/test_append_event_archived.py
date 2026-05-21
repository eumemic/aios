"""Integration test: ``queries.append_event`` must refuse to append
events to archived sessions.

Pre-fix the UPDATE WHERE clause in ``append_event`` only filtered
``id = $1 AND account_id = $2`` — archived rows still match, so the
session row's ``last_event_seq`` is incremented and the event INSERT
succeeds. The wake-sweep (``find_sessions_needing_inference`` in
``harness/sweep.py``) DOES filter ``archived_at IS NULL``, so the model
never wakes. Result: ``POST /v1/sessions/{id}/messages`` returns 201
Created, ``defer_wake`` enqueues, but the message vanishes into the
archived session — no inference, no error, no operator-visible signal
that the post was lost.

Same defect class as PR #521 (archived-connection inbound), one layer
deeper: the SESSION itself is archived, not the connection. The
root-cause fix is at the deepest layer — ``append_event``'s UPDATE
WHERE — so it covers EVERY caller path (direct API ``post_message``,
connector inbound via resolver tier-2/3 ``single_session`` and
``target_type='session'``, internal tool result handling, etc.) in one
guard rather than gating each call site individually.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session that has
    been archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_archived', NULL, TRUE, 'archived-session-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_archived", prefix="archived-test"
        )
        async with pool.acquire() as conn:
            await queries.archive_session(conn, session.id, account_id="acc_archived")
        yield pool, "acc_archived", session.id
    finally:
        await pool.close()


class TestAppendEventArchivedSession:
    async def test_append_event_refuses_archived_session(
        self,
        archived_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The UPDATE in ``append_event`` must filter
        ``archived_at IS NULL`` so an attempt to append to an archived
        session raises ``NotFoundError`` instead of silently
        succeeding with no downstream wake."""
        pool, account_id, session_id = archived_session

        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.append_event(
                    conn,
                    account_id=account_id,
                    session_id=session_id,
                    kind="message",
                    data={"role": "user", "content": "should be refused"},
                )
