"""Integration test: ``set_session_stop_reason`` must not rewrite an
archived session's row (status is derived now; ``stop_reason`` is the only
mutable column it writes)."""

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
async def archived_marked_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session whose
    ``stop_reason`` is pinned to a distinctive value, then archived — giving
    the post-call read a known expected ``stop_reason``."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_sss_arch', NULL, TRUE, 'set-status-archived')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_sss_arch", prefix="sss-arch"
        )
        async with pool.acquire() as conn:
            await queries.set_session_stop_reason(
                conn, session.id, {"type": "rescheduling"}, account_id="acc_sss_arch"
            )
            archived = await queries.archive_session(conn, session.id, account_id="acc_sss_arch")
        assert archived.archived_at is not None
        assert archived.stop_reason == {"type": "rescheduling"}
        yield pool, "acc_sss_arch", session.id
    finally:
        await pool.close()


async def test_set_session_stop_reason_refuses_archived_silently(
    archived_marked_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = archived_marked_session

    async with pool.acquire() as conn:
        await queries.set_session_stop_reason(
            conn, session_id, {"type": "interrupt"}, account_id=account_id
        )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT stop_reason, archived_at FROM sessions WHERE id = $1",
            session_id,
        )
    assert row is not None
    assert row["stop_reason"] == {"type": "rescheduling"}, (
        f"archived row was rewritten: stop_reason is {row['stop_reason']!r}."
    )
    assert row["archived_at"] is not None


async def test_set_session_stop_reason_no_rewrite_on_archived(
    archived_marked_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The no-row UPDATE on an archived session must not bump ``updated_at``
    (and therefore must not have written anything)."""
    pool, account_id, session_id = archived_marked_session

    async with pool.acquire() as conn:
        pre_updated_at = await conn.fetchval(
            "SELECT updated_at FROM sessions WHERE id = $1", session_id
        )
        await queries.set_session_stop_reason(
            conn, session_id, {"type": "end_turn"}, account_id=account_id
        )
        post_updated_at = await conn.fetchval(
            "SELECT updated_at FROM sessions WHERE id = $1", session_id
        )

    assert pre_updated_at == post_updated_at
