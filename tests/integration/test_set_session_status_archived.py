"""Integration test: ``set_session_status`` must not rewrite an
archived session's row, and must skip the connector-calls fan-out
when the row is gone."""

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
async def archived_running_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session pinned
    to ``running`` then archived — gives the post-call read a
    distinctive expected ``status`` value."""
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
            await queries.set_session_status(
                conn, session.id, "running", None, account_id="acc_sss_arch"
            )
            archived = await queries.archive_session(conn, session.id, account_id="acc_sss_arch")
        assert archived.archived_at is not None
        assert archived.status == "running"
        yield pool, "acc_sss_arch", session.id
    finally:
        await pool.close()


async def test_set_session_status_refuses_archived_silently(
    archived_running_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, account_id, session_id = archived_running_session

    async with pool.acquire() as conn:
        await queries.set_session_status(
            conn, session_id, "idle", {"type": "interrupt"}, account_id=account_id
        )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, stop_reason, archived_at FROM sessions WHERE id = $1",
            session_id,
        )
    assert row is not None
    assert row["status"] == "running", (
        f"archived row was rewritten: status is {row['status']!r}, expected 'running'."
    )
    assert row["stop_reason"] is None
    assert row["archived_at"] is not None


async def test_set_session_status_skips_notify_on_archived(
    archived_running_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The ``requires_action`` fan-out must short-circuit on the no-row
    UPDATE — otherwise ``_list_bound_connection_ids`` runs and (with
    bindings) emits ``pg_notify`` for an archived session."""
    pool, account_id, session_id = archived_running_session

    async with pool.acquire() as conn:
        pre_updated_at = await conn.fetchval(
            "SELECT updated_at FROM sessions WHERE id = $1", session_id
        )
        await queries.set_session_status(
            conn,
            session_id,
            "idle",
            {"type": "requires_action", "custom_tools": [{"name": "x"}]},
            account_id=account_id,
        )
        post_updated_at = await conn.fetchval(
            "SELECT updated_at FROM sessions WHERE id = $1", session_id
        )

    assert pre_updated_at == post_updated_at
