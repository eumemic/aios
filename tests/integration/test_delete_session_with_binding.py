"""Integration test: deleting a session that has a ``bindings`` row must
succeed — and the cascade is enforced by the schema, not by application
code.

``bindings.session_id`` originally (migration 0015) declared ``ON DELETE
CASCADE``; the 0033 connector redesign recreated ``bindings`` and dropped
it, leaving a bare ``session_id text REFERENCES sessions(id)``.
``delete_session`` compensated with an explicit hand-``DELETE FROM
bindings`` — the lone session-child held by application vigilance rather
than the schema.

Migration 0109 restores the cascade (in the single-column form
``session_id REFERENCES sessions(id) ON DELETE CASCADE`` — the original
0015 shape) and ``delete_session`` no longer pre-deletes from
``bindings``.

Two guarantees are tested:

* ``delete_session`` still succeeds and leaves no binding rows for a
  session that was attached to a connection (regression coverage for the
  route-500 the hand-DELETE used to prevent).
* The cascade is enforced at the DB level: a *raw* ``DELETE FROM
  sessions`` — bypassing ``delete_session`` entirely, simulating any new
  session-deletion path that doesn't replicate the old hand-DELETE —
  succeeds and removes the binding row. This is the correct-by-
  construction property the migration buys.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


async def _seed_session_with_binding(
    pool: asyncpg.Pool[Any],
) -> str:
    """Create an acc_a session with an active ``bindings`` row tying it to
    an acc_a connection. Returns the session id."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a')
            """
        )

    agent_a = await agents_service.create_agent(
        pool,
        account_id="acc_a",
        name="a-agent",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env_a = await environments_service.create_environment(
        pool,
        account_id="acc_a",
        name="a-env",
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id="acc_a",
            agent_id=agent_a.id,
            environment_id=env_a.id,
            agent_version=agent_a.version,
            title=None,
            metadata={},
        )
        connection = await queries.insert_connection(
            conn,
            account_id="acc_a",
            connector="signal",
            external_account_id="+15550001",
            metadata={},
        )
        await queries.insert_binding(
            conn,
            account_id="acc_a",
            connection_id=connection.id,
            mode="single_session",
            session_id=session.id,
        )
    return session.id


@pytest.fixture
async def pool_session_with_binding(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, session_id)`` for an acc_a session that has an
    active ``bindings`` row tying it to an acc_a connection."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        session_id = await _seed_session_with_binding(pool)
        yield pool, session_id
    finally:
        await pool.close()


class TestDeleteSessionWithBinding:
    async def test_delete_session_removes_bindings(
        self,
        pool_session_with_binding: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """``delete_session`` succeeds and cleans up the binding row even
        when the session has been attached to a connection."""
        pool, session_id = pool_session_with_binding
        async with pool.acquire() as conn:
            await queries.delete_session(conn, session_id, account_id="acc_a")
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM bindings WHERE session_id = $1",
                session_id,
            )
        assert remaining == 0, (
            f"delete_session left {remaining} binding row(s) referencing the deleted session"
        )

    async def test_raw_session_delete_cascades_to_bindings(
        self,
        pool_session_with_binding: tuple[asyncpg.Pool[Any], str],
    ) -> None:
        """A *raw* ``DELETE FROM sessions`` — bypassing ``delete_session``
        and any application-level hand-DELETE — succeeds and removes the
        binding row.

        This is the correct-by-construction property restored by migration
        0109: the cascade is enforced by Postgres regardless of which code
        path deletes the session. Pre-fix this raw DELETE raised
        ``asyncpg.ForeignKeyViolationError`` because ``bindings.session_id``
        had no ``ON DELETE CASCADE``.
        """
        pool, session_id = pool_session_with_binding
        async with pool.acquire() as conn:
            # No hand-DELETE from bindings — rely purely on the schema.
            await conn.execute(
                "DELETE FROM sessions WHERE id = $1 AND account_id = $2",
                session_id,
                "acc_a",
            )
            remaining_bindings = await conn.fetchval(
                "SELECT COUNT(*) FROM bindings WHERE session_id = $1",
                session_id,
            )
            remaining_sessions = await conn.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE id = $1",
                session_id,
            )
        assert remaining_sessions == 0, "session row not deleted"
        assert remaining_bindings == 0, (
            f"raw DELETE FROM sessions left {remaining_bindings} binding "
            f"row(s) — ON DELETE CASCADE on bindings.session_id is missing"
        )
