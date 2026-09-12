"""DB-backed regression: the snapshot-reset-notice outbox retires archived sessions.

``unscoped_list_pending_snapshot_reset_notices`` / ``unscoped_deliver_pending_snapshot_reset_notice``
previously filtered on ``snapshot_reset_pending_reason IS NOT NULL AND
snapshot_reset_pending_ready`` with no ``archived_at IS NULL`` guard, so a
session whose marker became deliverable and was THEN archived kept being listed
and claimed for delivery. The claim's ``append_event`` fences on
``archived_at IS NULL`` and raises ``NotFoundError``; the delivery transaction
rolled back without clearing the marker; and the GC flush swallowed the error
while "leaving the marker intact" — retrying the same doomed delivery every
hourly tick forever and never retiring the outbox entry.

The fix guards the list/claim queries with ``archived_at IS NULL`` and retires
the marker when a delivery claim misses an archived session. These tests drive
the production query functions (and a simulated flush body) against the migrated
testcontainer — not a transcribed SQL copy — so mutating any conjunct in
``src/aios/db/queries/sandboxes.py`` is what turns them red.
"""

from __future__ import annotations

import contextlib
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_REASON = "snapshot_pool_pressure"


async def _seed_account(pool: asyncpg.Pool[Any], account_id: str) -> None:
    await pool.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ($1, NULL, TRUE, $2)",
        account_id,
        account_id,
    )


async def _arm_marker(pool: asyncpg.Pool[Any], session_id: str) -> None:
    """Bring a session to the post-pressure-reclaim state: pointer cleared,
    marker deliverable."""
    await pool.execute(
        """UPDATE sessions
              SET snapshot_ref = NULL,
                  snapshot_reset_pending_reason = $2,
                  snapshot_reset_pending_ready = TRUE
            WHERE id = $1""",
        session_id,
        _REASON,
    )


async def _marker(conn: asyncpg.Connection[Any], session_id: str) -> tuple[str | None, bool]:
    row = await conn.fetchrow(
        "SELECT snapshot_reset_pending_reason, snapshot_reset_pending_ready "
        "FROM sessions WHERE id = $1",
        session_id,
    )
    assert row is not None
    return (row["snapshot_reset_pending_reason"], bool(row["snapshot_reset_pending_ready"]))


async def _reset_events(
    conn: asyncpg.Connection[Any], session_id: str
) -> list[tuple[int, str, str | None]]:
    rows = await conn.fetch(
        "SELECT seq, kind, data->>'event' AS event FROM events WHERE session_id = $1 ORDER BY seq",
        session_id,
    )
    return [(r["seq"], r["kind"], r["event"]) for r in rows]


async def test_archived_deliverable_marker_is_not_listed(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """The lister excludes archived sessions: the lifecycle event could not be
    consumed (``append_event`` fences on ``archived_at IS NULL``), so listing it
    would only drive the flush into a permanent retry loop with no retire."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        await _seed_account(pool, "acc_retire")
        live_id = await _seed_live_session(pool, "acc_retire", "live")
        archived_id = await _seed_archived_session(pool, "acc_retire", "archived")
        async with pool.acquire() as conn:
            listed = await queries.unscoped_list_pending_snapshot_reset_notices(conn)
        assert listed == [(live_id, _REASON)]
        assert archived_id not in {sid for sid, _ in listed}
    finally:
        await pool.close()


async def test_deliver_returns_false_for_archived_session_without_raising(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """A delivery claim on an archived session returns ``False`` (no claim)
    rather than raising ``NotFoundError`` out of ``append_event``; the marker is
    left intact by the rolled-back transaction so a subsequent retire can clear
    it. Pre-fix this raised and propagated through the synchronous emit to the
    GC tick handler, and the flush swallowed it while looping forever."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        await _seed_account(pool, "acc_retire")
        archived_id = await _seed_archived_session(pool, "acc_retire", "archived")
        async with pool.acquire() as conn:
            before_seq = await conn.fetchval(
                "SELECT last_event_seq FROM sessions WHERE id = $1", archived_id
            )
            delivered = await queries.unscoped_deliver_pending_snapshot_reset_notice(
                conn, archived_id, expected_reason=_REASON
            )
            after_marker = await _marker(conn, archived_id)
            after_seq = await conn.fetchval(
                "SELECT last_event_seq FROM sessions WHERE id = $1", archived_id
            )
            events = await _reset_events(conn, archived_id)
        assert delivered is False
        # The rollback left the marker fully deliverable.
        assert after_marker == (_REASON, True)
        # No sequence was allocated and no event row was written.
        assert before_seq == after_seq
        assert events == []
    finally:
        await pool.close()


async def test_deliver_appends_event_and_clears_marker_for_live_session(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """No regression in the happy path: a live session's deliverable marker IS
    listed, the delivery claim appends exactly one ``sandbox_fs_reset``
    lifecycle event and clears the marker in the same transaction."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        await _seed_account(pool, "acc_retire")
        live_id = await _seed_live_session(pool, "acc_retire", "live")
        async with pool.acquire() as conn:
            before_seq = await conn.fetchval(
                "SELECT last_event_seq FROM sessions WHERE id = $1", live_id
            )
            delivered = await queries.unscoped_deliver_pending_snapshot_reset_notice(
                conn, live_id, expected_reason=_REASON
            )
            after_marker = await _marker(conn, live_id)
            after_seq = await conn.fetchval(
                "SELECT last_event_seq FROM sessions WHERE id = $1", live_id
            )
            events = await _reset_events(conn, live_id)
            listed = await queries.unscoped_list_pending_snapshot_reset_notices(conn)
        assert delivered is True
        assert after_marker == (None, False)
        assert after_seq == before_seq + 1
        assert events == [(after_seq, "lifecycle", "sandbox_fs_reset")]
        # The cleared marker is no longer deliverable, so the lister drops it.
        assert listed == []
    finally:
        await pool.close()


async def test_simulated_flush_ticks_never_retry_archived_and_retire_clears_marker(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """The perpetual flush loop is gone. Two simulated GC ticks (list -> for
    each, deliver with the flush's broad except-swallow) make NO delivery
    attempt on the archived row because the lister excludes it, so no
    ``NotFoundError`` is raised and no event is written. The retire path then
    clears the marker so the outbox entry does not survive forever."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        await _seed_account(pool, "acc_retire")
        archived_id = await _seed_archived_session(pool, "acc_retire", "archived")

        async def _flush_tick(conn: asyncpg.Connection[Any]) -> int:
            attempts = 0
            pending = await queries.unscoped_list_pending_snapshot_reset_notices(conn)
            for sid, reason in pending:
                attempts += 1
                # Mirrors the flush's broad ``except Exception`` swallow: a
                # permanently-undeliverable entry must not abort the tick.
                with contextlib.suppress(Exception):
                    await queries.unscoped_deliver_pending_snapshot_reset_notice(
                        conn, sid, expected_reason=reason
                    )
            return attempts

        async with pool.acquire() as conn:
            assert await _flush_tick(conn) == 0
            assert await _flush_tick(conn) == 0
            # No retry noise: the marker is still deliverable (no delivery
            # attempted) and no event was written.
            assert await _marker(conn, archived_id) == (_REASON, True)
            assert await _reset_events(conn, archived_id) == []
            # The retire path the emit now runs on a claim miss: prove it clears
            # the stuck marker so the outbox entry does not survive forever.
            archived = await conn.fetchval(
                "SELECT (archived_at IS NOT NULL) FROM sessions WHERE id = $1",
                archived_id,
            )
            assert archived is True
            retired = await queries.unscoped_clear_pending_snapshot_reset_notice(
                conn, archived_id, expected_reason=_REASON
            )
            assert retired is True
            assert await _marker(conn, archived_id) == (None, False)
            assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == []
    finally:
        await pool.close()


async def _seed_live_session(pool: asyncpg.Pool[Any], account_id: str, prefix: str) -> str:
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix=f"{prefix}-live"
    )
    await _arm_marker(pool, session.id)
    return session.id


async def _seed_archived_session(pool: asyncpg.Pool[Any], account_id: str, prefix: str) -> str:
    _agent, _env, session = await seed_agent_env_session(
        pool, account_id=account_id, prefix=f"{prefix}-arch"
    )
    await _arm_marker(pool, session.id)
    async with pool.acquire() as conn:
        await queries.archive_session(conn, session.id, account_id=account_id)
    return session.id
