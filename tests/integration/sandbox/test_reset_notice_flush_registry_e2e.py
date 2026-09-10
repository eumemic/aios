"""E2E-through-the-registry regression: archived notices are retired, not retried.

Drives the production GC-tick entry points
(``SandboxRegistry._flush_pending_snapshot_reset_notices`` and
``_emit_pending_snapshot_reset_notice``) against a migrated Postgres
testcontainer with a real asyncpg pool, exercising the actual code paths the
hourly GC tick and the synchronous reclaim emit run — not hand-simulated bodies.

The archived+armed session state is the post-pressure-reclaim terminal state
the pre-fix code retried every tick forever (``NotFoundError`` swallowed,
marker left intact). Post-fix the behavior splits cleanly along the two emit
entry points:

* **Flush path** (crash recovery / retry): the lister now excludes archived
  rows, so ``_flush_pending_snapshot_reset_notices`` makes NO delivery
  attempt on the archived row across two ticks — no ``NotFoundError``, no
  ``sandbox.snapshot_reset_notice_retry_failed`` retry noise, and the
  process-local mirror is not repopulated. The archived row's marker stays
  armed but is inert (no consumer reads it; no clone inherits it).

* **Synchronous-emit path** (the reclaim race: marker armed, session archived
  before the claim): calling ``_emit_pending_snapshot_reset_notice`` directly
  — the exact path ``_reclaim_pool_candidate`` takes after clearing the
  pointer — claims nothing (the claim's ``archived_at IS NULL`` guard returns
  ``False`` without raising), detects the archive, and retires the marker
  via ``unscoped_clear_pending_snapshot_reset_notice`` in the same pool scope.
  No lifecycle event is written; no exception propagates to a tick handler.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.sandbox.registry import SandboxRegistry
from tests.helpers.sandbox import FakeBackend
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


async def _arm_marker_and_archive(
    pool: asyncpg.Pool[Any], session_id: str, account_id: str
) -> None:
    """Bring the session to the post-pressure-reclaim state (pointer cleared,
    marker deliverable) and then archive it — the race's terminal state."""
    await pool.execute(
        """UPDATE sessions
              SET snapshot_ref = NULL,
                  snapshot_reset_pending_reason = $2,
                  snapshot_reset_pending_ready = TRUE
            WHERE id = $1""",
        session_id,
        _REASON,
    )
    async with pool.acquire() as conn:
        await queries.archive_session(conn, session_id, account_id=account_id)


async def _marker(conn: asyncpg.Connection[Any], session_id: str) -> tuple[str | None, bool]:
    row = await conn.fetchrow(
        "SELECT snapshot_reset_pending_reason, snapshot_reset_pending_ready "
        "FROM sessions WHERE id = $1",
        session_id,
    )
    assert row is not None
    return (row["snapshot_reset_pending_reason"], bool(row["snapshot_reset_pending_ready"]))


async def test_registry_flush_excludes_archived_and_sync_emit_retires_marker(
    migrated_db_url: str, _reset_db_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ``_flush_pending_snapshot_reset_notices`` ticks make no delivery
    attempt on the archived row (no retry noise), and a direct
    ``_emit_pending_snapshot_reset_notice`` (the synchronous-emit path) retires
    the marker via the real claim-miss + archived-check + clear sequence."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    prev_runtime_pool = runtime.pool
    runtime.pool = cast(Any, pool)
    try:
        await _seed_account(pool, "acc_flush_e2e")
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_flush_e2e", prefix="flush-e2e"
        )
        await _arm_marker_and_archive(pool, session.id, "acc_flush_e2e")

        # The retry-failed log fires only when emit RAISES into the flush's broad
        # ``except Exception``. Post-fix the lister excludes the archived row, so
        # emit is never called for it and the log MUST NOT fire across two ticks.
        retry_failed = Mock()
        monkeypatch.setattr("aios.sandbox.registry.log.exception", retry_failed, raising=False)

        async with pool.acquire() as conn:
            assert await _marker(conn, session.id) == (_REASON, True)

        registry = SandboxRegistry(backend=FakeBackend())  # flush/emit are pool-only
        assert registry._pending_snapshot_reset_notices == {}

        # ── Flush path: the lister excludes the archived row ──
        await registry._flush_pending_snapshot_reset_notices()  # tick 1
        await registry._flush_pending_snapshot_reset_notices()  # tick 2

        retry_failed.assert_not_called()
        async with pool.acquire() as conn:
            # No delivery was attempted, so the marker is still armed — but it is
            # inert (excluded from the lister, so no further retry).
            assert await _marker(conn, session.id) == (_REASON, True)
            assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == []
            assert (
                await conn.fetchval("SELECT count(*) FROM events WHERE session_id = $1", session.id)
                == 0
            )
        # The flush repopulated no process-local mirror (nothing was listed).
        assert registry._pending_snapshot_reset_notices == {}

        # ── Synchronous-emit path: the reclaim race, driven directly ──
        # _reclaim_pool_candidate calls this exact method after arming the
        # marker; in the race the session is archived between the fresh re-read
        # and the claim. Here the session is already archived, which the claim
        # guard treats identically (returns False without raising).
        await registry._emit_pending_snapshot_reset_notice(session.id, _REASON)

        async with pool.acquire() as conn:
            # The claim missed (archived guard), the archived check ran, and the
            # retire cleared the marker in the same pool scope.
            assert await _marker(conn, session.id) == (None, False)
            assert await queries.unscoped_list_pending_snapshot_reset_notices(conn) == []
            # No lifecycle event was written for the archived session.
            assert (
                await conn.fetchval("SELECT count(*) FROM events WHERE session_id = $1", session.id)
                == 0
            )
        assert session.id not in registry._pending_snapshot_reset_notices
    finally:
        runtime.pool = prev_runtime_pool
        await pool.close()
