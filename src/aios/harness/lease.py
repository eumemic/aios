"""Row-level lease on the sessions table.

Phase 2 splits the runtime into an api process (stateless, defers wake jobs)
and a worker process (drains the queue). The lease ensures that exactly one
worker is advancing a given session at a time.

The lease lives on two columns of the ``sessions`` row, both added in Phase 1's
migration 0001:

* ``lease_worker_id text`` — the worker process holding the lease, or NULL
* ``lease_expires_at timestamptz`` — when the lease expires, or NULL

The protocol is:

1. **Acquire**: an UPDATE that succeeds if the lease is unheld, expired, or
   already ours. The acquire also flips ``status`` to ``running``.
2. **Refresh**: a background asyncio task that extends ``lease_expires_at``
   periodically. If the refresh affects 0 rows, the lease has been stolen
   (or expired before we could refresh) and the loop must unwind.
3. **Release**: an UPDATE that clears the lease columns and sets ``status``
   back to whatever the loop ended at (typically ``idle``). If the release
   affects 0 rows, the lease was already stolen — log and move on.
4. **Fenced append**: a variant of ``append_event`` that requires the
   session's current ``lease_worker_id`` to match ours. Used by the harness
   for assistant messages, lifecycle events, and span events. The plain
   ``append_event`` (no fence) is still used by the API for user messages —
   user input must always land regardless of lease state.

The fence is the safety net for "the refresh task crashed and the loop kept
going". Without it, two workers could append to the same session's event log
concurrently. With it, the second worker's appends raise :class:`LeaseLost`
and unwind cleanly.

Phase 2 only checks the cancel event between iterations of the loop, not
mid-LiteLLM-call. Phase 5 will thread the cancel event into the LiteLLM
wrapper for true mid-call cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import asyncpg

from aios.config import get_settings
from aios.ids import EVENT, make_id
from aios.logging import get_logger
from aios.models.events import Event, EventKind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger("aios.harness.lease")


class LeaseLost(Exception):
    """Raised when a lease-required operation finds the lease is no longer ours.

    Carries the session id and the worker id we expected to hold it.
    """

    def __init__(self, session_id: str, worker_id: str) -> None:
        super().__init__(f"lease lost on session {session_id} (expected worker {worker_id})")
        self.session_id = session_id
        self.worker_id = worker_id


# ─── acquire / release ────────────────────────────────────────────────────────


async def acquire_lease(
    pool: asyncpg.Pool[Any],
    session_id: str,
    worker_id: str,
) -> int | None:
    """Acquire the lease on ``session_id`` for ``worker_id``.

    Returns the session's ``last_event_seq`` on success, or ``None`` if
    another worker holds an unexpired lease (caller should reschedule).

    Atomically flips ``status`` to ``running`` on success.
    """
    settings = get_settings()
    duration = settings.lease_duration_seconds
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE sessions
               SET lease_worker_id = $1,
                   lease_expires_at = now() + interval '{duration} seconds',
                   status = 'running',
                   updated_at = now()
             WHERE id = $2
               AND archived_at IS NULL
               AND (lease_worker_id IS NULL
                    OR lease_worker_id = $1
                    OR lease_expires_at < now())
            RETURNING last_event_seq
            """,
            worker_id,
            session_id,
        )
    if row is None:
        return None
    return int(row["last_event_seq"])


async def release_lease(
    pool: asyncpg.Pool[Any],
    session_id: str,
    worker_id: str,
    *,
    new_status: str = "idle",
    stop_reason: str | None = None,
) -> bool:
    """Release the lease and set the session's terminal status.

    Returns True on a clean release, False if the lease was already stolen
    by another worker. A False return is logged but is not an error: the new
    lease holder is responsible for the session now.
    """
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
               SET lease_worker_id = NULL,
                   lease_expires_at = NULL,
                   status = $1,
                   stop_reason = $2,
                   updated_at = now()
             WHERE id = $3
               AND lease_worker_id = $4
            """,
            new_status,
            stop_reason,
            session_id,
            worker_id,
        )
    rows_affected = int(result.rsplit(" ", 1)[-1])
    if rows_affected == 0:
        log.warning(
            "lease.release_no_op",
            session_id=session_id,
            worker_id=worker_id,
            reason="lease was already stolen or cleared",
        )
        return False
    return True


# ─── refresh task ─────────────────────────────────────────────────────────────


async def refresh_lease_loop(
    pool: asyncpg.Pool[Any],
    session_id: str,
    worker_id: str,
    cancel: asyncio.Event,
) -> None:
    """Background task that periodically extends the lease.

    Sleeps for ``lease_refresh_seconds``, then runs an UPDATE that requires
    the lease still belong to us AND not be expired. If the UPDATE affects
    zero rows, the lease has been lost — set the cancel event and return.

    The cancel event is also a poll-out: if it's set externally (e.g., the
    main loop completed), this task wakes up and exits cleanly.
    """
    settings = get_settings()
    interval = settings.lease_refresh_seconds
    duration = settings.lease_duration_seconds

    while not cancel.is_set():
        try:
            # Sleep, but wake immediately if cancel is set externally.
            await asyncio.wait_for(cancel.wait(), timeout=interval)
            return  # cancel was set during sleep — clean exit
        except TimeoutError:
            pass

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE sessions
                   SET lease_expires_at = now() + interval '{duration} seconds'
                 WHERE id = $1
                   AND lease_worker_id = $2
                   AND lease_expires_at > now()
                """,
                session_id,
                worker_id,
            )
        rows_affected = int(result.rsplit(" ", 1)[-1])
        if rows_affected == 0:
            log.warning(
                "lease.refresh_lost",
                session_id=session_id,
                worker_id=worker_id,
            )
            cancel.set()
            return


@asynccontextmanager
async def lease_refresher(
    pool: asyncpg.Pool[Any],
    session_id: str,
    worker_id: str,
    cancel: asyncio.Event,
) -> AsyncIterator[asyncio.Task[None]]:
    """Run :func:`refresh_lease_loop` as a background task for the duration
    of the context.

    On exit, cancels the refresh task and awaits its termination so the
    asyncpg connection it borrowed is released cleanly.
    """
    task = asyncio.create_task(
        refresh_lease_loop(pool, session_id, worker_id, cancel),
        name=f"lease-refresh:{session_id}",
    )
    try:
        yield task
    finally:
        if not cancel.is_set():
            cancel.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


# ─── fenced event append ──────────────────────────────────────────────────────


async def append_event_with_fence(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    expected_worker_id: str,
    kind: EventKind,
    data: dict[str, Any],
) -> Event:
    """Like :func:`aios.db.queries.append_event` but refuses to write if the
    session's current ``lease_worker_id`` is not ``expected_worker_id``.

    Used by the harness for assistant messages, lifecycle events, and span
    events. The plain unfenced append is still used by the API for user
    messages — user input always lands regardless of lease state.

    Raises :class:`LeaseLost` if the fence check fails.
    """
    new_id = make_id(EVENT)
    data_json = json.dumps(data)

    async with pool.acquire() as conn:
        async with conn.transaction():
            seq_row = await conn.fetchrow(
                """
                UPDATE sessions
                   SET last_event_seq = last_event_seq + 1
                 WHERE id = $1
                   AND lease_worker_id = $2
                RETURNING last_event_seq
                """,
                session_id,
                expected_worker_id,
            )
            if seq_row is None:
                raise LeaseLost(session_id, expected_worker_id)
            seq = int(seq_row["last_event_seq"])
            row = await conn.fetchrow(
                """
                INSERT INTO events (id, session_id, seq, kind, data)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING *
                """,
                new_id,
                session_id,
                seq,
                kind,
                data_json,
            )
            assert row is not None
        # NOTIFY happens outside the transaction so subscribers don't see it
        # until the row is committed (matches Phase 1 append_event semantics).
        await conn.execute("SELECT pg_notify($1, $2)", f"events_{session_id}", new_id)

    raw_data = row["data"]
    parsed = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    return Event(
        id=row["id"],
        session_id=row["session_id"],
        seq=row["seq"],
        kind=row["kind"],
        data=parsed,
        created_at=row["created_at"],
    )
