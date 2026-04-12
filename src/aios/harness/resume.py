"""Orphan recovery for sessions left in a half-state by a crashed worker.

Runs once at worker startup, before the worker begins draining the queue.
Finds sessions that meet ALL of:

1. ``status = 'running'``
2. ``archived_at IS NULL``
3. The most recent event is NOT a ``turn_ended`` lifecycle

Condition 3 prevents re-enqueuing sessions that legitimately finished
but whose status update never ran (worker killed between event append
and the status UPDATE). Those sessions are actually done; the next
user message wakes them via the normal path.

For each orphan, recovery sets status to ``idle`` and defers a fresh
``wake_session(cause='resume')`` job. The defer swallows
``AlreadyEnqueued`` because two workers racing recovery may both try
to enqueue the same session.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from procrastinate import App
from procrastinate import exceptions as procrastinate_exceptions

from aios.logging import get_logger

log = get_logger("aios.harness.resume")


async def recover_orphans(pool: asyncpg.Pool[Any], app: App) -> int:
    """Find orphaned sessions and re-enqueue wake jobs for each."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.id
              FROM sessions s
             WHERE s.status = 'running'
               AND s.archived_at IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM events e
                    WHERE e.session_id = s.id
                      AND e.seq = s.last_event_seq
                      AND e.kind = 'lifecycle'
                      AND e.data->>'event' = 'turn_ended'
               )
            """
        )

    if not rows:
        log.info("recover_orphans.none")
        return 0

    recovered = 0
    for row in rows:
        sid = row["id"]
        # Reset status so the step function can set it to running.
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET status = 'idle', updated_at = now() WHERE id = $1",
                sid,
            )
        try:
            await app.configure_task("harness.wake_session").defer_async(
                session_id=sid,
                cause="resume",
            )
            recovered += 1
            log.info("recover_orphans.deferred", session_id=sid)
        except procrastinate_exceptions.AlreadyEnqueued:
            log.info("recover_orphans.already_enqueued", session_id=sid)

    log.info("recover_orphans.summary", recovered=recovered, total_orphans=len(rows))
    return recovered
