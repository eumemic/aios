"""Event-driven scheduler for scheduled_tasks (#636).

Runs inside the worker process as a single async task. Instead of polling
on a fixed interval, it sleeps until either:

- The next-due ``next_fire`` instant elapses (computed via the cheap
  ``fetch_next_scheduled_task_event`` query on each loop iteration), OR
- The dedicated LISTEN connection delivers a NOTIFY on
  ``aios_scheduled_tasks_due`` (any insert/delete or scheduling-relevant
  UPDATE on ``session_scheduled_tasks`` — see the
  ``notify_scheduled_tasks_due`` trigger added in migration 0058), OR
- The cold-path heartbeat cap (``_HEARTBEAT_SECONDS``) elapses — bounds
  how long the LISTEN connection can sit idle before we cycle through a
  recompute, which is the only thing that detects a silently-dropped
  listener.

On wake, claims due tasks and defers ``harness.run_scheduled_task`` jobs;
the actual fire happens in :mod:`aios.harness.scheduled_task_runner`.

Overlap-prevention is correct-by-construction: the claim transaction
advances ``next_fire`` (cron) or leaves it (one-shot, deleted by runner)
and sets ``running_since`` so subsequent ticks skip the row until the
fire-handler clears it (or the stuck-recovery threshold elapses).
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import asyncpg
from procrastinate import exceptions as procrastinate_exceptions

from aios.db import queries
from aios.db.listen import listen_for_scheduled_tasks_due
from aios.harness.procrastinate_app import app
from aios.logging import get_logger

log = get_logger("aios.harness.scheduler")

# Cap on how long the scheduler will sleep without re-querying. Two
# guarantees:
#  1. A silently-dropped LISTEN connection (TCP RST, NAT timeout, etc.)
#     can't strand pending fires for more than this — we recompute MIN
#     and notice.
#  2. Edge cases that shouldn't NOTIFY but could theoretically affect
#     the next-due (e.g. ``sessions.archived_at`` flipped) get picked up
#     within this cap without needing trigger logic on that table.
_HEARTBEAT_SECONDS = 3600.0

# Backoff between LISTEN-connection reconnect attempts when
# ``listen_for_scheduled_tasks_due`` raises (network blip, Postgres
# failover, etc.). Capped low so a transient outage doesn't strand
# scheduled fires for long; cron rows still fire within
# `_HEARTBEAT_SECONDS` via the heartbeat path.
_LISTEN_RECONNECT_BACKOFF_SECONDS = 5.0


async def event_driven_scheduler(
    pool: asyncpg.Pool[Any],
    db_url: str,
) -> None:
    """Sleep-until-next-fire loop, woken on NOTIFY or timeout.

    Spawned once per worker. Wraps the LISTEN connection acquisition
    in an outer retry loop so a transient asyncpg failure (failover,
    network partition) doesn't permanently disable scheduled task firing
    on this worker — the inner loop's structured `try/except` covers
    transient SQL errors; this outer loop covers the LISTEN connection
    itself.

    Cancelled by the worker's shutdown sequence via task cancellation,
    which propagates ``CancelledError`` out of the awaits and ends the
    loop.
    """
    while True:
        try:
            async with listen_for_scheduled_tasks_due(db_url) as notify_event:
                await _scheduler_loop(pool, notify_event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # LISTEN connection or its acquisition failed. Sleep briefly
            # and retry — without this guard, a transient Postgres blip at
            # worker startup would silently disable all scheduled tasks
            # for the worker's lifetime while the heartbeat-touch keeps
            # the healthcheck green.
            log.exception("scheduler.listen_failed_will_retry")
            await asyncio.sleep(_LISTEN_RECONNECT_BACKOFF_SECONDS)


async def _scheduler_loop(
    pool: asyncpg.Pool[Any],
    notify_event: asyncio.Event,
) -> None:
    """Inner loop: compute next sleep, await NOTIFY-or-timeout, claim, repeat.

    The ``notify_event.clear()`` runs BEFORE ``_compute_sleep_seconds`` so
    a NOTIFY that arrives during the MIN-query window is preserved on the
    event and immediately wakes ``asyncio.wait_for``. Mirrors the
    LISTEN-before-backfill pattern documented in :mod:`aios.db.listen`.
    """
    while True:
        # Clear FIRST so any NOTIFY arriving after this line — including
        # during the SELECT MIN query below — remains on the event and
        # interrupts the subsequent wait_for.
        notify_event.clear()

        try:
            sleep_seconds = await _compute_sleep_seconds(pool)
        except Exception:
            # Defensive: if the MIN query fails (transient pool issue,
            # etc.), sleep on the heartbeat cadence and retry. We don't
            # want to hot-loop on a broken pool.
            log.exception("scheduler.compute_sleep_failed")
            sleep_seconds = _HEARTBEAT_SECONDS

        # Either NOTIFY arrives, the next fire's due-time elapses, or
        # the heartbeat caps the sleep — all three converge on the
        # same next action (recompute MIN, claim due, repeat).
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(notify_event.wait(), timeout=sleep_seconds)

        try:
            await _claim_and_enqueue_due_tasks(pool)
        except Exception:
            log.exception("scheduler.claim_error")


async def _compute_sleep_seconds(pool: asyncpg.Pool[Any]) -> float:
    """Return how long to sleep before the next wake, in seconds.

    Clamped to ``[0, _HEARTBEAT_SECONDS]``: a past-due ``next_fire`` (e.g.
    we just inserted a one-shot row in the past) yields ``0`` so the next
    claim happens immediately; a far-future ``next_fire`` is capped at
    the heartbeat for connection-resilience.
    """
    async with pool.acquire() as conn:
        next_event = await queries.fetch_next_scheduled_task_event(conn)
    if next_event is None:
        return _HEARTBEAT_SECONDS
    delta = (next_event - datetime.now(UTC)).total_seconds()
    return max(0.0, min(_HEARTBEAT_SECONDS, delta))


async def _claim_and_enqueue_due_tasks(pool: asyncpg.Pool[Any]) -> None:
    """Claim due rows in one transaction; defer a procrastinate job per row.

    If ``defer_async`` fails for a claimed row, the claim transaction has
    already committed (``running_since`` set, cron ``next_fire`` advanced).
    The compensating ``release_scheduled_task_claim`` clears
    ``running_since`` so the next iteration re-claims the row instead of
    leaving it stuck for ``stale_threshold_seconds`` (2h). For cron the
    next_fire advance is not reverted — one slot is effectively skipped,
    which is acceptable churn for a transient broker error.
    """
    now = datetime.now(UTC)
    async with pool.acquire() as conn, conn.transaction():
        claimed = await queries.fetch_and_claim_due_scheduled_tasks(conn, now_utc=now)
    for task in claimed:
        try:
            await app.configure_task(
                "harness.run_scheduled_task",
                queueing_lock=f"scheduled_task:{task.id}",
            ).defer_async(task_id=task.id)
        except procrastinate_exceptions.AlreadyEnqueued:
            log.debug("scheduler.already_enqueued", task_id=task.id, name=task.name)
        except Exception:
            log.exception("scheduler.defer_error", task_id=task.id, name=task.name)
            try:
                async with pool.acquire() as conn:
                    await queries.release_scheduled_task_claim(conn, task.id)
            except Exception:
                # Best-effort: if compensation itself fails, the row will
                # be picked up by stuck-recovery after the stale threshold.
                log.exception(
                    "scheduler.release_claim_failed",
                    task_id=task.id,
                    name=task.name,
                )
