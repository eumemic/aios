"""Periodic scheduler-tick loop for scheduled_tasks (#636).

Runs every ``interval`` seconds inside the worker process. Each tick
atomically claims due tasks from ``session_scheduled_tasks`` and
enqueues a procrastinate ``harness.run_scheduled_task`` job per claimed
row. The actual fire happens in :mod:`aios.harness.scheduled_task_runner`.

Overlap-prevention is correct-by-construction: the claim transaction
advances ``next_fire`` and sets ``running_since`` so subsequent ticks
skip the row until the fire-handler clears ``running_since`` (or the
stuck-recovery threshold elapses).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import asyncpg
from procrastinate import exceptions as procrastinate_exceptions

from aios.db import queries
from aios.harness.procrastinate_app import app
from aios.logging import get_logger

log = get_logger("aios.harness.scheduler")


async def periodic_scheduler_tick(
    pool: asyncpg.Pool[Any],
    *,
    interval: float = 30.0,
) -> None:
    """Background loop: every ``interval`` seconds claim and enqueue due tasks.

    Each iteration is best-effort — exceptions are logged and the loop
    continues. Cancelled by the worker's shutdown sequence via task
    cancellation, which surfaces as :class:`asyncio.CancelledError`
    propagating out of the ``await asyncio.sleep`` and ending the loop.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await _claim_and_enqueue_due_tasks(pool)
        except Exception:
            log.exception("scheduler.tick_error")


async def _claim_and_enqueue_due_tasks(pool: asyncpg.Pool[Any]) -> None:
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
