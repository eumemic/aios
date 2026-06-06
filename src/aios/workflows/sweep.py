"""Periodic re-enqueue sweep for workflow runs — the durable backstop for a lost
run wake.

Mirrors the session sweep: every non-terminal run gets a deferred wake (deduped
by ``queueing_lock``, so an already-queued/running step is a no-op). A wake lost
to a crash therefore self-heals within one sweep interval, and a run suspended on
a gate whose resume signal landed while the wake was dropped still gets stepped.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.logging import get_logger
from aios.services.wake import defer_run_wake

log = get_logger("aios.workflows.sweep")


async def wake_runs_needing_step(pool: asyncpg.Pool[Any]) -> int:
    """Defer a wake for every non-terminal run. Returns the number swept."""
    async with pool.acquire() as conn:
        run_ids = await wf_queries.list_active_run_ids(conn)
    for run_id in run_ids:
        try:
            await defer_run_wake(run_id)
        except Exception:
            log.exception("wf_sweep.defer_failed", run_id=run_id)
    return len(run_ids)
