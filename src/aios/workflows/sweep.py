"""Periodic re-enqueue sweep for workflow runs — the durable backstop for a lost
run wake.

Mirrors the session sweep, but with a **needs-step filter** (#780): only a run
with something for a step to do is woken — an unharvested signal, a stale
inflight call, or a non-parked status (the per-step ``running`` lease). A parked
run with nothing new is skipped, because every wake costs a full memo reship +
script replay; the old blanket every-non-terminal-run sweep made each tick O(memo)
per parked run. A wake lost to a crash still self-heals within one sweep interval:
every loss mode leaves SQL-visible state one of the filter clauses matches (see
``list_run_ids_needing_step``).
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db.queries import workflows as wf_queries
from aios.logging import get_logger
from aios.services.wake import defer_run_wake

log = get_logger("aios.workflows.sweep")

# How long an inflight ``tool`` call may sit without a result signal before the
# sweep re-wakes its run for re-dispatch (the task crashed / its signal write
# failed). Re-dispatch is idempotent — the harvest point-reads the signal and
# checks the in-process task registry before launching — so the only cost of a
# false positive is one wasted wake. Run tools are network calls (seconds), so a
# minute of grace is generous.
TOOL_REDISPATCH_STALE_SECONDS = 60.0


async def wake_runs_needing_step(pool: asyncpg.Pool[Any]) -> int:
    """Defer a wake for every run needing a step. Returns the number swept."""
    async with pool.acquire() as conn:
        run_ids = await wf_queries.list_run_ids_needing_step(
            conn,
            agent_deadline_seconds=get_settings().workflow_agent_deadline_seconds,
            tool_stale_seconds=TOOL_REDISPATCH_STALE_SECONDS,
        )
    for run_id in run_ids:
        try:
            await defer_run_wake(run_id)
        except Exception:
            log.exception("wf_sweep.defer_failed", run_id=run_id)
    return len(run_ids)
