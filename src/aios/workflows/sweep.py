"""Periodic re-enqueue sweep for workflow runs — the durable backstop for a lost
run wake.

Mirrors the session sweep, but with a **needs-step filter** (#780): only a run
with something for a step to do is woken — an unharvested signal, a stale
inflight call, or a non-parked status (the per-step ``running`` lease). A parked
run with nothing new is skipped, because every wake costs a full memo reship +
script replay; the old blanket every-non-terminal-run sweep made each tick O(memo)
per parked run. A wake lost to a crash still self-heals: every loss mode leaves
SQL-visible state one of the filter clauses matches (see
``list_run_ids_needing_step``) — within one sweep interval for signal-backed and
lease-backed modes, within the staleness horizon for a tool task that crashed
before writing its signal.
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
# failed). MUST dominate the slowest legitimate tool: tavily's 60s client
# timeout and http_request's 60s PER-READ timeout (a trickling chunked response
# can legitimately exceed 60s wall-clock). A tool still running when its run is
# re-woken is absorbed same-worker by the in-process task registry (one wasted
# wake) — but CROSS-worker the registry doesn't apply, and a premature horizon
# would double-dispatch a non-idempotent http_request. 180s clears the 60s
# bounds with margin while keeping crashed-task recovery within a few ticks.
TOOL_REDISPATCH_STALE_SECONDS = 180.0


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
