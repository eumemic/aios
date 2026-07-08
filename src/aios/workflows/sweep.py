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
from aios.jobs.app import defer_run_wake
from aios.logging import get_logger

log = get_logger("aios.workflows.sweep")

# Slack added on top of the bash exec ceiling to cover the one-shot provisioning
# of an ephemeral scratch container (image pull / create / egress-CA / package
# install) before the exec's own wall-clock starts.
SANDBOX_PROVISIONING_SLACK_SECONDS = 180.0


def _sandbox_redispatch_horizon(bash_ceiling_seconds: int) -> float:
    """How long an inflight ``tool`` call may sit without a result signal before the
    sweep re-wakes its run for re-dispatch (#988, Option 1).

    ``bash`` rides the ``tool`` capability, so the ``tool`` stale-clause is the
    backstop that recovers a crashed bash exec. The horizon MUST exceed the maximum
    wall-clock a live bash exec can occupy — the bash exec ceiling
    (``settings.bash_default_timeout_seconds``, the SAME ceiling
    ``run_sandbox._execute`` clamps ``timeout_seconds`` to) PLUS provisioning slack — so
    the sweep never re-drives a still-running exec. Deriving the horizon from the
    ceiling keeps the two from drifting: an operator who raises
    ``bash_default_timeout_seconds`` automatically widens the horizon to match. The
    300s floor preserves the original 180s tool value for the 120s default
    (120 + 180 = 300), so the common case is unchanged. The worker network tools
    (web_*/http_request) inherit this wider horizon too — accepted: they are
    at-least-once idempotent and the horizon is a safety margin, not a deadline.

    Re-driving a still-running exec same-worker is cheap and harmless: the harvest's
    ``has_inflight`` guard suppresses a double-launch, sandbox is worker-pinned, and
    a run sandbox is ephemeral scratch so even a legitimate re-exec never corrupts
    durable state.
    """
    return max(300.0, bash_ceiling_seconds + SANDBOX_PROVISIONING_SLACK_SECONDS)


async def wake_runs_needing_step(pool: asyncpg.Pool[Any]) -> int:
    """Defer a wake for every run needing a step. Returns the number swept."""
    settings = get_settings()
    async with pool.acquire() as conn:
        run_ids = await wf_queries.list_run_ids_needing_step(
            conn,
            agent_deadline_seconds=settings.workflow_agent_deadline_seconds,
            # bash rides the `tool` capability, so the tool stale-clause covers it —
            # widened to the sandbox horizon (#988, Option 1).
            tool_stale_seconds=_sandbox_redispatch_horizon(settings.bash_default_timeout_seconds),
            # call_llm is worker-task-backed like `tool`/`agent`: a crash mid-inference
            # leaves no signal and no external resume, so it needs the stale backstop to
            # re-wake and re-dispatch (#1706).
            call_llm_stale_seconds=settings.workflow_call_llm_stale_seconds,
        )
    for run_id in run_ids:
        try:
            await defer_run_wake(run_id)
        except Exception:
            log.exception("wf_sweep.defer_failed", run_id=run_id)
    return len(run_ids)
