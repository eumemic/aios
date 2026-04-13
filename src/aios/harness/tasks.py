"""Procrastinate task definitions.

Two tasks:

``wake_session``
    The API process defers a wake job when a user posts a message; the async
    tool dispatcher defers one when a tool completes. A worker picks up the
    job and runs a single inference step.

    Key parameters:

    * ``lock="{session_id}"``: procrastinate enforces mutual exclusion via
      a DB unique index (``WHERE status = 'doing'``). Only one step runs
      per session at a time. The lock releases the instant the job handler
      returns. On worker crash, procrastinate's heartbeat timeout (~30s)
      detects the stalled worker and marks the job as failed, freeing the
      lock. This replaces the Phase 2 custom DB-row lease entirely.

    * ``queueing_lock="{session_id}"``: deduplicates wake jobs in ``todo``
      status. Multiple tool completions or user messages that arrive while
      a step is running produce at most one queued wake. The queued step
      reads ALL new events when it runs.

    * ``retry=False``: failed jobs land in procrastinate's failed table.
      The orphan recovery sweep catches sessions that need re-waking.

``orphan_sweep``
    Periodic task (every 60s) that finds sessions stuck in ``running``
    status with no active job and re-enqueues them. This closes the gap
    where orphan recovery only ran at worker startup — if worker A dies
    but worker B stays healthy, B now detects and reclaims A's orphans
    within ~60s instead of waiting for a restart.
"""

from __future__ import annotations

from aios.harness.procrastinate_app import app


@app.task(
    name="harness.wake_session",
    queue="sessions",
    lock="{session_id}",
    queueing_lock="{session_id}",
    retry=False,
    pass_context=False,
)
async def wake_session(session_id: str, cause: str = "message") -> None:
    """Run one inference step for the session."""
    from aios.harness.loop import run_session_step

    await run_session_step(session_id, cause=cause)


@app.periodic(cron="*/1 * * * *")
@app.task(name="harness.orphan_sweep", queue="sessions", retry=False, pass_context=False)
async def orphan_sweep(timestamp: int) -> None:
    """Periodic sweep for orphaned sessions."""
    from aios.harness import runtime
    from aios.harness.resume import recover_orphans

    pool = runtime.require_pool()
    await recover_orphans(pool, app)
