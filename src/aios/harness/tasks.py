"""Procrastinate task definitions.

``wake_session``
    The API process defers a wake job when a user posts a message; the
    sweep defers one when sessions need inference. A worker picks up
    the job and runs a single inference step.

    The lock and queueing_lock are NOT set on the decorator — they are
    passed per call from :mod:`aios.services.wake` (``defer_wake``).
    Procrastinate stores the decorator's lock arguments verbatim with
    no kwarg-template substitution, so a decorator-level
    ``lock="{session_id}"`` would assign every job the same literal
    lock value and serialize all sessions through one job slot — the
    bug behind issue #192.

    Per-call values used by :mod:`aios.services.wake`:

    * ``lock=session_id``: procrastinate enforces mutual exclusion via
      a partial unique index ``WHERE status='doing'``. With distinct
      values per session, only one step runs per session at a time
      while different sessions run concurrently up to
      ``worker_concurrency``. The lock releases the instant the job
      handler returns. On worker crash, procrastinate's heartbeat
      timeout (~30s) detects the stalled worker and marks the job as
      failed, freeing the lock.

    * ``queueing_lock=session_id``: dedups wake jobs in ``todo`` status
      *for the same session*. Multiple tool completions or user
      messages that arrive while a step is running produce at most one
      queued wake per session. Other sessions are unaffected.

    * ``retry=False``: failed jobs land in procrastinate's failed table.
      The periodic sweep catches sessions that need re-waking.
"""

from __future__ import annotations

from aios.harness.procrastinate_app import app


@app.task(
    name="harness.wake_session",
    queue="sessions",
    retry=False,
    pass_context=False,
)
async def wake_session(
    session_id: str,
    cause: str = "message",
) -> None:
    """Run one inference step for the session."""
    from aios.harness.loop import run_session_step

    await run_session_step(session_id, cause=cause)


@app.task(
    name="harness.run_scheduled_task",
    queue="sessions",
    retry=False,
    pass_context=False,
)
async def run_scheduled_task(task_id: str) -> None:
    """Fire one scheduled-task entry — runs bash in the session's sandbox.

    Per-task ``queueing_lock`` (set by the scheduler tick at defer time)
    deduplicates pending fires. No decorator-level ``lock`` — cron fires
    must not block concurrent session inference; overlap-prevention is
    enforced upstream via the ``running_since`` column on the row.
    """
    from aios.harness.scheduled_task_runner import run_scheduled_task_step

    await run_scheduled_task_step(task_id)
