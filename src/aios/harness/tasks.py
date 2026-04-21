"""Procrastinate task definitions.

``wake_session``
    The API process defers a wake job when a user posts a message; the
    sweep defers one when sessions need inference. A worker picks up
    the job and runs a single inference step.

    Key parameters:

    * ``lock="{session_id}"``: procrastinate enforces mutual exclusion via
      a DB unique index (``WHERE status = 'doing'``). Only one step runs
      per session at a time. The lock releases the instant the job handler
      returns. On worker crash, procrastinate's heartbeat timeout (~30s)
      detects the stalled worker and marks the job as failed, freeing the
      lock.

    * ``queueing_lock="{session_id}"``: deduplicates wake jobs in ``todo``
      status. Multiple tool completions or user messages that arrive while
      a step is running produce at most one queued wake.

    * ``retry=False``: failed jobs land in procrastinate's failed table.
      The periodic sweep catches sessions that need re-waking.
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
async def wake_session(
    session_id: str,
    cause: str = "message",
    wake_reason: str | None = None,
) -> None:
    """Run one inference step for the session."""
    from aios.harness.loop import run_session_step

    await run_session_step(session_id, cause=cause, wake_reason=wake_reason)
