"""Procrastinate task definitions.

Currently exactly one task: ``wake_session``. The api process calls
``defer_async`` to enqueue a wake job; a worker picks it up and runs the
session loop wrapped in lease management.

The task body is just a thin wrapper around
:func:`aios.harness.loop.run_session_turn_with_lease`. All the actual logic —
lease, refresh, fence, early-return guard, retry-on-lease-busy — lives in the
loop module so it's testable in isolation from procrastinate.

Task parameters worth understanding:

* ``name="harness.wake_session"``: explicit name. Never rely on procrastinate
  auto-deriving from the function qualname; renaming or moving the function
  would break on-disk job payloads.

* ``queue="sessions"``: the worker is configured to pull from this queue.
  Future phases can add additional queues for things like memory training
  jobs without affecting session processing.

* ``queueing_lock="{session_id}"``: procrastinate refuses to enqueue a second
  ``wake_session`` job for the same session while one is in ``todo``. This
  coalesces — posting two messages in quick succession yields one queued
  job, and the loop reads ALL pending message events from the DB so both
  user messages are processed in one turn. The lock releases when the job
  transitions from ``todo`` to ``doing``, so a follow-up message after the
  worker starts running the first job will produce a second queued job.

* ``lock="{session_id}"``: forbids two concurrent executions of the same
  session. Belt-and-suspenders with the DB lease — the lease is the source
  of truth, but ``lock`` blocks at the procrastinate layer too.

* ``retry=False``: failed tasks land in procrastinate's failed-jobs table
  for inspection. We handle the legitimate retry case (lease unavailable)
  via an explicit reschedule inside ``run_session_turn_with_lease``.

* ``pass_context=False``: we don't need procrastinate's job context in
  Phase 2. Phase 5 may flip this for span attributes.

The ``cause`` parameter is **observability-only**. The loop must be fully
recoverable from the events log alone; never branch behavior on ``cause``.
"""

from __future__ import annotations

from aios.harness.procrastinate_app import app


@app.task(
    name="harness.wake_session",
    queue="sessions",
    queueing_lock="{session_id}",
    lock="{session_id}",
    retry=False,
    pass_context=False,
)
async def wake_session(session_id: str, cause: str = "message") -> None:
    """Run one wake of the session loop. Body lives in harness.loop."""
    # Imported lazily so this module doesn't pull in litellm at app boot.
    from aios.harness.loop import run_session_turn_with_lease

    await run_session_turn_with_lease(session_id, cause=cause)
