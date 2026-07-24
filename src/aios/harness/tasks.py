"""Procrastinate job definitions.

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

from aios.jobs.app import app


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
    name="harness.run_trigger",
    queue="sessions",
    retry=False,
    pass_context=False,
)
async def run_trigger(trigger_id: str, trigger_run_id: str | None = None) -> None:
    """Fire one trigger — runs its action (sandbox_command / wake_owner /
    workflow).

    Two defer origins, distinguished by ``trigger_run_id`` (the §1.2 additive
    per-event kwarg — old tick payloads still deserialize):

    - Scheduler tick (cron/one_shot): per-trigger ``queueing_lock`` set at
      defer time deduplicates pending fires; ``trigger_run_id`` is ``None``.
    - run_completion dispatch: one job per completed run, ``queueing_lock``
      keyed on the ``trigger_runs`` carrier row (never the bare trigger id —
      distinct completions must not coalesce); ``trigger_run_id`` names the
      carrier the runner claims.

    No decorator-level ``lock`` — fires must not block concurrent session
    inference; overlap-prevention is the ``running_since`` claim for tick
    fires and the carrier-row claim for event fires.

    Renamed from ``harness.run_scheduled_task`` (#818, delete-don't-deprecate).
    Deploy-window caveat: a job enqueued pre-restart under the old name
    fails job-lookup; the claimed row recovers via stale-recovery (~2h
    worst case, sub-second on a lockstep drain-and-restart).
    """
    from aios.harness.trigger_runner import run_trigger_step

    await run_trigger_step(trigger_id, trigger_run_id=trigger_run_id)


@app.task(
    name="harness.wake_workflow",
    queue="workflows",
    retry=False,
    pass_context=False,
)
async def wake_workflow(run_id: str) -> None:
    """Run one durable step of a workflow run.

    Per-call ``lock=run_id`` + ``queueing_lock=run_id`` (from
    :func:`aios.services.wake.defer_run_wake`) serialize a run's steps and dedup
    queued wakes — the same idiom as ``wake_session``.
    """
    from aios.workflows.step import run_workflow_step

    await run_workflow_step(run_id)


@app.task(
    name="harness.recycle_sandbox",
    queue="sessions",
    retry=False,
    pass_context=False,
)
async def recycle_sandbox(session_id: str, requested_by: str) -> None:
    """Discard every container/corpse and provision fresh current config."""
    from aios.harness import runtime
    from aios.services import sessions as sessions_service

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    registry = runtime.require_sandbox_registry()
    inflight = runtime.require_inflight_tool_registry()
    inflight.cancel_session(session_id)
    await registry.recycle(session_id)
    await registry.get_or_provision(session_id, pool=pool)
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": "sandbox_recycled", "requested_by": requested_by},
        account_id=account_id,
    )
