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

``recycle_sandbox``
    The one task here that does NOT use ``retry=False``: it is destructive,
    rate-limited, and has no sweep to re-drive it, so a transient failure must
    not permanently consume an admitted request. It retries with bounded
    backoff and records a typed ``sandbox_recycle_failed`` lifecycle event when
    the budget is exhausted. See the comment above the task.
"""

from __future__ import annotations

from procrastinate import JobContext, RetryStrategy
from procrastinate.jobs import Job

from aios.jobs.app import app
from aios.logging import get_logger
from aios.models.events import SANDBOX_RECYCLE_FAILED_EVENT, SANDBOX_RECYCLED_EVENT

log = get_logger("aios.harness.tasks")


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


# Recycle is DESTRUCTIVE and rate-limited: an admitted request must not be
# silently consumed by a transient Docker/proxy/DB hiccup. So unlike the other
# harness tasks (``retry=False`` — the sweep re-drives them), the recycle job
# retries with a bounded exponential backoff and, on the genuinely FINAL run,
# records a typed terminal failure event so the outcome is durable and
# redrivable.
#
# Every stage is convergent, so a retry is safe:
#   * ``inflight.cancel_session`` — idempotent (no in-flight tasks ⇒ no-op),
#   * ``registry.recycle`` — removes containers/artifacts to a goal state
#     ("gone"); a partial first attempt just finds less to do the second time,
#     and the artifact-before-pointer ordering means a crash mid-way never
#     leaves an unattributable orphan (see ``SandboxRegistry.recycle``),
#   * ``registry.get_or_provision`` — a normal cold provision.
#
# ── Attempt accounting (procrastinate 3.8.1) ───────────────────────────────
# The terminal event must be emitted EXACTLY ONCE, on the run after which no
# further retry is scheduled. Getting that predicate right requires reading the
# library rather than guessing, because ``job.attempts`` is NOT "runs so far":
#
#   * ``procrastinate_fetch_job_v2`` does NOT touch ``attempts``. A job's very
#     first run therefore observes ``job.attempts == 0``.
#   * On failure the worker asks ``RetryStrategy.get_retry_decision()``, which
#     returns ``None`` (⇒ no retry, job goes to ``failed``) iff
#     ``max_attempts and job.attempts >= max_attempts`` — i.e. it keeps
#     retrying while ``job.attempts < max_attempts``, comparing the value the
#     RUNNING job was fetched with.
#   * Only when a retry IS scheduled does ``procrastinate_retry_job_v*`` do
#     ``attempts = attempts + 1`` (the in-memory testing connector mirrors
#     this). ``procrastinate_finish_job_v1`` also increments, but that is the
#     bookkeeping write for a run that is already terminal.
#
# So the observed sequence of ``context.job.attempts`` across the ladder for
# ``max_attempts = N`` is 0, 1, 2, ... N, i.e. N + 1 runs, and the LAST one —
# the only one after which no retry is scheduled — is the one that observes
# ``job.attempts == N``.
#
# The previous implementation computed ``attempt = job.attempts + 1`` and
# declared terminal at ``attempt >= N``, i.e. at ``job.attempts == N - 1``.
# That run still satisfies ``job.attempts < max_attempts``, so procrastinate
# scheduled ANOTHER run, which then also saw the terminal predicate satisfied:
# two ``sandbox_recycle_failed`` events per exhausted request, the first one
# premature (a retry that might still have succeeded was already journalled as
# a terminal failure).
#
# The predicate below is therefore derived from the strategy object itself
# (:func:`_is_final_attempt`) rather than restating the ``+ 1`` arithmetic, so
# it stays true to whatever the configured strategy actually decides — and the
# strategy is consulted through ``get_retry_decision`` with the real exception,
# which also honours ``retry_exceptions`` filtering.
_RECYCLE_MAX_ATTEMPTS = 4

recycle_retry = RetryStrategy(
    max_attempts=_RECYCLE_MAX_ATTEMPTS,
    exponential_wait=2,  # 2s, 4s, 8s
)

# Total number of RUNS the ladder above performs before the job is failed for
# good: the initial run (``attempts == 0``) plus ``max_attempts`` retries.
_RECYCLE_MAX_RUNS = _RECYCLE_MAX_ATTEMPTS + 1


def _is_final_attempt(job: Job, exc: BaseException) -> bool:
    """Return True iff procrastinate will NOT schedule another run of *job*.

    Asks the very strategy the task is registered with, with the very exception
    that was raised — the same call ``procrastinate.worker.Worker._process_job``
    makes via ``Task.get_retry_exception`` — so this predicate cannot drift from
    the library's decision (nor from ``retry_exceptions`` filtering, should the
    strategy ever grow one). ``None`` ⇒ no retry ⇒ this run is terminal.
    """
    return recycle_retry.get_retry_decision(exception=exc, job=job) is None


@app.task(
    name="harness.recycle_sandbox",
    queue="sessions",
    retry=recycle_retry,
    pass_context=True,
)
async def recycle_sandbox(
    context: JobContext,
    session_id: str,
    requested_by: str,
    request_id: str | None = None,
) -> None:
    """Discard every container/corpse/snapshot and provision fresh current config.

    Emits exactly one terminal lifecycle event per admitted request:
    ``sandbox_recycled`` on success, or ``sandbox_recycle_failed`` on the run
    after which procrastinate schedules no further retry — so the 202 the caller
    already received always resolves to an observable outcome in the journal
    rather than dead-ending at ``sandbox_recycle_requested``. The failure event
    carries ``error`` and ``attempts`` and is state a caller/operator can redrive
    from (the admission limit is per accepted request, so a redrive is a new
    request; the failure event is what tells anyone it is needed).

    Terminality is decided by :func:`_is_final_attempt` — the configured
    ``RetryStrategy`` itself — never by re-deriving attempt arithmetic here; see
    the attempt-accounting note above the strategy for why the naive
    ``attempts + 1 >= max_attempts`` form double-emitted.

    ``request_id`` is the id of the admitting ``sandbox_recycle_requested``
    event (see :func:`aios.jobs.app.defer_sandbox_recycle`); it is stamped onto
    the terminal event so each admitted, rate-limit-counted request is joinable
    to exactly one outcome. Optional/defaulted so a job deferred by a previous
    release still deserializes across the deploy window.
    """
    from aios.harness import runtime
    from aios.services import sessions as sessions_service

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    # ``job.attempts`` counts retries ALREADY SCHEDULED for this job, so the
    # first run sees 0; the human-facing run counter is one more than that.
    run_number = context.job.attempts + 1
    try:
        registry = runtime.require_sandbox_registry()
        inflight = runtime.require_inflight_tool_registry()
        inflight.cancel_session(session_id)
        await registry.recycle(session_id)
        await registry.get_or_provision(session_id, pool=pool)
    except Exception as exc:
        if not _is_final_attempt(context.job, exc):
            log.warning(
                "sandbox.recycle_attempt_failed",
                session_id=session_id,
                attempt=run_number,
                error=str(exc),
            )
            raise  # procrastinate re-drives under ``recycle_retry``
        # No further retry will be scheduled: record the typed terminal failure
        # BEFORE the job lands in the failed table, so the outcome is durable.
        # The append is ordered ahead of the re-raise; if IT fails too, the
        # raise still surfaces the original.
        log.error(
            "sandbox.recycle_failed",
            session_id=session_id,
            attempts=run_number,
            error=str(exc),
        )
        await sessions_service.append_event(
            pool,
            session_id,
            "lifecycle",
            {
                "event": SANDBOX_RECYCLE_FAILED_EVENT,
                "requested_by": requested_by,
                "request_id": request_id,
                "attempts": run_number,
                "error": str(exc),
            },
            account_id=account_id,
        )
        raise
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {
            "event": SANDBOX_RECYCLED_EVENT,
            "requested_by": requested_by,
            "request_id": request_id,
            "attempts": run_number,
        },
        account_id=account_id,
    )
