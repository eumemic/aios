"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    import asyncpg

log = get_logger("aios.services.wake")

# Foreground protection: procrastinate fetches todo jobs in (priority DESC, id ASC)
# order, so a negative priority makes a job yield to higher-priority ones. Background
# workflow work — agent() children + run steps — is demoted below user-facing
# (foreground) sessions so a workflow's fan-out can't starve a user's message.
# Foreground stays at the procrastinate default; only background is demoted.
_FOREGROUND_PRIORITY = 0
_BACKGROUND_PRIORITY = -10


async def defer_wake(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    cause: str = "message",
    delay_seconds: float | None = None,
) -> None:
    """Enqueue a ``wake_session`` job, swallowing ``AlreadyEnqueued``.

    If a wake is already queued for this session (``queueing_lock``
    deduplication), the existing job will process any new events when
    it runs — no need for a second job.

    ``delay_seconds`` schedules the job that many seconds in the future
    (procrastinate's ``schedule_in``). Used by the harness retry-backoff
    path (``cause="reschedule"``) and the connector-inbound debounce path
    (``cause="inbound"``, gated by the ``inbound_debounce_seconds`` setting);
    the user-visible scheduled-wake feature now goes through
    :mod:`aios.tools.schedule_wake` and creates one-shot triggers
    instead.

    Appends a ``wake_deferred`` span event before enqueuing — emitted
    regardless of whether procrastinate coalesces this deferral with
    an existing queued wake, so the profiler (issue #132) can observe
    coalescing as N ``wake_deferred`` → 1 ``step_start``.
    """
    from aios.harness.procrastinate_app import app

    # Foreground protection: a request-serving descendant yields to user-facing sessions so a
    # fan-out can't starve a user's message. The signal is derived per-stimulus from the
    # triggering edge's up-link — #1123's ``request_opened`` ``caller`` — so every caller kind
    # (api/session/run) demotes uniformly when its ancestor is background, not just the run path:
    # a run-launched child stays background (behavior-preserved), a background-rooted session
    # invoke demotes too, and a root / fg-user session stays foreground. Read here at this single
    # chokepoint, so every wake of the descendant (first spawn, each tool-completion, the sweep)
    # is demoted uniformly with no caller plumbing. A missing row (deleted-session race) →
    # foreground default; the wake then no-ops harmlessly. Resolved before the span/enqueue so
    # it isn't a new failure point between them.
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, session_id)
    is_background = ctx[1] if ctx is not None else False  # (account_id, is_background) | None
    priority = _BACKGROUND_PRIORITY if is_background else _FOREGROUND_PRIORITY

    span_data: dict[str, Any] = {"event": "wake_deferred", "cause": cause}
    if delay_seconds is not None:
        span_data["delay_seconds"] = delay_seconds
    await sessions_service.append_event(pool, session_id, "span", span_data, account_id=account_id)

    task_kwargs: dict[str, JSONValue] = {"session_id": session_id, "cause": cause}

    # configure_task accepts schedule_in=None (treated as "no schedule").
    deferrer = app.configure_task(
        "harness.wake_session",
        lock=session_id,
        queueing_lock=session_id,
        priority=priority,
        schedule_in={"seconds": delay_seconds} if delay_seconds is not None else None,
    )

    try:
        await deferrer.defer_async(**task_kwargs)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug(
            "wake.already_enqueued",
            session_id=session_id,
            cause=cause,
            delay_seconds=delay_seconds,
        )


async def defer_run_wake(run_id: str, *, batch: bool = False) -> None:
    """Enqueue a ``wake_workflow`` job for a run, swallowing ``AlreadyEnqueued``.

    Unlike :func:`defer_wake`, this appends **no** journal span: ``wf_run_events``
    is single-writer — only ``run_workflow_step`` (under ``lock=run_id``) writes
    it. ``queueing_lock`` dedups concurrent wakes for the same run.

    ``batch`` (#780) schedules the wake ``workflow_wake_batch_seconds`` out
    instead of immediately, so a burst of completions collapses into one
    re-drive: the scheduled job sits in ``todo`` holding the ``queueing_lock``,
    absorbing every further defer until it runs — including otherwise-immediate
    wakes (a gate resume, the step's self-wake) arriving inside the window.
    That absorption is sound because every wake source commits its signal/
    response BEFORE deferring, so the delayed step harvests it; the cost is
    latency bounded by the window. The high-frequency sources (child
    completions, tool results) pass ``batch=True``; with the setting at 0
    (the default) batching is off and every wake is immediate.
    """
    from aios.harness.procrastinate_app import app

    window = get_settings().workflow_wake_batch_seconds if batch else 0.0
    deferrer = app.configure_task(
        "harness.wake_workflow",
        lock=run_id,
        queueing_lock=run_id,
        priority=_BACKGROUND_PRIORITY,  # workflow run steps yield to foreground too
        schedule_in={"seconds": window} if window > 0 else None,
    )
    try:
        await deferrer.defer_async(run_id=run_id)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("run_wake.already_enqueued", run_id=run_id)


async def defer_trigger_fire(trigger_id: str, trigger_run_id: str) -> None:
    """Enqueue one ``run_trigger`` job for one run_completion fire.

    One job PER event fire: the ``queueing_lock`` keys the FIRE (the
    ``trigger_runs`` carrier row's id), never the bare trigger id — distinct
    completions of a watched workflow must never coalesce (the scheduler
    tick's per-trigger lock is single-flight by design; correct for cron,
    silently lossy for events). The lock still dedups a sweep re-defer racing
    a queued-but-unstarted job; a re-defer racing a RUNNING job is resolved by
    the carrier row's pending→running claim instead. No ``lock``: fires must
    not serialize against each other or session inference.
    """
    from aios.harness.procrastinate_app import app

    try:
        await app.configure_task(
            "harness.run_trigger",
            queueing_lock=f"trigger_run:{trigger_run_id}",
        ).defer_async(trigger_id=trigger_id, trigger_run_id=trigger_run_id)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("trigger_fire.already_enqueued", trigger_run_id=trigger_run_id)
