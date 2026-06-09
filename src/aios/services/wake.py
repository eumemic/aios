"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

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
    (procrastinate's ``schedule_in``). Used for the harness retry-backoff
    path; the user-visible scheduled-wake feature now goes through
    :mod:`aios.tools.schedule_wake` and creates one-shot scheduled_tasks
    rows instead.

    Appends a ``wake_deferred`` span event before enqueuing — emitted
    regardless of whether procrastinate coalesces this deferral with
    an existing queued wake, so the profiler (issue #132) can observe
    coalescing as N ``wake_deferred`` → 1 ``step_start``.
    """
    from aios.harness.procrastinate_app import app

    # Foreground protection: a workflow agent() child yields to user-facing sessions so a
    # fan-out can't starve a user's message. The signal is a non-null parent_run_id — set
    # (together with origin='background') only by create_child_session — read here from the
    # session's immutable row, so every wake of the child (first spawn, each tool-completion,
    # the sweep) is demoted uniformly at this single chokepoint, with no caller plumbing. A
    # missing row (deleted-session race) → foreground default; the wake then no-ops harmlessly.
    # Resolved before the span/enqueue so it isn't a new failure point between them.
    async with pool.acquire() as conn:
        ctx = await queries.get_session_workflow_context(conn, session_id)
    parent_run_id = ctx[1] if ctx is not None else None  # (account_id, parent_run_id) | None
    priority = _BACKGROUND_PRIORITY if parent_run_id is not None else _FOREGROUND_PRIORITY

    span_data: dict[str, Any] = {"event": "wake_deferred", "cause": cause}
    if delay_seconds is not None:
        span_data["delay_seconds"] = delay_seconds
    await sessions_service.append_event(pool, session_id, "span", span_data, account_id=account_id)

    task_kwargs: dict[str, JSONValue] = {"session_id": session_id, "cause": cause}

    if delay_seconds is not None:
        deferrer = app.configure_task(
            "harness.wake_session",
            lock=session_id,
            queueing_lock=session_id,
            priority=priority,
            schedule_in={"seconds": delay_seconds},
        )
    else:
        deferrer = app.configure_task(
            "harness.wake_session",
            lock=session_id,
            queueing_lock=session_id,
            priority=priority,
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


async def defer_run_wake(run_id: str) -> None:
    """Enqueue a ``wake_workflow`` job for a run, swallowing ``AlreadyEnqueued``.

    Unlike :func:`defer_wake`, this appends **no** journal span: ``wf_run_events``
    is single-writer — only ``run_workflow_step`` (under ``lock=run_id``) writes
    it. ``queueing_lock`` dedups concurrent wakes for the same run.
    """
    from aios.harness.procrastinate_app import app

    deferrer = app.configure_task(
        "harness.wake_workflow",
        lock=run_id,
        queueing_lock=run_id,
        priority=_BACKGROUND_PRIORITY,  # workflow run steps yield to foreground too
    )
    try:
        await deferrer.defer_async(run_id=run_id)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("run_wake.already_enqueued", run_id=run_id)
