"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

from aios.logging import get_logger
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    import asyncpg

log = get_logger("aios.harness.wake")


async def defer_wake(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    cause: str = "message",
    delay_seconds: float | None = None,
    wake_reason: str | None = None,
) -> None:
    """Enqueue a ``wake_session`` job, swallowing ``AlreadyEnqueued``.

    If a wake is already queued for this session (``queueing_lock``
    deduplication), the existing job will process any new events when
    it runs — no need for a second job.

    ``delay_seconds`` schedules the job that many seconds in the future
    (procrastinate's ``schedule_in``).  ``wake_reason`` is a short string
    carried as a task kwarg and surfaced to the agent at wake time when
    ``cause == "scheduled"``; see ``run_session_step``.

    Appends a ``wake_deferred`` span event before enqueuing — emitted
    regardless of whether procrastinate coalesces this deferral with
    an existing queued wake, so the profiler (issue #132) can observe
    coalescing as N ``wake_deferred`` → 1 ``step_start``.
    """
    from aios.harness.procrastinate_app import app

    span_data: dict[str, Any] = {"event": "wake_deferred", "cause": cause}
    if delay_seconds is not None:
        span_data["delay_seconds"] = delay_seconds
    await sessions_service.append_event(pool, session_id, "span", span_data)

    task_kwargs: dict[str, JSONValue] = {"session_id": session_id, "cause": cause}
    if wake_reason is not None:
        task_kwargs["wake_reason"] = wake_reason

    if delay_seconds is not None:
        deferrer = app.configure_task(
            "harness.wake_session",
            lock=session_id,
            queueing_lock=session_id,
            schedule_in={"seconds": delay_seconds},
        )
    else:
        deferrer = app.configure_task(
            "harness.wake_session",
            lock=session_id,
            queueing_lock=session_id,
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
