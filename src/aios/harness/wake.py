"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

from aios.logging import get_logger

log = get_logger("aios.harness.wake")


async def defer_wake(
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
    """
    from aios.harness.procrastinate_app import app

    task_kwargs: dict[str, JSONValue] = {"session_id": session_id, "cause": cause}
    if wake_reason is not None:
        task_kwargs["wake_reason"] = wake_reason

    if delay_seconds is not None:
        deferrer = app.configure_task(
            "harness.wake_session",
            schedule_in={"seconds": delay_seconds},
        )
    else:
        deferrer = app.configure_task("harness.wake_session")

    try:
        await deferrer.defer_async(**task_kwargs)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug(
            "wake.already_enqueued",
            session_id=session_id,
            cause=cause,
            delay_seconds=delay_seconds,
        )


async def defer_retry_wake(session_id: str, *, delay_seconds: float) -> None:
    """Enqueue a delayed ``wake_session`` job for retry after a transient error.

    Mirrors :func:`defer_wake` but schedules the job ``delay_seconds``
    in the future via procrastinate's ``schedule_in``.  An already-queued
    wake for the same session dedups this retry via ``queueing_lock``;
    that's benign — the queued wake will run, hit the same error, and
    the handler will defer a fresh retry.
    """
    from aios.harness.procrastinate_app import app

    try:
        await app.configure_task(
            "harness.wake_session",
            schedule_in={"seconds": delay_seconds},
        ).defer_async(
            session_id=session_id,
            cause="reschedule",
        )
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug(
            "wake.retry_already_enqueued",
            session_id=session_id,
            delay_seconds=delay_seconds,
        )
