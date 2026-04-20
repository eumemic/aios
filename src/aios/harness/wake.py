"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from procrastinate import exceptions as procrastinate_exceptions

from aios.logging import get_logger

log = get_logger("aios.harness.wake")


async def defer_wake(session_id: str, *, cause: str = "message") -> None:
    """Enqueue a ``wake_session`` job, swallowing ``AlreadyEnqueued``.

    If a wake is already queued for this session (``queueing_lock``
    deduplication), the existing job will process any new events when
    it runs — no need for a second job.
    """
    from aios.harness.procrastinate_app import app

    try:
        await app.configure_task("harness.wake_session").defer_async(
            session_id=session_id,
            cause=cause,
        )
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("wake.already_enqueued", session_id=session_id, cause=cause)


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
        await app.configure_task("harness.wake_session").defer_async(
            session_id=session_id,
            cause="reschedule",
            schedule_in={"seconds": delay_seconds},
        )
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug(
            "wake.retry_already_enqueued",
            session_id=session_id,
            delay_seconds=delay_seconds,
        )
