"""Procrastinate App singleton + job-deferral primitives.

Phase 2 introduces a job queue substrate. Procrastinate uses psycopg3 (its own
``PsycopgConnector``), independent from our ``asyncpg`` pool. Both run in the
same process — they don't share connections, just the same Postgres database.

Both the api process (which calls ``defer_async`` to enqueue jobs) and the
worker process (which actually runs the jobs) import this module and use the
same ``app`` singleton.

**This module is job-queue INFRASTRUCTURE only.** It constructs the client and
exposes the deferral primitives; it does NOT register any ``@app.task``
handlers. Registration is a worker-only side effect: the worker entrypoint
imports :mod:`aios.harness.tasks` (mirroring its ``import aios.tools`` line) so
the ``@app.task(...)`` decorators bind against this exact ``app`` instance
where — and only where — the tasks execute. ``configure_task`` defers by name
string with ``allow_unknown=True``, so enqueuing never consults ``app.tasks``:
a process that only defers (e.g. the api) pays none of the harness import cost.

Dependencies are strictly downward: ``procrastinate`` + ``aios.config`` +
``aios.db.queries``. Nothing here reaches up into ``services`` or ``harness``,
so both may import these primitives at module level (issue #1476).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from procrastinate import App, PsycopgConnector
from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

from aios.config import get_settings
from aios.db import queries
from aios.logging import get_logger

if TYPE_CHECKING:
    import asyncpg

log = get_logger("aios.jobs.app")


def _sync_dsn(db_url: str) -> str:
    """Strip any +async driver suffix from a Postgres DSN.

    psycopg3 accepts the bare ``postgresql://`` form. ``postgresql+psycopg://``
    and ``postgresql+asyncpg://`` are SQLAlchemy/alembic conventions that need
    to be stripped before passing to psycopg.
    """
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


def _build_connector() -> PsycopgConnector:
    settings = get_settings()
    return PsycopgConnector(conninfo=_sync_dsn(settings.db_url))


# Module-level singleton. Constructed eagerly at import time so any
# `from aios.jobs.app import app` line resolves immediately. NO tasks are
# registered here — that is a worker-only side effect (see the module
# docstring); importing this module never drags in the harness execution graph.
app: App = App(connector=_build_connector())


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
    # Append the ``wake_deferred`` span directly via ``aios.db.queries`` (a pure
    # downward dependency). The old ``services.sessions.append_event`` wrapper is
    # a 3-line pass-through over exactly this call, and ``deliver_cross_session_wake``
    # already writes spans via ``queries.append_event`` directly (issue #1476).
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="span",
            data=span_data,
        )

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
    try:
        await app.configure_task(
            "harness.run_trigger",
            queueing_lock=f"trigger_run:{trigger_run_id}",
        ).defer_async(trigger_id=trigger_id, trigger_run_id=trigger_run_id)
    except procrastinate_exceptions.AlreadyEnqueued:
        log.debug("trigger_fire.already_enqueued", trigger_run_id=trigger_run_id)
