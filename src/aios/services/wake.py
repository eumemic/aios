"""Shared helper to enqueue a ``wake_session`` job.

Used by the API router (on user messages) and by the async tool
dispatcher (on tool completion). Extracted so both sides import from
one place without creating a circular ``api → harness`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from procrastinate import exceptions as procrastinate_exceptions
from procrastinate.types import JSONValue

from aios.config import get_settings
from aios.db import queries
from aios.errors import AiosError
from aios.logging import get_logger
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    import asyncpg

log = get_logger("aios.services.wake")

# ─── cross-session wake: caps + trusted lineage carrier ──────────────────────
#
# These were inlined in ``tools/wake_session.py`` (the agent tool) until the
# trigger-action layer needed the SAME delivery primitive (issue #1280). They
# now live here — the one shared home — and the tool imports them. Both callers
# (the ``wake_session`` tool, and the ``wake_session`` TRIGGER ACTION) route
# through :func:`deliver_cross_session_wake`, parameterized by the lineage
# ROOT, so the depth/per-pair-rate caps apply uniformly.

WAKE_SESSION_MAX_DEPTH = 10
WAKE_SESSION_MAX_PER_HOUR = 10

# The system-owned event that carries trusted wake lineage.  It is a
# ``kind='span'`` event — a kind NO caller-facing path can append (the
# operator POST and connector inbound paths only ever write ``kind='message'``
# user events).  This is what makes the wake-depth / wake-source cap
# non-forgeable: the cap reads provenance from this span, never from the
# user message's ``metadata`` (which the operator-POST / connector paths
# pass through unstripped — see issue #1083).
WAKE_LINEAGE_SPAN_EVENT = "wake_lineage"

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


# ─── cross-session wake error classes ────────────────────────────────────────
#
# Status-code-bearing AiosError subclasses (the tool's public contract). They
# live HERE — the single home of the cross-session delivery primitive — so the
# tool, the runner, and this module all import them from one place (no cycle:
# services.wake does not import tools.wake_session). The tool re-imports them
# so its long-standing public import path ``aios.tools.wake_session`` keeps
# working for callers and tests.


class WakeSessionArgumentError(AiosError):
    error_type = "wake_session_argument_error"
    status_code = 400


class WakeSessionPermissionError(AiosError):
    error_type = "wake_session_permission_error"
    status_code = 403


class WakeSessionTargetUnavailableError(AiosError):
    error_type = "wake_session_target_unavailable"
    status_code = 409


class WakeSessionDepthExceededError(AiosError):
    error_type = "wake_session_depth_exceeded"
    status_code = 429


class WakeSessionRateLimitedError(AiosError):
    error_type = "wake_session_rate_limited"
    status_code = 429


class CrossSessionWakeRoot(NamedTuple):
    """Lineage root for a cross-session wake — what the depth/rate caps
    attribute the wake TO. Agent wakes root at the calling session
    (source_id=session_id, source_depth=read from its log). Trigger wakes
    root at the firing trigger (source_id=f"trigger:{trigger_id}",
    source_depth=0 — a trigger has no session log to read a depth from)."""

    source_id: str
    source_depth: int


async def read_source_wake_depth(pool: Any, session_id: str, *, account_id: str) -> int:
    """Return the trusted wake_depth for the most recent wake of this session.

    Reads the source's own log, but ONLY from the system-owned
    ``wake_lineage`` span event — never from user-message metadata, which
    the operator-POST / connector ingestion paths can forge (issue #1083).
    Returns 0 when no wake-lineage span is present (root call — a
    human-originated message or first wake of a chain).
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(
                (data->>'wake_depth')::int,
                0
            ) AS depth
            FROM events
            WHERE session_id = $1
              AND account_id = $2
              AND kind = 'span'
              AND data->>'event' = $3
            ORDER BY seq DESC
            LIMIT 1
            """,
            session_id,
            account_id,
            WAKE_LINEAGE_SPAN_EVENT,
        )
    if row is None:
        return 0
    return int(row["depth"])


async def count_recent_wakes_from(
    pool: Any,
    *,
    source_session_id: str,
    target_session_id: str,
    account_id: str,
) -> int:
    """Count target-side ``wake_lineage`` spans stamped with
    ``wake_source_session_id`` equal to ``source_session_id`` in the last hour.

    Counts the system-owned ``wake_lineage`` span — NOT user-message
    metadata, which the operator-POST / connector ingestion paths can forge
    (issue #1083) to either evade or inflate the count.

    The rate-limit window is per (source, target) pair so an honest
    broadcast to many distinct targets isn't throttled; a single source
    only gets ``WAKE_SESSION_MAX_PER_HOUR`` wakes/hour at any one target.
    The ``source_session_id`` is an OPAQUE string: a trigger root passes
    ``f"trigger:{trigger_id}"``, getting its own per-(trigger,target) bucket.
    """
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT count(*)
            FROM events
            WHERE session_id = $1
              AND account_id = $2
              AND kind = 'span'
              AND data->>'event' = $4
              AND data->>'wake_source_session_id' = $3
              AND created_at > now() - interval '1 hour'
            """,
            target_session_id,
            account_id,
            source_session_id,
            WAKE_LINEAGE_SPAN_EVENT,
        )
    return int(count or 0)


async def deliver_cross_session_wake(
    pool: asyncpg.Pool[Any],
    *,
    target_session_id: str,
    content: str,
    account_id: str,
    root: CrossSessionWakeRoot,
    cause: str,
) -> int:
    """Deliver ``content`` to ``target_session_id``, waking it — the shared
    primitive behind the ``wake_session`` tool and the ``wake_session`` trigger
    action.

    Validate the target (same ``account_id``, not archived) → enforce
    ``WAKE_SESSION_MAX_DEPTH`` (chain) and ``WAKE_SESSION_MAX_PER_HOUR``
    (per ``(root.source_id, target)`` pair) BEFORE any side effect → in ONE
    transaction stamp the non-forgeable ``wake_lineage`` span
    (``wake_depth = root.source_depth + 1``,
    ``wake_source_session_id = root.source_id``) THEN append ``content`` as a
    user-role message to the target (display-only wake metadata) →
    ``defer_wake(cause=cause)``. Span-first-and-atomic so the trusted depth
    carrier is never visible later than the message that makes the target
    wakeable. Returns the new depth.

    Raises :class:`WakeSessionPermissionError` (cross-account target — same
    shape an attacker would see for an unknown id, no existence leak),
    :class:`WakeSessionArgumentError` (target not found),
    :class:`WakeSessionTargetUnavailableError` (archived target),
    :class:`WakeSessionDepthExceededError`, :class:`WakeSessionRateLimitedError`.

    Does NOT enforce a content-length cap — each CALLER validates its own
    bound first (tool: 100_000; trigger: 16_384). The DB ``content`` CHECK
    only asserts it is a string.
    """
    # Load the target row directly (unscoped) so we can produce a clear
    # permission error when the target exists under a different account
    # rather than a generic 404. This is the only place in the codebase
    # that needs an unscoped-by-id read for a cross-session check, so it
    # justifies bypassing the usual ``account_id``-scoped helper.
    async with pool.acquire() as conn:
        target_row = await conn.fetchrow(
            "SELECT account_id, archived_at FROM sessions WHERE id = $1",
            target_session_id,
        )
    if target_row is None:
        raise WakeSessionArgumentError(
            f"target session {target_session_id} not found",
            detail={"target_session_id": target_session_id},
        )

    target_account_id: str = target_row["account_id"]
    if target_account_id != account_id:
        # Don't leak the existence of cross-account sessions: present the
        # same shape an attacker would see for an unknown id.
        raise WakeSessionPermissionError(
            f"target session {target_session_id} not found",
            detail={"target_session_id": target_session_id},
        )

    if target_row["archived_at"] is not None:
        raise WakeSessionTargetUnavailableError(
            f"target session {target_session_id} is archived",
            detail={"target_session_id": target_session_id, "reason": "archived"},
        )

    # Wake-depth: inherit from the root; bail before any side effect if the
    # new depth would breach the cap.
    new_depth = root.source_depth + 1
    if new_depth > WAKE_SESSION_MAX_DEPTH:
        raise WakeSessionDepthExceededError(
            f"wake-depth would exceed {WAKE_SESSION_MAX_DEPTH} "
            f"(current={root.source_depth}); refusing to extend the chain",
            detail={
                "current_depth": root.source_depth,
                "max_depth": WAKE_SESSION_MAX_DEPTH,
            },
        )

    # Per (root.source_id, target) rate limit. Counted BEFORE the append so
    # the window starts fresh for the next hour — a burst at the cap doesn't
    # rebuild forward on every retry.
    recent_wakes = await count_recent_wakes_from(
        pool,
        source_session_id=root.source_id,
        target_session_id=target_session_id,
        account_id=target_account_id,
    )
    if recent_wakes >= WAKE_SESSION_MAX_PER_HOUR:
        raise WakeSessionRateLimitedError(
            f"rate limit: {recent_wakes} wakes from {root.source_id} to "
            f"{target_session_id} in the last hour (max "
            f"{WAKE_SESSION_MAX_PER_HOUR})",
            detail={
                "recent_wakes": recent_wakes,
                "max_per_hour": WAKE_SESSION_MAX_PER_HOUR,
            },
        )

    # Append the lineage span + the user message to the TARGET atomically,
    # then defer a wake there.
    #
    # The ``metadata`` on the user message is DISPLAY-ONLY: it drives the
    # model-visible wake header (``_wake_header`` in harness/context.py).
    # It is NOT trusted by the depth / rate-limit cap, because the
    # operator-POST and connector ingestion paths pass caller-supplied
    # ``metadata`` through unstripped and could forge these keys (#1083).
    metadata: dict[str, Any] = {
        "wake_source_session_id": root.source_id,
        "wake_depth": new_depth,
    }
    # The TRUSTED carrier: a system-owned ``kind='span'`` event. No
    # caller-facing route appends span events, so the cap's reads
    # (read_source_wake_depth / count_recent_wakes_from) can rely on it
    # as non-forgeable provenance.
    lineage_span: dict[str, Any] = {
        "event": WAKE_LINEAGE_SPAN_EVENT,
        "wake_source_session_id": root.source_id,
        "wake_depth": new_depth,
    }
    # ONE transaction, span FIRST: the trusted depth carrier must never lag its
    # message. The message append bumps ``last_stimulus_seq`` and makes the target
    # sweep-wakeable; if it committed before the span, the periodic sweep could wake
    # the target into a step whose own cross-session wake reads a stale wake-depth
    # (the span the cap trusts not yet visible) and undercounts the chain. Atomic
    # commit closes that window — the same append+request_opened pattern the Ask arm
    # uses (``_stimulate_existing_ask``). NOTIFY is queued to the outermost COMMIT, so
    # the nested per-append ``pg_notify`` fires once, after commit (invariant #6).
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_event(
            conn,
            account_id=target_account_id,
            session_id=target_session_id,
            kind="span",
            data=lineage_span,
        )
        await queries.append_event(
            conn,
            account_id=target_account_id,
            session_id=target_session_id,
            kind="message",
            data={"role": "user", "content": content, "metadata": metadata},
        )
    await defer_wake(
        pool,
        target_session_id,
        cause=cause,
        account_id=target_account_id,
    )

    return new_depth
