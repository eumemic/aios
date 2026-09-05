"""Cross-session wake delivery primitive + its public error contract.

This module is the single home of :func:`deliver_cross_session_wake` and the
lineage-cap readers behind the ``wake_session`` tool and the ``wake_session``
trigger action, plus the ``WakeSession*Error`` classes and
``CrossSessionWakeRoot`` that ``aios.tools.wake_session`` re-exports as its
public contract.

The job-queue *infrastructure* — the ``app`` singleton and the ``defer_*``
deferral primitives — moved DOWN to :mod:`aios.jobs.app` (issue #1476), so the
``services ↔ harness`` job-queue cycle is severed at the root and callers import
the deferral primitive downward. This module imports :func:`defer_wake` from
there like any other lower-layer consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from aios.db import queries
from aios.errors import AiosError
from aios.jobs.app import defer_wake
from aios.logging import get_logger

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

# Per-(source, target) transaction-scoped advisory lock guarding the rate-cap
# count+append.  Mirrors ``outbound_tool_quota._LOCK_SQL``: the lock is taken
# inside the append transaction, so concurrent wakes sharing the SAME
# (source, target) pair serialize — every prior caller's span is committed
# (and visible under READ COMMITTED) before the next caller's count read.
# Wakes from OTHER sources to the same target take a DIFFERENT lock and are
# unaffected, preserving the per-pair (not per-target) scope of the cap.
# ``hashtextextended`` maps the pair string to the ``bigint`` key
# ``pg_advisory_xact_lock`` requires; any hash collision is a harmless
# spurious wait between unrelated pairs (a rate-limit lock, not a
# correctness lock).
_WAKE_RATE_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))"

# The target-side ``wake_lineage`` span count for the per-pair rate cap.  The
# window is the last hour.  ``$3``-then-``$4`` parameter ordering is
# historical (``wake_source_session_id`` was added after ``event``); kept
# stable so callers bind positionally.
_COUNT_RECENT_WAKES_SQL = """
    SELECT count(*)
    FROM events
    WHERE session_id = $1
      AND account_id = $2
      AND kind = 'span'
      AND data->>'event' = $4
      AND data->>'wake_source_session_id' = $3
      AND created_at > now() - interval '1 hour'
"""

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

    This is the public, self-contained wrapper (acquires its own pooled
    connection).  The concurrency-safe rate-cap enforcement in
    :func:`deliver_cross_session_wake` does NOT use this wrapper — it runs
    the SAME count via :func:`_count_recent_wakes_from_conn` INSIDE a
    transaction-scoped advisory lock so concurrent same-pair callers
    serialize on the count+append.
    """
    async with pool.acquire() as conn:
        return await _count_recent_wakes_from_conn(
            conn,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            account_id=account_id,
        )


async def _count_recent_wakes_from_conn(
    conn: Any,
    *,
    source_session_id: str,
    target_session_id: str,
    account_id: str,
) -> int:
    """Connection-bound form of :func:`count_recent_wakes_from` — runs the
    count on the caller-supplied connection (typically inside the locked
    append transaction in :func:`deliver_cross_session_wake`)."""
    count = await conn.fetchval(
        _COUNT_RECENT_WAKES_SQL,
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
    ``WAKE_SESSION_MAX_DEPTH`` (chain) BEFORE any side effect → in ONE
    transaction take a per-``(root.source_id, target)`` transaction-scoped
    advisory lock, enforce ``WAKE_SESSION_MAX_PER_HOUR`` under that lock (so
    concurrent same-pair callers serialize and the count+append is atomic),
    THEN stamp the non-forgeable ``wake_lineage`` span
    (``wake_depth = root.source_depth + 1``,
    ``wake_source_session_id = root.source_id``) and append ``content`` as a
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
    # rebuild forward on every retry.  The count + append run in ONE
    # transaction serialized by a per-pair transaction-scoped advisory lock:
    # a concurrent burst of ``wake_session`` calls all acquired separate
    # pooled connections for the count (READ COMMITTED — each sees only
    # committed rows), so without serialization each caller read a stale
    # count and overshot the cap.  The lock is taken on the SAME pair
    # ``(root.source_id, target)`` the cap keys on, so concurrent wakes
    # from OTHER sources to this target are unaffected.
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
        # Per-pair serialization: hold until COMMIT/ROLLBACK so the count
        # read sees every prior concurrent caller's committed span.
        await conn.execute(_WAKE_RATE_LOCK_SQL, f"{root.source_id}:{target_session_id}")
        recent_wakes = await _count_recent_wakes_from_conn(
            conn,
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
