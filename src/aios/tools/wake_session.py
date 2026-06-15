"""The wake_session tool — wake another session in the same account.

The generalising primitive behind supervisor-wake, broadcast notification,
inter-lieutenant coordination, and friends (see issue #373).  Calling this
tool appends a user-role ``message`` event to the target session and
defers a ``wake_session`` job for it, so the target's next step
picks the new prompt up via the standard event-log mechanics.

Same-account scoping: the caller's session loads its ``account_id``;
the target session must have the SAME ``account_id``.  Cross-account
wake is refused.  This collapses the earlier v1 (same-bearer) vs v2
(account-scoped) framing — there's one model from day one.

Wake-depth and rate-limit are enforced in this handler (not in
``defer_wake``, which is a generic harness wake mechanism used for
user messages, scheduled wakes, and tool completions; cross-session
concerns don't belong there):

* Wake-depth — the depth counter is stamped on the appended message's
  ``data.metadata.wake_depth``.  The handler reads the source session's
  most recent user-role message; if its metadata carries a ``wake_depth``,
  the new message's depth is ``N + 1``.  Otherwise the new depth is 1.
  Refuses at ``>= WAKE_SESSION_MAX_DEPTH`` so a wake loop
  (A → B → A → ...) terminates in bounded steps.

* Rate-limit — counts target-side user messages stamped with
  ``data.metadata.wake_source_session_id == <this session>`` in the
  last hour.  Refuses at ``>= WAKE_SESSION_MAX_PER_HOUR``.  This is
  per (source, target) pair so an honest broadcast to many targets is
  not throttled, but a single source can't DoS a single target.
"""

from __future__ import annotations

from typing import Any

from aios.errors import AiosError
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services.wake import defer_wake
from aios.tools.registry import registry

WAKE_SESSION_MAX_DEPTH = 10
WAKE_SESSION_MAX_PER_HOUR = 10
WAKE_SESSION_MAX_PROMPT_CHARS = 100_000


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


WAKE_SESSION_DESCRIPTION = (
    "Wake another session by appending a user-role message to its log and "
    "scheduling its next step. Use this for cross-session coordination — "
    "escalate to a supervisor, hand work off to a peer, notify a watcher. "
    "The target must be in the same account as you; archived or terminated "
    "targets are refused. Wake-depth (chained wakes from one wake) is "
    f"capped at {WAKE_SESSION_MAX_DEPTH}; per-pair rate is capped at "
    f"{WAKE_SESSION_MAX_PER_HOUR} wakes/hour. You must already know the "
    "target's session_id — there is no name lookup."
)

WAKE_SESSION_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_session_id": {
            "type": "string",
            "minLength": 1,
            "description": "The session_id (e.g. 'sess_01...') of the session to wake.",
        },
        "prompt": {
            "type": "string",
            "minLength": 1,
            "maxLength": WAKE_SESSION_MAX_PROMPT_CHARS,
            "description": (
                "The message to deliver to the target session. Appears in "
                "the target's log as a user-role message — write it as if "
                "speaking directly to the target agent. Include any urgency "
                "or escalation framing in the prose; there is no separate "
                "urgency parameter."
            ),
        },
    },
    "required": ["target_session_id", "prompt"],
    "additionalProperties": False,
}


# The system-owned event that carries trusted wake lineage.  It is a
# ``kind='span'`` event — a kind NO caller-facing path can append (the
# operator POST and connector inbound paths only ever write ``kind='message'``
# user events).  This is what makes the wake-depth / wake-source cap
# non-forgeable: the cap reads provenance from this span, never from the
# user message's ``metadata`` (which the operator-POST / connector paths
# pass through unstripped — see issue #1083).
WAKE_LINEAGE_SPAN_EVENT = "wake_lineage"


async def _read_source_wake_depth(pool: Any, session_id: str, *, account_id: str) -> int:
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


async def _count_recent_wakes_from(
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


async def wake_session_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from aios.db import queries

    target_session_id = arguments.get("target_session_id")
    prompt = arguments.get("prompt")
    if not isinstance(target_session_id, str) or not target_session_id:
        raise WakeSessionArgumentError("target_session_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise WakeSessionArgumentError("prompt must be a non-empty string")
    if len(prompt) > WAKE_SESSION_MAX_PROMPT_CHARS:
        raise WakeSessionArgumentError(
            f"prompt exceeds {WAKE_SESSION_MAX_PROMPT_CHARS} characters (got {len(prompt)})"
        )

    if target_session_id == session_id:
        # Self-wake has dedicated tools — reject so an agent can't
        # accidentally bypass those surfaces and the depth tracking
        # doesn't conflate self-reentry with cross-session loops.
        raise WakeSessionArgumentError(
            "cannot wake your own session; use wake_self for immediate "
            "self-delivery, or schedule_wake for a delayed self-wake"
        )

    pool = runtime.require_pool()

    # Caller's account_id: this is the authority the tool inherits.
    source_account_id = await sessions_service.load_session_account_id(pool, session_id)

    # Load the target row directly (unscoped) so we can produce a clear
    # permission error when the target exists under a different account
    # rather than a generic 404.  This is the only place in the codebase
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
    if target_account_id != source_account_id:
        # Don't leak the existence of cross-account sessions: present
        # the same shape an attacker would see for an unknown id.
        raise WakeSessionPermissionError(
            f"target session {target_session_id} not found",
            detail={"target_session_id": target_session_id},
        )

    if target_row["archived_at"] is not None:
        raise WakeSessionTargetUnavailableError(
            f"target session {target_session_id} is archived",
            detail={"target_session_id": target_session_id, "reason": "archived"},
        )

    # Wake-depth: inherit from the source's most recent user message;
    # bail before any side effect if the new depth would breach the cap.
    source_depth = await _read_source_wake_depth(pool, session_id, account_id=source_account_id)
    new_depth = source_depth + 1
    if new_depth > WAKE_SESSION_MAX_DEPTH:
        raise WakeSessionDepthExceededError(
            f"wake-depth would exceed {WAKE_SESSION_MAX_DEPTH} "
            f"(current={source_depth}); refusing to extend the chain",
            detail={
                "current_depth": source_depth,
                "max_depth": WAKE_SESSION_MAX_DEPTH,
            },
        )

    # Per (source, target) rate limit.  Counted BEFORE the append so the
    # window starts fresh for the next hour — a burst at the cap doesn't
    # rebuild forward on every retry.
    recent_wakes = await _count_recent_wakes_from(
        pool,
        source_session_id=session_id,
        target_session_id=target_session_id,
        account_id=source_account_id,
    )
    if recent_wakes >= WAKE_SESSION_MAX_PER_HOUR:
        raise WakeSessionRateLimitedError(
            f"rate limit: {recent_wakes} wakes from {session_id} to "
            f"{target_session_id} in the last hour (max "
            f"{WAKE_SESSION_MAX_PER_HOUR})",
            detail={
                "recent_wakes": recent_wakes,
                "max_per_hour": WAKE_SESSION_MAX_PER_HOUR,
            },
        )

    # Append the user message to the TARGET, then defer a wake there.
    # Using ``queries.append_event`` directly (rather than
    # ``append_user_message``) so we can stamp wake-tracking metadata
    # without involving the API-side size check / channel lift logic
    # — those are not relevant for an in-process wake.
    #
    # The ``metadata`` on the user message is DISPLAY-ONLY: it drives the
    # model-visible wake header (``_wake_header`` in harness/context.py).
    # It is NOT trusted by the depth / rate-limit cap, because the
    # operator-POST and connector ingestion paths pass caller-supplied
    # ``metadata`` through unstripped and could forge these keys (#1083).
    metadata: dict[str, Any] = {
        "wake_source_session_id": session_id,
        "wake_depth": new_depth,
    }
    # The TRUSTED carrier: a system-owned ``kind='span'`` event.  No
    # caller-facing route appends span events, so the cap's reads
    # (_read_source_wake_depth / _count_recent_wakes_from) can rely on it
    # as non-forgeable provenance.
    lineage_span: dict[str, Any] = {
        "event": WAKE_LINEAGE_SPAN_EVENT,
        "wake_source_session_id": session_id,
        "wake_depth": new_depth,
    }
    async with pool.acquire() as conn:
        await queries.append_event(
            conn,
            account_id=target_account_id,
            session_id=target_session_id,
            kind="message",
            data={"role": "user", "content": prompt, "metadata": metadata},
        )
        await queries.append_event(
            conn,
            account_id=target_account_id,
            session_id=target_session_id,
            kind="span",
            data=lineage_span,
        )
    await defer_wake(
        pool,
        target_session_id,
        cause="agent_wake",
        account_id=target_account_id,
    )

    return {
        "woken": True,
        "target_session_id": target_session_id,
        "wake_depth": new_depth,
    }


def _register() -> None:
    registry.register(
        name="wake_session",
        description=WAKE_SESSION_DESCRIPTION,
        parameters_schema=WAKE_SESSION_PARAMETERS_SCHEMA,
        handler=wake_session_handler,
        transport="agent_tool",
    )


_register()
