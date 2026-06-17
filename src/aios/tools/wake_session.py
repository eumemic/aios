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

from aios.harness import runtime
from aios.services import sessions as sessions_service

# The cross-session delivery primitive, the caps, the trusted-lineage span,
# and the status-code-bearing error classes all live in ``services.wake`` (the
# single home — issue #1280) so BOTH this tool and the ``wake_session`` trigger
# action call one function. Re-imported here so the tool's long-standing public
# import path keeps working for callers and tests.
from aios.services.wake import (
    WAKE_SESSION_MAX_DEPTH,
    WAKE_SESSION_MAX_PER_HOUR,
    CrossSessionWakeRoot,
    WakeSessionArgumentError,
    WakeSessionDepthExceededError,
    WakeSessionPermissionError,
    WakeSessionRateLimitedError,
    WakeSessionTargetUnavailableError,
    deliver_cross_session_wake,
    read_source_wake_depth,
)
from aios.tools.registry import registry

__all__ = [
    "WAKE_SESSION_MAX_DEPTH",
    "WAKE_SESSION_MAX_PER_HOUR",
    "WAKE_SESSION_MAX_PROMPT_CHARS",
    "WakeSessionArgumentError",
    "WakeSessionDepthExceededError",
    "WakeSessionPermissionError",
    "WakeSessionRateLimitedError",
    "WakeSessionTargetUnavailableError",
    "wake_session_handler",
]

WAKE_SESSION_MAX_PROMPT_CHARS = 100_000


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


async def wake_session_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
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

    # Caller's account_id: this is the authority the tool inherits. The
    # tool's lineage ROOT is the calling session — depth read from its own
    # log; the caps then attribute the wake to (this session, target).
    source_account_id = await sessions_service.load_session_account_id(pool, session_id)
    root = CrossSessionWakeRoot(
        source_id=session_id,
        source_depth=await read_source_wake_depth(pool, session_id, account_id=source_account_id),
    )
    new_depth = await deliver_cross_session_wake(
        pool,
        target_session_id=target_session_id,
        content=prompt,
        account_id=source_account_id,
        root=root,
        cause="agent_wake",
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
