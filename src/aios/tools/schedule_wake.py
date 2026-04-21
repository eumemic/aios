"""The schedule_wake tool — ask the harness to wake this session after a delay.

Gives an agent without bash access a first-class "wait N seconds and then
do X" primitive.  The tool returns immediately; at T+delay the session
wakes with ``cause="scheduled"`` and a user-role marker whose content
echoes ``reason`` is appended before the sweep guard runs, so the model
has something to react to.
"""

from __future__ import annotations

from typing import Any

from aios.errors import AiosError
from aios.harness import runtime
from aios.harness.wake import defer_wake
from aios.tools.registry import registry


class ScheduleWakeArgumentError(AiosError):
    error_type = "schedule_wake_argument_error"
    status_code = 400


SCHEDULE_WAKE_MIN_DELAY_SECONDS = 1
SCHEDULE_WAKE_MAX_DELAY_SECONDS = 3600


SCHEDULE_WAKE_DESCRIPTION = (
    "Ask the harness to wake you again after a delay. Use this for "
    "'wait N seconds and then do X' tasks instead of occupying a sandbox "
    "with `sleep`. Returns immediately; you'll receive a new step at "
    "T+delay_seconds with a marker echoing your `reason` so you remember "
    "why you scheduled this. While the scheduled wake is pending, new "
    "user messages will queue but not trigger an earlier step — keep "
    "delays short if the conversation is active."
)

SCHEDULE_WAKE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "delay_seconds": {
            "type": "integer",
            "minimum": SCHEDULE_WAKE_MIN_DELAY_SECONDS,
            "maximum": SCHEDULE_WAKE_MAX_DELAY_SECONDS,
            "description": (
                f"How long to wait before the next wake, in seconds. "
                f"Range {SCHEDULE_WAKE_MIN_DELAY_SECONDS}"
                f"-{SCHEDULE_WAKE_MAX_DELAY_SECONDS}."
            ),
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Short note shown back to you at wake time so you remember why you scheduled this."
            ),
        },
    },
    "required": ["delay_seconds", "reason"],
    "additionalProperties": False,
}


async def schedule_wake_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    delay_seconds = arguments.get("delay_seconds")
    if not isinstance(delay_seconds, int):
        raise ScheduleWakeArgumentError("delay_seconds must be an integer")
    if (
        delay_seconds < SCHEDULE_WAKE_MIN_DELAY_SECONDS
        or delay_seconds > SCHEDULE_WAKE_MAX_DELAY_SECONDS
    ):
        raise ScheduleWakeArgumentError(
            f"delay_seconds must be between {SCHEDULE_WAKE_MIN_DELAY_SECONDS} "
            f"and {SCHEDULE_WAKE_MAX_DELAY_SECONDS}"
        )
    reason = arguments.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ScheduleWakeArgumentError("reason must be a non-empty string")

    await defer_wake(
        runtime.require_pool(),
        session_id,
        cause="scheduled",
        delay_seconds=delay_seconds,
        wake_reason=reason,
    )

    return {
        "scheduled": True,
        "delay_seconds": delay_seconds,
        "reason": reason,
    }


def _register() -> None:
    registry.register(
        name="schedule_wake",
        description=SCHEDULE_WAKE_DESCRIPTION,
        parameters_schema=SCHEDULE_WAKE_PARAMETERS_SCHEMA,
        handler=schedule_wake_handler,
    )


_register()
