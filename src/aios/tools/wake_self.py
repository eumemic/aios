"""The ``wake_self`` tool — append a user-role message to *this* session.

Canonical self-wake primitive for cron-fired bash commands (and any
other sandbox-side caller). Replaces the older curl-the-broker idiom
so the broker secret never has to be interpolated into the command
string.
"""

from __future__ import annotations

from typing import Any

from aios.errors import AiosError
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services.wake import defer_wake
from aios.tools.registry import registry


class WakeSelfArgumentError(AiosError):
    error_type = "wake_self_argument_error"
    status_code = 400


WAKE_SELF_DESCRIPTION = (
    "Append content as a user-role message to your own session and "
    "schedule the next step. This is the canonical self-wake primitive "
    "from inside a trigger's sandbox_command (or any sandbox bash "
    "command) — it replaces the older "
    "``curl $TOOL_BROKER_URL/v1/$TOOL_BROKER_SECRET/sessions/messages`` "
    "idiom and keeps the broker secret out of the command string. The "
    "appended message is delivered to your next model call as a normal "
    "user-role event. Available on both the model-tool surface and as "
    '``tool wake_self \'{"content":"..."}\'`` from inside the sandbox.'
)

WAKE_SELF_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": ["content"],
    "additionalProperties": False,
}


async def wake_self_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, str) or not content:
        raise WakeSelfArgumentError("content must be a non-empty string")

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    event = await sessions_service.append_user_message(
        pool, session_id, content, account_id=account_id
    )
    await defer_wake(pool, session_id, cause="message", account_id=account_id)
    return {
        "woken": True,
        "session_id": session_id,
        "event_id": event.id,
    }


def _register() -> None:
    registry.register(
        name="wake_self",
        description=WAKE_SELF_DESCRIPTION,
        parameters_schema=WAKE_SELF_PARAMETERS_SCHEMA,
        handler=wake_self_handler,
        transport="both",
    )


_register()
