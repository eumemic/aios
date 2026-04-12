"""The cancel tool — cancel in-flight tool executions.

Gives the model the ability to cancel background tool tasks. When the
model decides a running tool is no longer needed (user changed their
mind, tool is taking too long, different approach preferred), it calls
this tool to cancel it.

If ``tool_call_id`` is provided, cancels that specific task. If omitted,
cancels all in-flight tasks for the session.

The cancelled task's ``finally`` block appends an error tool_result
(``"cancelled"``) and defers a wake, so the model will see the
cancellation result on the next step.
"""

from __future__ import annotations

from typing import Any

from aios.errors import AiosError
from aios.harness import runtime
from aios.tools.registry import registry


class CancelArgumentError(AiosError):
    error_type = "cancel_argument_error"
    status_code = 400


CANCEL_DESCRIPTION = (
    "Cancel in-flight tool executions for this session. If "
    "`tool_call_id` is provided, cancels that specific tool call. "
    "If omitted, cancels all currently running tool calls. "
    "Cancelled tools will report an error result. Use this when "
    "you want to stop a long-running operation (e.g. a slow bash "
    "command) and try a different approach."
)

CANCEL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_call_id": {
            "type": "string",
            "description": (
                "The ID of the specific tool call to cancel. "
                "If omitted, cancels all in-flight tool calls for this session."
            ),
        },
    },
    "additionalProperties": False,
}


async def cancel_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for the cancel tool."""
    task_reg = runtime.require_task_registry()
    tool_call_id = arguments.get("tool_call_id")

    if tool_call_id is not None:
        if not isinstance(tool_call_id, str):
            raise CancelArgumentError("tool_call_id must be a string")
        cancelled = task_reg.cancel_task(session_id, tool_call_id)
        return {"cancelled": cancelled, "tool_call_id": tool_call_id}

    count = task_reg.cancel_session(session_id)
    return {"cancelled": True, "count": count}


def _register() -> None:
    registry.register(
        name="cancel",
        description=CANCEL_DESCRIPTION,
        parameters_schema=CANCEL_PARAMETERS_SCHEMA,
        handler=cancel_handler,
    )


_register()
