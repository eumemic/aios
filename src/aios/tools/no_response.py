"""The no_response tool — end a step without replying (#2092).

A channel-bound session's assistant text is never delivered: replies go
through the connector's send/react tools. ``no_response`` is the explicit
way to decline: it sends nothing and its result is **quiet** (see
:data:`aios.models.events.QUIET_RESULT_METADATA_KEY`) — persisted and
model-visible, but not a stimulus, so it cannot wake the session. A step
whose only call is ``no_response`` therefore leaves the session idle; one
that also called other tools still wakes on those results.

A waking result would invite a loop — each decline fed back as another
inference, which declines again; a quiet result rules that out by
construction rather than bounding it.

Calling it also discharges the session's reply-or-decline obligation for
every inbound the step saw (:mod:`aios.harness.reply_obligation`). Injected
alongside ``switch_channel`` whenever the session has bound channels.
"""

from __future__ import annotations

from typing import Any

from aios.harness.reply_obligation import NO_RESPONSE_TOOL_NAME
from aios.models.events import QUIET_RESULT_METADATA_KEY
from aios.tools.registry import ToolResult, registry

NO_RESPONSE_DESCRIPTION = (
    "Decide not to reply. Your assistant text is never delivered to a "
    "channel, so this is how you end a step on purpose without sending "
    "anything. Use it when the latest messages need no reply from you: a "
    "group conversation that doesn't involve you, an acknowledgement or "
    "reaction that closes a thread, a step with nothing left to send. It "
    "sends nothing and doesn't start another step. Do NOT use it to avoid a "
    "question, a request, criticism or a correction addressed to you; "
    "answer those with the send tool."
)

NO_RESPONSE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Why no reply is needed. Recorded in your history; never sent.",
        },
    },
    "required": ["reason"],
    "additionalProperties": False,
}

NO_RESPONSE_RESULT = "Nothing sent."


async def no_response_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    """Return the quiet ack. The reason lives in the call's arguments."""
    return ToolResult(
        content=NO_RESPONSE_RESULT,
        metadata={QUIET_RESULT_METADATA_KEY: {"tool": NO_RESPONSE_TOOL_NAME}},
    )


def _register() -> None:
    registry.register(
        name=NO_RESPONSE_TOOL_NAME,
        description=NO_RESPONSE_DESCRIPTION,
        parameters_schema=NO_RESPONSE_PARAMETERS_SCHEMA,
        handler=no_response_handler,
        transport="agent_tool",
    )


_register()
