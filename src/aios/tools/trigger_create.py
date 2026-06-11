"""The ``trigger_create`` tool — add a trigger to this session.

A trigger pairs a ``source`` (what fires it: a recurring ``cron`` schedule
or a one-shot ``fire_at``) with an ``action`` (what runs: a
``sandbox_command`` bash task that does NOT wake the model, or a
``wake_owner`` message delivered to this session that DOES wake the model).

Hand-written precursor to the autogen'd self-state tools. Granular ops only
— no whole-list ``set``. Each entry is identified by ``name`` (unique per
session).
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.triggers import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_COMMAND_CHARS,
    MAX_MAX_OUTPUT_BYTES,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_TIMEOUT_SECONDS,
    MAX_WAKE_CONTENT_CHARS,
    MIN_MAX_OUTPUT_BYTES,
    MIN_TIMEOUT_SECONDS,
    TriggerCreate,
)
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.registry import registry

TRIGGER_CREATE_DESCRIPTION = (
    "Add a trigger to this session: a `source` (what fires it) paired with "
    "an `action` (what runs at fire time).\n"
    "\n"
    "Sources: `{kind: 'cron', schedule}` (recurring, standard 5-field cron "
    "in UTC) or `{kind: 'one_shot', fire_at}` (fires once at an absolute UTC "
    "time, then self-deletes).\n"
    "\n"
    "Actions: `{kind: 'sandbox_command', command}` runs bash in the "
    "session's sandbox WITHOUT waking the model; `{kind: 'wake_owner', "
    "content}` delivers `content` as a user-role message to THIS session, "
    "waking the model.\n"
    "\n"
    "Use a sandbox_command for deterministic polling (file watchers, API "
    "checks, periodic syncs) without burning model tokens per fire; use "
    "wake_owner for a recurring or scheduled self-reminder. Names must be "
    "unique per session."
)

# Shared source/action schemas; #819's workflow action is one added oneOf
# branch.
_SOURCE_SCHEMA: dict[str, Any] = {
    "description": "What fires the trigger.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "cron"},
                "schedule": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SCHEDULE_CHARS,
                    "description": "Standard 5-field cron expression in UTC (e.g. '*/5 * * * *').",
                },
            },
            "required": ["kind", "schedule"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "one_shot"},
                "fire_at": {
                    "type": "string",
                    "description": (
                        "Absolute UTC fire time, ISO 8601 with offset (e.g. "
                        "'2026-06-11T09:00:00Z'). Fires once, then self-deletes."
                    ),
                },
            },
            "required": ["kind", "fire_at"],
            "additionalProperties": False,
        },
    ],
}

_SANDBOX_COMMAND_DESCRIPTION = (
    "Bash command run in the session's sandbox at each fire, WITHOUT waking "
    "the model. To escalate (wake the model with a user-role message), the "
    "canonical bash invocation is:\n"
    "\n"
    '  tool wake_self \'{"content":"<message to deliver to yourself>"}\'\n'
    "\n"
    "This keeps the broker secret out of the command string. Advanced or "
    "scripted callers may still POST to "
    "``$TOOL_BROKER_URL/v1/$TOOL_BROKER_SECRET/sessions/messages`` directly."
)

_ACTION_SCHEMA: dict[str, Any] = {
    "description": "What runs at fire time.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "sandbox_command"},
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_COMMAND_CHARS,
                    "description": _SANDBOX_COMMAND_DESCRIPTION,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": MIN_TIMEOUT_SECONDS,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS,
                    "description": "Per-fire timeout. The fire is killed if it runs longer.",
                },
                "max_output_bytes": {
                    "type": "integer",
                    "minimum": MIN_MAX_OUTPUT_BYTES,
                    "maximum": MAX_MAX_OUTPUT_BYTES,
                    "default": DEFAULT_MAX_OUTPUT_BYTES,
                    "description": "Per-fire stdout/stderr capture cap, in bytes.",
                },
            },
            "required": ["kind", "command"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "wake_owner"},
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_WAKE_CONTENT_CHARS,
                    "description": (
                        "Message delivered as a user-role event to THIS session at fire "
                        "time, waking the model. No sandbox involved."
                    ),
                },
            },
            "required": ["kind", "content"],
            "additionalProperties": False,
        },
    ],
}

TRIGGER_CREATE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "pattern": r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
            "description": "Unique identifier for this trigger within the session.",
        },
        "source": _SOURCE_SCHEMA,
        "action": _ACTION_SCHEMA,
        "enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether the trigger fires. Set false to add a paused entry.",
        },
        "metadata": {
            "type": "object",
            "description": "Opaque key/value tags stored with the trigger.",
        },
    },
    "required": ["name", "source", "action"],
    "additionalProperties": False,
}


async def trigger_create_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    spec = TriggerCreate.model_validate(arguments)
    echo = await triggers_service.add_trigger(pool, session_id, spec, account_id=account_id)
    return echo.model_dump(mode="json")


def _register() -> None:
    registry.register(
        name="trigger_create",
        description=TRIGGER_CREATE_DESCRIPTION,
        parameters_schema=TRIGGER_CREATE_PARAMETERS_SCHEMA,
        handler=trigger_create_handler,
    )


_register()
