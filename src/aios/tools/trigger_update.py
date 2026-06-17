"""The ``trigger_update`` tool — update a trigger by name.

Granular operation per #270 — accepts only the fields to change. ``name``
is the addressable identifier and cannot itself be changed. ``source`` and
``action`` are replaced WHOLESALE when provided (a cron↔one-shot or
sandbox↔wake conversion is just a different object — send the complete
object; fetch current values via ``trigger_list``). Toggling ``enabled``
true→false clears ``next_fire``; false→true recomputes it.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.triggers import (
    MAX_COMMAND_CHARS,
    MAX_INPUT_TEMPLATE_BYTES,
    MAX_MAX_OUTPUT_BYTES,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_TIMEOUT_SECONDS,
    MAX_WAKE_CONTENT_CHARS,
    MIN_MAX_OUTPUT_BYTES,
    MIN_TIMEOUT_SECONDS,
    RUN_TERMINAL_STATUSES,
    TriggerUpdate,
)
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.registry import registry

TRIGGER_UPDATE_DESCRIPTION = (
    "Update a trigger by name. Omitted top-level fields are left unchanged. "
    "`source` and `action`, when provided, REPLACE the stored object "
    "wholesale — send the complete object (fetch the current values via "
    "`trigger_list`). Toggling `enabled` true→false pauses the trigger "
    "(clears next_fire); false→true resumes it (recomputes next_fire). "
    "Any update whose resulting state is an enabled cron trigger always "
    "yields a recomputed (non-NULL) next_fire, so a cron trigger that "
    "somehow became enabled with a cleared schedule is automatically "
    "re-armed by any update (e.g. just re-enabling it) without re-sending "
    "`source`; one-shot and run_completion triggers are not re-armed this way."
)

_SOURCE_SCHEMA: dict[str, Any] = {
    "description": "Replacement source (wholesale). Omit to leave unchanged.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "cron"},
                "schedule": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SCHEDULE_CHARS,
                    "description": "Standard 5-field cron expression in UTC.",
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
                    "description": "Absolute UTC fire time, ISO 8601 with offset.",
                },
            },
            "required": ["kind", "fire_at"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "run_completion"},
                "workflow_id": {
                    "type": "string",
                    "description": "Id of the workflow whose run completions fire this trigger.",
                },
                "statuses": {
                    "type": "array",
                    "items": {"enum": list(RUN_TERMINAL_STATUSES)},
                    "minItems": 1,
                    "description": (
                        "Which terminal statuses fire (required on update — no "
                        "implicit default; send the complete filter)."
                    ),
                },
            },
            "required": ["kind", "workflow_id", "statuses"],
            "additionalProperties": False,
        },
    ],
}

# Update-side action schema: the sandbox_command branch marks ALL of
# command / timeout_seconds / max_output_bytes REQUIRED (wholesale replace —
# a partial action 422s instead of silently resetting to create-time
# defaults).
_ACTION_SCHEMA: dict[str, Any] = {
    "description": "Replacement action (wholesale). Omit to leave unchanged.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "sandbox_command"},
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_COMMAND_CHARS,
                    "description": "Bash command run in the session's sandbox at each fire.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": MIN_TIMEOUT_SECONDS,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "description": "Per-fire timeout (required on update — no implicit default).",
                },
                "max_output_bytes": {
                    "type": "integer",
                    "minimum": MIN_MAX_OUTPUT_BYTES,
                    "maximum": MAX_MAX_OUTPUT_BYTES,
                    "description": "Per-fire output cap (required on update — no implicit default).",
                },
            },
            "required": ["kind", "command", "timeout_seconds", "max_output_bytes"],
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
                    "description": "Message delivered to THIS session at fire time.",
                },
            },
            "required": ["kind", "content"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "wake_session"},
                "target_session_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Session id (sess_...) of ANOTHER session in this account to "
                        "wake at fire time."
                    ),
                },
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_WAKE_CONTENT_CHARS,
                    "description": (
                        "Message delivered as a user-role event to the target session, "
                        "waking it. Delivered verbatim."
                    ),
                },
            },
            "required": ["kind", "target_session_id", "content"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "workflow"},
                "workflow_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Id of the workflow to launch at each fire.",
                },
                "workflow_version": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": (
                        "Required on update — explicit null floats to the current "
                        "version; an integer must equal the workflow's current version "
                        "and is re-asserted at each fire (drift records an error)."
                    ),
                },
                "input_template": {
                    "description": (
                        "Required on update (explicit null = no payload; at most "
                        f"{MAX_INPUT_TEMPLATE_BYTES} serialized bytes). The run's "
                        "input is the envelope {'trigger': <firing context>, 'input': "
                        "<this template verbatim>}."
                    ),
                },
                "vault_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Required on update (explicit [] = none) — must be a subset of "
                        "this session's vaults, re-checked at every fire."
                    ),
                },
            },
            "required": ["kind", "workflow_id", "workflow_version", "input_template", "vault_ids"],
            "additionalProperties": False,
        },
    ],
}

TRIGGER_UPDATE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "description": "Name of the trigger to update.",
        },
        "source": _SOURCE_SCHEMA,
        "action": _ACTION_SCHEMA,
        "enabled": {"type": "boolean", "description": "New enabled state."},
        "metadata": {"type": "object", "description": "Replacement opaque tags."},
    },
    "required": ["name"],
    "additionalProperties": False,
}


async def trigger_update_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    name = arguments.get("name")
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    patch_data = {k: v for k, v in arguments.items() if k != "name"}
    update = TriggerUpdate.model_validate(patch_data)
    echo = await triggers_service.update_trigger(
        pool, session_id, name, update, account_id=account_id
    )
    return echo.model_dump(mode="json")


def _register() -> None:
    registry.register(
        name="trigger_update",
        description=TRIGGER_UPDATE_DESCRIPTION,
        parameters_schema=TRIGGER_UPDATE_PARAMETERS_SCHEMA,
        handler=trigger_update_handler,
    )


_register()
