"""The ``trigger_create`` tool — add a trigger to this session.

A trigger pairs a ``source`` (what fires it: a recurring ``cron`` schedule,
a one-shot ``fire_at``, or a reactive ``run_completion`` watch) with an
``action`` (what runs: a ``sandbox_command`` bash task that does NOT wake
the model, a ``wake_owner`` message delivered to this session that DOES wake
the model, or a ``workflow`` launch — deterministic, no model wake).

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
    MAX_INPUT_TEMPLATE_BYTES,
    MAX_MAX_OUTPUT_BYTES,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_TIMEOUT_SECONDS,
    MAX_WAKE_CONTENT_CHARS,
    MIN_MAX_OUTPUT_BYTES,
    MIN_TIMEOUT_SECONDS,
    RUN_TERMINAL_STATUSES,
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
    "in UTC); `{kind: 'one_shot', fire_at}` (fires once at an absolute UTC "
    "time, then self-deletes); `{kind: 'run_completion', workflow_id, "
    "statuses?}` (reactive — fires once per terminal completion of any run "
    "of that workflow; narrow `statuses` to e.g. ['errored'] for "
    "failure-only reactions).\n"
    "\n"
    "Actions: `{kind: 'sandbox_command', command}` runs bash in the "
    "session's sandbox WITHOUT waking the model; `{kind: 'wake_owner', "
    "content}` delivers `content` as a user-role message to THIS session, "
    "waking the model; `{kind: 'wake_session', target_session_id, content}` "
    "delivers `content` as a user-role message to ANOTHER same-account "
    "session (named by id), waking it — subject to the same wake-depth / "
    "per-pair-rate caps as the wake_session tool; `{kind: 'workflow', "
    "workflow_id, input_template?, "
    "workflow_version?, vault_ids?}` launches a run of that workflow — "
    "deterministic, no model wake; the run launches into this session's own "
    "environment with your authority (its surface and vaults are checked "
    "against yours at every fire).\n"
    "\n"
    "Use a sandbox_command for deterministic polling (file watchers, API "
    "checks, periodic syncs) without burning model tokens per fire; use "
    "wake_owner for a recurring or scheduled self-reminder; pair "
    "run_completion with a workflow action for fully deterministic "
    "run-to-run pipelines. Names must be unique per session."
)

# Create-side source/action schemas. trigger_update.py carries deliberately
# divergent copies (the Replace rule: optional-at-create fields required).
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
        {
            "type": "object",
            "properties": {
                "kind": {"const": "run_completion"},
                "workflow_id": {
                    "type": "string",
                    "description": (
                        "Id of a workflow under this account to watch. The trigger "
                        "fires once per terminal completion of any of its runs."
                    ),
                },
                "statuses": {
                    "type": "array",
                    "items": {"enum": list(RUN_TERMINAL_STATUSES)},
                    "minItems": 1,
                    "default": list(RUN_TERMINAL_STATUSES),
                    "description": (
                        "Which terminal statuses fire the trigger. Defaults to all "
                        "three; narrow to e.g. ['errored'] for failure-only reactions."
                    ),
                },
            },
            "required": ["kind", "workflow_id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "external_event"},
            },
            "required": ["kind"],
            "additionalProperties": False,
            "description": (
                "Reactive source fired by an authenticated inbound webhook. The "
                "POST body becomes input['trigger']['event'] of the fired run. A "
                "per-trigger ingest secret is minted server-side and returned ONCE "
                "as `ingest_token` on this create (POST to "
                "/v1/triggers/ingest/{ingest_token}); it is never re-readable — "
                "rotate via trigger_update."
            ),
        },
    ],
}

_SANDBOX_COMMAND_DESCRIPTION = (
    "Bash command run in the session's sandbox at each fire, WITHOUT waking "
    "the model. To escalate (wake the model with a user-role message), call "
    '``tool wake_self \'{"content":"<message to deliver to yourself>"}\'`` '
    "from inside the cron command."
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
                        "time, waking the model. No sandbox involved. Delivered verbatim "
                        "— a run_completion fire does not interpolate the completing "
                        "run's identity (use a workflow action to consume the event)."
                    ),
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
                    "description": "Id of a workflow under this account to launch at each fire.",
                },
                "workflow_version": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "default": None,
                    "description": (
                        "null (default): each fire runs the workflow's CURRENT version. "
                        "An integer must equal the workflow's current version when you "
                        "write the trigger and is re-asserted at each fire — if the "
                        "workflow has been edited since, the fire records an error "
                        "instead of running the unreviewed edit (re-pin after review)."
                    ),
                },
                "input_template": {
                    "default": None,
                    "description": (
                        "Arbitrary JSON (any type; null = no payload), at most "
                        f"{MAX_INPUT_TEMPLATE_BYTES} serialized bytes. The launched run's input is ALWAYS the "
                        "envelope {'trigger': <firing context>, 'input': <this template "
                        "verbatim>} — for run_completion fires the context carries the "
                        "completing run's id, status, output, and error kind under "
                        "trigger.run. A workflow built to be triggered reads "
                        "input['trigger'] and input['input']."
                    ),
                },
                "vault_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "Vaults to bind to the launched run — must be a subset of this "
                        "session's vaults, re-checked at every fire."
                    ),
                },
            },
            "required": ["kind", "workflow_id"],
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
    created = await triggers_service.add_trigger(pool, session_id, spec, account_id=account_id)
    # The dump carries ``ingest_token`` (the once-only ingest secret) for an
    # external_event source, ``None`` otherwise — the agent must see the minted
    # token to drive the ingress.
    return created.model_dump(mode="json")


def _register() -> None:
    registry.register(
        name="trigger_create",
        description=TRIGGER_CREATE_DESCRIPTION,
        parameters_schema=TRIGGER_CREATE_PARAMETERS_SCHEMA,
        handler=trigger_create_handler,
    )


_register()
