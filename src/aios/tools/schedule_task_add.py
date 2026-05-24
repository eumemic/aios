"""The ``schedule_task_add`` tool — add a cron-fired bash task to this session.

Hand-written precursor to the autogen'd self-state tools envisaged in
#652. Per #270 we deliberately expose granular ops only — no whole-list
``set``. Each entry is identified by ``name`` (unique per session).

Fires run in the session's sandbox without waking the model; bash must
explicitly escalate via the broker's ``sessions/messages`` endpoint to
deliver a user-role event back to the session.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.scheduled_tasks import (
    MAX_COMMAND_CHARS,
    MAX_NAME_CHARS,
    MAX_SCHEDULE_CHARS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    ScheduledTaskCreate,
)
from aios.services import scheduled_tasks as scheduled_tasks_service
from aios.services import sessions as sessions_service
from aios.tools.registry import registry

SCHEDULE_TASK_ADD_DESCRIPTION = (
    "Add a cron-fired bash task to this session. The task runs at the "
    "specified schedule in the session's sandbox WITHOUT waking the "
    "model. To escalate (wake the model with a user-role message), the "
    "bash command must POST to the broker. The canonical invocation, "
    "which works in both TCP and Unix-socket broker transports (the "
    "broker exposes ``unix://`` or ``http://`` via TOOL_BROKER_URL), is:\n"
    "\n"
    '  curl -fsS "$TOOL_BROKER_URL/v1/$TOOL_BROKER_SECRET/sessions/messages" \\\n'
    "       -X POST -H 'Content-Type: application/json' \\\n"
    '       -d \'{"content":"<message to deliver to yourself>"}\'\n'
    "\n"
    "Use this primitive for deterministic polling (file watchers, API "
    "checks, periodic state syncs) without burning model tokens per "
    "fire. Names must be unique per session; cron expressions are "
    "standard 5-field, UTC."
)

SCHEDULE_TASK_ADD_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "pattern": r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
            "description": "Unique identifier for this task within the session.",
        },
        "schedule": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SCHEDULE_CHARS,
            "description": "Standard 5-field cron expression in UTC (e.g. '*/5 * * * *').",
        },
        "command": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_COMMAND_CHARS,
            "description": "Bash command to run in the session's sandbox at each fire.",
        },
        "enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether the task fires. Set false to add a paused entry.",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": MIN_TIMEOUT_SECONDS,
            "maximum": MAX_TIMEOUT_SECONDS,
            "default": 300,
            "description": "Per-fire timeout. The fire is killed if it runs longer.",
        },
    },
    "required": ["name", "schedule", "command"],
    "additionalProperties": False,
}


async def schedule_task_add_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    spec = ScheduledTaskCreate.model_validate(arguments)
    echo = await scheduled_tasks_service.add_task(pool, session_id, spec, account_id=account_id)
    return echo.model_dump(mode="json")


def _register() -> None:
    registry.register(
        name="schedule_task_add",
        description=SCHEDULE_TASK_ADD_DESCRIPTION,
        parameters_schema=SCHEDULE_TASK_ADD_PARAMETERS_SCHEMA,
        handler=schedule_task_add_handler,
    )


_register()
