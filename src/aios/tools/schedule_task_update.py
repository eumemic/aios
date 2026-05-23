"""The ``schedule_task_update`` tool — patch fields of a scheduled task.

Granular operation per #270 — accepts only the fields to change. ``name``
is the addressable identifier and cannot itself be changed. Toggling
``enabled`` true→false clears the task's ``next_fire``; false→true
recomputes it from now. Changing ``schedule`` also recomputes
``next_fire``.
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
    ScheduledTaskUpdate,
)
from aios.services import scheduled_tasks as scheduled_tasks_service
from aios.services import sessions as sessions_service
from aios.tools.registry import registry

SCHEDULE_TASK_UPDATE_DESCRIPTION = (
    "Update fields of an existing scheduled task by name. Omitted fields "
    "are left unchanged. Toggling ``enabled`` true→false pauses the task "
    "(clears next_fire); false→true resumes it (recomputes next_fire). "
    "Changing the schedule recomputes next_fire from now."
)

SCHEDULE_TASK_UPDATE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "description": "Name of the scheduled task to update.",
        },
        "schedule": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SCHEDULE_CHARS,
            "description": "New cron expression. Recomputes next_fire if changed.",
        },
        "command": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_COMMAND_CHARS,
            "description": "New bash command.",
        },
        "enabled": {
            "type": "boolean",
            "description": "New enabled state.",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": MIN_TIMEOUT_SECONDS,
            "maximum": MAX_TIMEOUT_SECONDS,
            "description": "New per-fire timeout.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


async def schedule_task_update_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    name = arguments.get("name")
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    patch_data = {k: v for k, v in arguments.items() if k != "name"}
    update = ScheduledTaskUpdate.model_validate(patch_data)
    echo = await scheduled_tasks_service.update_task(
        pool, session_id, name, update, account_id=account_id
    )
    return echo.model_dump(mode="json")


def _register() -> None:
    registry.register(
        name="schedule_task_update",
        description=SCHEDULE_TASK_UPDATE_DESCRIPTION,
        parameters_schema=SCHEDULE_TASK_UPDATE_PARAMETERS_SCHEMA,
        handler=schedule_task_update_handler,
    )


_register()
