"""The ``schedule_task_remove`` tool — remove a scheduled task from this session.

Idempotency note: removing a non-existent name raises ``NotFoundError``
(surfaced as a tool error). The model can list and confirm before remove
if uncertain — there is no ``list`` tool yet, but ``Session.scheduled_tasks``
in the system context lists current entries.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.scheduled_tasks import MAX_NAME_CHARS
from aios.services import scheduled_tasks as scheduled_tasks_service
from aios.services import sessions as sessions_service
from aios.tools.registry import registry

SCHEDULE_TASK_REMOVE_DESCRIPTION = (
    "Remove a scheduled task from this session by name. Raises "
    "NotFoundError if no task with that name exists. The task stops "
    "firing immediately; any in-flight fire completes."
)

SCHEDULE_TASK_REMOVE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "description": "Name of the scheduled task to remove.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


async def schedule_task_remove_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    name = arguments["name"]
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    await scheduled_tasks_service.remove_task(pool, session_id, name, account_id=account_id)
    return {"removed": True, "name": name}


def _register() -> None:
    registry.register(
        name="schedule_task_remove",
        description=SCHEDULE_TASK_REMOVE_DESCRIPTION,
        parameters_schema=SCHEDULE_TASK_REMOVE_PARAMETERS_SCHEMA,
        handler=schedule_task_remove_handler,
    )


_register()
