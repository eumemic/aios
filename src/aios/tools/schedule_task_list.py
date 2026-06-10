"""The ``schedule_task_list`` tool — list this session's scheduled tasks.

Complements ``schedule_task_add`` / ``schedule_task_remove`` (#798) by giving
the model a direct read of each task's health: name, enabled state, last fire
status, and consecutive-failure count. This is the signal it needs to notice a
task that auto-disabled after repeated failures and decide what to do next.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.services import scheduled_tasks as scheduled_tasks_service
from aios.services import sessions as sessions_service
from aios.tools.registry import registry

SCHEDULE_TASK_LIST_DESCRIPTION = (
    "List all scheduled tasks for this session, with each task's name, "
    "enabled state, last fire status, and consecutive failure count."
)

SCHEDULE_TASK_LIST_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
    "required": [],
}


async def schedule_task_list_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    echoes = await scheduled_tasks_service.list_tasks(pool, session_id, account_id=account_id)
    return {"tasks": [e.model_dump(mode="json") for e in echoes]}


def _register() -> None:
    registry.register(
        name="schedule_task_list",
        description=SCHEDULE_TASK_LIST_DESCRIPTION,
        parameters_schema=SCHEDULE_TASK_LIST_PARAMETERS_SCHEMA,
        handler=schedule_task_list_handler,
    )


_register()
