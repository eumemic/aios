"""The ``trigger_list`` tool — list this session's triggers.

Complements ``trigger_create`` / ``trigger_remove`` by giving the model a
direct read of each trigger's health: name, source, action, enabled state,
last fire status, and consecutive-failure count. This is the signal it needs
to notice a trigger that auto-disabled after repeated failures and decide
what to do next.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.registry import registry

TRIGGER_LIST_DESCRIPTION = (
    "List all triggers for this session, with each trigger's name, source kind, "
    "action kind, enabled state, last fire status, and consecutive failure count."
)

TRIGGER_LIST_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
    "required": [],
}


async def trigger_list_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    echoes = await triggers_service.list_triggers(pool, session_id, account_id=account_id)
    return {"triggers": [e.model_dump(mode="json") for e in echoes]}


def _register() -> None:
    registry.register(
        name="trigger_list",
        description=TRIGGER_LIST_DESCRIPTION,
        parameters_schema=TRIGGER_LIST_PARAMETERS_SCHEMA,
        handler=trigger_list_handler,
    )


_register()
