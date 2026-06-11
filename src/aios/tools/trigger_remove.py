"""The ``trigger_remove`` tool — remove a trigger from this session.

Idempotency note: removing a non-existent name raises ``NotFoundError``
(surfaced as a tool error). The model can list and confirm before remove if
uncertain — the ``trigger_list`` tool enumerates current entries, and
``Session.triggers`` in the system context also lists them.
"""

from __future__ import annotations

from typing import Any

from aios.harness import runtime
from aios.models.triggers import MAX_NAME_CHARS
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.registry import registry

TRIGGER_REMOVE_DESCRIPTION = (
    "Remove a trigger from this session by name. Raises NotFoundError if no "
    "trigger with that name exists. The trigger stops firing immediately; "
    "any in-flight fire completes."
)

TRIGGER_REMOVE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_NAME_CHARS,
            "description": "Name of the trigger to remove.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


async def trigger_remove_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    name = arguments["name"]
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    await triggers_service.remove_trigger(pool, session_id, name, account_id=account_id)
    return {"removed": True, "name": name}


def _register() -> None:
    registry.register(
        name="trigger_remove",
        description=TRIGGER_REMOVE_DESCRIPTION,
        parameters_schema=TRIGGER_REMOVE_PARAMETERS_SCHEMA,
        handler=trigger_remove_handler,
    )


_register()
