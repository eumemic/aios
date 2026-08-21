"""The ``wake_self`` tool — append a user-role message to *this* session.

Self-wake primitive for cron-fired bash commands (and any other
sandbox-side caller): escalate a tool result, finding, or external
event into a model wake.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.errors import AiosError
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services.trigger_lint import OBSERVED_WAKE_WARNING, observed_wake_is_noisy
from aios.tools.registry import registry


class WakeSelfArgumentError(AiosError):
    error_type = "wake_self_argument_error"
    status_code = 400


WAKE_SELF_DESCRIPTION = (
    "Append content as a user-role message to your own session and "
    "schedule the next step. Use from inside a trigger's sandbox_command "
    "(or any sandbox bash command) to escalate a tool result, finding, or "
    "external event into a model wake. The appended message is delivered "
    "to your next model call as a normal user-role event. Available on "
    "both the model-tool surface and as "
    '``tool wake_self \'{"content":"..."}\'`` from inside the sandbox.'
)

WAKE_SELF_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": ["content"],
    "additionalProperties": False,
}


async def wake_self_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments.get("content")
    if not isinstance(content, str) or not content:
        raise WakeSelfArgumentError("content must be a non-empty string")

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    event = await sessions_service.tell_existing_session(
        pool, session_id, content=content, cause="message", account_id=account_id
    )
    # A workflow launched by a recurring trigger can wake its owner through an
    # agent child's ordinary model/builtin surface. Attribute that successful
    # wake to the originating fire just as the sandbox broker attributes CLI
    # wake_self calls; the durable run/session lineage is the observation token.
    warnings: list[tuple[str, str, str]] = []
    async with pool.acquire() as conn:
        origins = await queries.mark_trigger_run_woken_by_workflow_session(conn, session_id)
        for trigger_id, origin_account_id, owner_session_id, trigger_name in origins:
            outcomes = await queries.list_recent_trigger_wake_outcomes(conn, trigger_id)
            if observed_wake_is_noisy(outcomes) and not observed_wake_is_noisy(outcomes[1:]):
                warnings.append((origin_account_id, owner_session_id, trigger_name))
    for origin_account_id, owner_session_id, trigger_name in warnings:
        await sessions_service.tell_existing_session(
            pool,
            owner_session_id,
            content=f"[Trigger '{trigger_name}' warning: {OBSERVED_WAKE_WARNING}]",
            cause="message",
            account_id=origin_account_id,
        )
    return {
        "woken": True,
        "session_id": session_id,
        "event_id": event.id,
    }


def _register() -> None:
    registry.register(
        name="wake_self",
        description=WAKE_SELF_DESCRIPTION,
        parameters_schema=WAKE_SELF_PARAMETERS_SCHEMA,
        handler=wake_self_handler,
        transport="both",
    )


_register()
