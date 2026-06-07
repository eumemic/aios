"""The ``return`` / ``error`` completion tools — a workflow agent child's only,
explicit way to finish (completion is NEVER inferred from idle — §3.5).

Injected ONLY into a workflow child (``origin='background'`` with a
``parent_run_id``; see ``compute_step_prelude``). The handler is the completion
writer: it appends the single ``workflow_child_done`` marker on the child,
soft-archives the child (making it genuinely terminal via ``archived_at``, not
``status``), and wakes the parent run to harvest the marker. The child's result
lives in the marker, so no signal row is needed — the parent's harvest reads
markers, and the periodic ``wf_runs`` sweep is the lost-wake backstop.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.harness import runtime
from aios.services.wake import defer_run_wake
from aios.tools.registry import ToolResult, registry

RETURN_TOOL_NAME = "return"
ERROR_TOOL_NAME = "error"

RETURN_DESCRIPTION = (
    "Finish this task and return a value to the workflow that spawned you. Call "
    "this exactly once when your work is done; `value` is the result the workflow "
    "receives. After calling it you are finished — do not keep working."
)
ERROR_DESCRIPTION = (
    "Finish this task as a failure. Call this when you cannot complete the task; "
    "`message` explains why. The workflow receives an errored result."
)

_RETURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"value": {"description": "the result returned to the workflow (any JSON)"}},
    "required": ["value"],
    "additionalProperties": False,
}
_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string", "description": "why the task failed"}},
    "required": ["message"],
    "additionalProperties": False,
}

_NOT_A_CHILD = ToolResult(
    content="return/error is only available to a workflow agent child", is_error=True
)


async def _finish(
    session_id: str, *, is_error: bool, result: Any, error: dict[str, Any] | None
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        ctx = await queries.get_session_workflow_context(conn, session_id)
    if ctx is None:
        return _NOT_A_CHILD
    account_id, parent_run_id = ctx
    if parent_run_id is None:
        return _NOT_A_CHILD  # fail closed — never signal a NULL parent run

    async with pool.acquire() as conn, conn.transaction():
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={
                "event": "workflow_child_done",
                "is_error": is_error,
                "result": result,
                "error": error,
            },
        )
        await queries.set_session_archived(conn, session_id, account_id=account_id)
    # After commit: wake the parent run to harvest the marker. The periodic
    # wf_runs sweep is the durable backstop if this wake is lost.
    await defer_run_wake(parent_run_id)
    return {"status": "errored" if is_error else "returned"}


async def return_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any] | ToolResult:
    return await _finish(session_id, is_error=False, result=arguments.get("value"), error=None)


async def error_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any] | ToolResult:
    return await _finish(
        session_id, is_error=True, result=None, error={"message": arguments.get("message")}
    )


def workflow_completion_tool_specs() -> list[dict[str, Any]]:
    """The chat-completions tool entries for ``return``/``error`` — injected into
    a workflow child's tool list by ``compute_step_prelude``."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }
        for tool in (registry.get(RETURN_TOOL_NAME), registry.get(ERROR_TOOL_NAME))
    ]


def _register() -> None:
    registry.register(
        name=RETURN_TOOL_NAME,
        description=RETURN_DESCRIPTION,
        parameters_schema=_RETURN_SCHEMA,
        handler=return_handler,
        transport="agent_tool",
    )
    registry.register(
        name=ERROR_TOOL_NAME,
        description=ERROR_DESCRIPTION,
        parameters_schema=_ERROR_SCHEMA,
        handler=error_handler,
        transport="agent_tool",
    )


_register()
