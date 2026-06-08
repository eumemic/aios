"""The ``return`` / ``error`` tools — a workflow agent child's **response** to the
request it was invoked with (a response is NEVER inferred from idle).

Injected ONLY into a workflow child (``origin='background'`` with a
``parent_run_id``; see ``compute_step_prelude``). A child is spawned with a
**request** (its first user message, stamped ``metadata.request``); ``return``/
``error`` answer it. The handler:

* writes the request's response **exactly once** (``write_response_if_absent`` —
  first-writer-wins, so a ``return``+``error`` batch, a model double-call, or a
  ``return`` racing the totality backstop all collapse to one response), and
* wakes the caller (the run) to harvest it.

It does **not** archive or terminate the child. Responding resumes the caller
regardless of the child's subsequent fate; the child carries on (a fresh
``agent()`` child has nothing else to do, so it quiesces, and run-end reclaim
archives it — off the correctness path). The response is the durable record the
caller's harvest reads; the periodic ``wf_runs`` sweep is the lost-wake backstop.
"""

from __future__ import annotations

from typing import Any

from aios.db import queries
from aios.harness import runtime
from aios.services.wake import defer_run_wake
from aios.tools.registry import ToolResult, openai_tool_entry, registry

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
_NO_OPEN_REQUEST = ToolResult(content="there is no open request to answer", is_error=True)


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
        request_id = await queries.get_open_request_id(conn, session_id, account_id=account_id)
        if request_id is None:
            return _NO_OPEN_REQUEST  # a child, but its request is already answered / absent
        # Respond to the request — exactly once (first-writer-wins). NOT archive,
        # NOT terminate: responding resumes the caller; the child carries on (a
        # fresh agent() child simply has nothing else to do and quiesces; run-end
        # reclaim archives it). The response is the durable record the harvest reads.
        wrote = await queries.write_response_if_absent(
            conn,
            session_id,
            account_id=account_id,
            request_id=request_id,
            is_error=is_error,
            result=result,
            error=error,
        )
    # Wake the caller to harvest the response only when we actually wrote it; a
    # duplicate response (the request was already answered) is a no-op — the first
    # response already woke the caller. The periodic wf_runs sweep is the backstop.
    if wrote:
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
    return [openai_tool_entry(registry.get(name)) for name in (RETURN_TOOL_NAME, ERROR_TOOL_NAME)]


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
