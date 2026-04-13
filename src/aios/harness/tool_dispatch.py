"""Async fire-and-forget tool dispatch.

When the step function gets an assistant message with ``tool_calls``,
it calls :func:`launch_tool_calls` which spawns one ``asyncio.Task``
per tool call. Each task:

1. Parses the arguments, looks up the handler, and invokes it.
2. Appends a tool-role event to the session log (success or error).
3. Defers a ``wake_session`` job so the next step picks up the result.

The contract: **every task MUST append exactly one tool-role event and
defer a wake in its finally block.** This is enforced by the
try/except/finally structure. Only a worker SIGKILL can break it.

Tool tasks run on the worker's event loop and outlive the procrastinate
job handler that spawned them. They're tracked in the per-worker
:class:`~aios.harness.task_registry.TaskRegistry` for cancellation
and shutdown support.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from aios.harness import runtime
from aios.harness.wake import defer_wake
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolNotFoundError, registry

log = get_logger("aios.harness.tool_dispatch")


def launch_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Launch each tool call as an asyncio task. Returns immediately."""
    task_reg = runtime.require_task_registry()
    for call in tool_calls:
        call_id = call.get("id") or "unknown"
        task = asyncio.create_task(
            _execute_tool_async(pool, session_id, call),
            name=f"tool:{session_id}:{call_id}",
        )
        task_reg.add(session_id, call_id, task)

        def _on_done(t: asyncio.Task[None], s: str = session_id, c: str = call_id) -> None:
            task_reg.remove(s, c)

        task.add_done_callback(_on_done)


async def _execute_tool_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
) -> None:
    """Execute one tool call: parse, invoke, wait for step, append result, defer wake.

    The "wait for step" gate ensures that tool results are appended AFTER
    the assistant message from the step that launched them. This preserves
    perception order in the event log: if a tool completes during inference,
    the model's response (based on a "pending" snapshot) appears first, and
    the real tool result appears after — so the next step sees it as new.
    """
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)

    # This will hold the event data to append after the gate opens.
    result_data: dict[str, Any] | None = None

    try:
        # Parse arguments.
        arguments = _parse_arguments(raw_args)
        if arguments is None:
            bound_log.warning("tool.bad_arguments")
            result_data = _error_data(call_id, name, "arguments were not valid JSON")
            return

        # Look up tool handler.
        try:
            tool = registry.get(name)
        except ToolNotFoundError as err:
            bound_log.warning("tool.not_registered")
            result_data = _error_data(call_id, name, err.message)
            return

        # Invoke handler.
        result = await tool.handler(session_id, arguments)
        content_str = json.dumps(result, ensure_ascii=False)
        bound_log.info("tool.completed")
        result_data = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content_str,
        }

    except asyncio.CancelledError:
        bound_log.info("tool.cancelled")
        result_data = _error_data(call_id, name, "cancelled")

    except Exception as err:
        bound_log.exception("tool.handler_failed")
        _evict_session_container(session_id)
        result_data = _error_data(call_id, name, f"{type(err).__name__}: {err}")

    finally:
        # Wait for the step that launched us to finish appending its
        # assistant message. If the step already finished, this returns
        # immediately.
        from aios.harness.loop import get_step_done_event

        gate = get_step_done_event(session_id)
        if gate is not None and not gate.is_set():
            await gate.wait()

        # Append the tool result.
        if result_data is not None:
            await sessions_service.append_event(pool, session_id, "message", result_data)

        # Trigger the next step.
        try:
            await defer_wake(session_id, cause="tool_result")
        except Exception:
            bound_log.warning("tool.defer_wake_failed")


def _parse_arguments(raw_args: Any) -> dict[str, Any] | None:
    """Parse tool arguments from a JSON string or dict. Returns None on failure."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_data(call_id: str, name: str, error: str) -> dict[str, Any]:
    """Build an error tool-result data dict (does not append — the finally block does)."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps({"error": error}, ensure_ascii=False),
        "is_error": True,
    }


def _evict_session_container(session_id: str) -> None:
    """Best-effort eviction of the session's cached sandbox container."""
    if runtime.sandbox_registry is None:
        return
    runtime.sandbox_registry.evict(session_id)
