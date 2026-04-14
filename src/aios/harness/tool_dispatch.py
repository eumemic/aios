"""Async fire-and-forget tool dispatch.

When the step function gets an assistant message with ``tool_calls``,
it calls :func:`launch_tool_calls` which spawns one ``asyncio.Task``
per tool call. Each task:

1. Parses the arguments, looks up the handler, and invokes it.
2. Appends a tool-role event to the session log (success or error).
3. Triggers the sweep so the next step picks up the result.

The contract: **every task MUST append exactly one tool-role event and
trigger the sweep in its finally block.** This is enforced by the
try/except/finally structure. Only a worker SIGKILL can break it —
and the periodic sweep recovers from that.

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
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolNotFoundError, registry

log = get_logger("aios.harness.tool_dispatch")


def _launch_tasks(
    session_id: str,
    tool_calls: list[dict[str, Any]],
    coro_factory: Any,
    *,
    prefix: str,
) -> None:
    """Shared launcher: spawn one asyncio task per tool call, register in task registry."""
    task_reg = runtime.require_task_registry()
    for call in tool_calls:
        call_id = call.get("id") or "unknown"
        task = asyncio.create_task(
            coro_factory(call),
            name=f"{prefix}:{session_id}:{call_id}",
        )
        task_reg.add(session_id, call_id, task)

        def _on_done(t: asyncio.Task[None], s: str = session_id, c: str = call_id) -> None:
            task_reg.remove(s, c)

        task.add_done_callback(_on_done)


def launch_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Launch each tool call as an asyncio task. Returns immediately."""
    _launch_tasks(
        session_id,
        tool_calls,
        lambda call: _execute_tool_async(pool, session_id, call),
        prefix="tool",
    )


async def _execute_tool_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
) -> None:
    """Execute one tool call: parse, invoke, append result, defer wake."""
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)

    try:
        # Parse arguments.
        arguments = _parse_arguments(raw_args)
        if arguments is None:
            bound_log.warning("tool.bad_arguments")
            await _append_tool_result(
                pool, session_id, call_id, name, error="arguments were not valid JSON"
            )
            return

        # Look up tool handler.
        try:
            tool = registry.get(name)
        except ToolNotFoundError as err:
            bound_log.warning("tool.not_registered")
            await _append_tool_result(pool, session_id, call_id, name, error=err.message)
            return

        # Invoke handler.
        result = await tool.handler(session_id, arguments)
        content_str = json.dumps(result, ensure_ascii=False)
        bound_log.info("tool.completed")
        await sessions_service.append_event(
            pool,
            session_id,
            "message",
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": content_str},
        )

    except asyncio.CancelledError:
        bound_log.info("tool.cancelled")
        await _append_tool_result(pool, session_id, call_id, name, error="cancelled")

    except Exception as err:
        bound_log.exception("tool.handler_failed")
        _evict_session_container(session_id)
        await _append_tool_result(
            pool, session_id, call_id, name, error=f"{type(err).__name__}: {err}"
        )

    finally:
        try:
            from aios.harness.sweep import wake_sessions_needing_inference

            await wake_sessions_needing_inference(
                pool, runtime.require_task_registry(), session_id=session_id
            )
        except Exception:
            bound_log.warning("tool.sweep_failed")


def _parse_arguments(raw_args: Any) -> dict[str, Any] | None:
    """Parse tool arguments from a JSON string or dict. Returns None on failure."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _append_tool_result(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call_id: str,
    name: str,
    *,
    error: str,
) -> None:
    """Append a tool-role error event."""
    content = json.dumps({"error": error}, ensure_ascii=False)
    await sessions_service.append_event(
        pool,
        session_id,
        "message",
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
            "is_error": True,
        },
    )


def _evict_session_container(session_id: str) -> None:
    """Best-effort eviction of the session's cached sandbox container."""
    if runtime.sandbox_registry is None:
        return
    runtime.sandbox_registry.evict(session_id)


# ── MCP tool dispatch ─────────────────────────────────────────────────────────


def launch_mcp_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
    mcp_server_map: dict[str, str],
) -> None:
    """Launch MCP tool calls as asyncio tasks. Returns immediately."""
    _launch_tasks(
        session_id,
        tool_calls,
        lambda call: _execute_mcp_tool_async(pool, session_id, call, mcp_server_map),
        prefix="mcp_tool",
    )


def _parse_mcp_tool_name(name: str) -> tuple[str, str]:
    """Parse ``mcp__<server_name>__<tool_name>`` into ``(server_name, tool_name)``.

    Raises ``ValueError`` on malformed names.
    """
    parts = name.split("__", 2)
    if len(parts) < 3 or not parts[1] or not parts[2]:
        raise ValueError(f"malformed MCP tool name: {name!r}")
    return parts[1], parts[2]


async def _execute_mcp_tool_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    mcp_server_map: dict[str, str],
) -> None:
    """Execute one MCP tool call: connect, invoke, append result, defer wake."""
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name: str = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)

    try:
        arguments = _parse_arguments(raw_args)
        if arguments is None:
            bound_log.warning("mcp_tool.bad_arguments")
            await _append_tool_result(
                pool, session_id, call_id, name, error="arguments were not valid JSON"
            )
            return

        server_name, tool_name = _parse_mcp_tool_name(name)
        url = mcp_server_map.get(server_name)
        if url is None:
            bound_log.warning("mcp_tool.server_not_found", server_name=server_name)
            await _append_tool_result(
                pool, session_id, call_id, name, error=f"MCP server {server_name!r} not found"
            )
            return

        from aios.mcp.client import call_mcp_tool, resolve_auth_headers

        crypto_box = runtime.require_crypto_box()
        headers = await resolve_auth_headers(pool, crypto_box, session_id, url)
        result = await call_mcp_tool(url, headers, tool_name, arguments)

        content_str = json.dumps(result, ensure_ascii=False)
        is_error = "error" in result
        event_data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content_str,
        }
        if is_error:
            event_data["is_error"] = True

        bound_log.info("mcp_tool.completed", is_error=is_error)
        await sessions_service.append_event(pool, session_id, "message", event_data)

    except asyncio.CancelledError:
        bound_log.info("mcp_tool.cancelled")
        await _append_tool_result(pool, session_id, call_id, name, error="cancelled")

    except Exception as err:
        bound_log.exception("mcp_tool.handler_failed")
        await _append_tool_result(
            pool, session_id, call_id, name, error=f"{type(err).__name__}: {err}"
        )

    finally:
        try:
            from aios.harness.sweep import wake_sessions_needing_inference

            await wake_sessions_needing_inference(
                pool, runtime.require_task_registry(), session_id=session_id
            )
        except Exception:
            bound_log.warning("mcp_tool.sweep_failed")
