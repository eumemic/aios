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
import jsonschema  # type: ignore[import-untyped]

from aios.harness import runtime
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolNotFoundError, ToolResult, registry

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
    """Execute one tool call: parse, invoke, append result, defer wake.

    Brackets the lifecycle in a ``tool_execute_*`` span pair (issue #78).
    """
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)

    span_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "tool_execute_start",
            "tool_call_id": call_id,
            "tool_name": name,
        },
    )
    is_error = False

    try:
        # Parse arguments.
        arguments = _parse_arguments(raw_args)
        if arguments is None:
            bound_log.warning("tool.bad_arguments")
            is_error = True
            await _append_tool_result(
                pool, session_id, call_id, name, error="arguments were not valid JSON"
            )
            return

        # Look up tool handler.
        try:
            tool = registry.get(name)
        except ToolNotFoundError as err:
            bound_log.warning("tool.not_registered")
            is_error = True
            await _append_tool_result(pool, session_id, call_id, name, error=err.message)
            return

        # Validate arguments against the tool's parameters_schema before
        # dispatch.  Weaker models often emit tool calls with wrong
        # parameter names or types; without validation, the handler runs
        # against partially-malformed input (e.g. a missing required key
        # becomes ``None``, a bad-name key becomes an ignored extra) and
        # silently returns a no-op-shaped result.  The model sees the
        # no-op as "worked," loops forever.  Surfacing the schema errors
        # explicitly gives the model feedback to self-correct.
        schema_error = _validate_arguments(arguments, tool.parameters_schema)
        if schema_error is not None:
            bound_log.info("tool.schema_error", error=schema_error)
            is_error = True
            await _append_tool_result(pool, session_id, call_id, name, error=schema_error)
            return

        # Invoke handler.  Handlers return either a plain dict (JSON-
        # encoded into the tool message's content) or a ToolResult
        # (carries per-event metadata and/or a plain-string content).
        result = await tool.handler(session_id, arguments)
        event_data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
        }
        if isinstance(result, ToolResult):
            if isinstance(result.content, str):
                event_data["content"] = result.content
            else:
                event_data["content"] = json.dumps(result.content, ensure_ascii=False)
            if result.metadata:
                event_data["metadata"] = result.metadata
            if result.is_error:
                event_data["is_error"] = True
                is_error = True
        else:
            event_data["content"] = json.dumps(result, ensure_ascii=False)
        bound_log.info("tool.completed")
        await sessions_service.append_event(
            pool,
            session_id,
            "message",
            event_data,
        )

    except asyncio.CancelledError:
        bound_log.info("tool.cancelled")
        is_error = True
        await _append_tool_result(pool, session_id, call_id, name, error="cancelled")

    except Exception as err:
        bound_log.exception("tool.handler_failed")
        is_error = True
        _evict_session_container(session_id)
        await _append_tool_result(
            pool, session_id, call_id, name, error=f"{type(err).__name__}: {err}"
        )

    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "tool_execute_end",
                "tool_execute_start_id": span_start.id,
                "tool_call_id": call_id,
                "tool_name": name,
                "is_error": is_error,
            },
        )
        await _trigger_sweep(pool, session_id, bound_log)


def _parse_arguments(raw_args: Any) -> dict[str, Any] | None:
    """Parse tool arguments from a JSON string or dict. Returns None on failure."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate ``arguments`` against the tool's JSON Schema.

    Returns ``None`` on success, or a human-readable error string that
    enumerates every validation failure (missing required keys,
    unexpected extra keys, wrong types).  The string is what ends up in
    the tool_result's ``error`` body, so the model sees every issue at
    once and can self-correct without iterating one-at-a-time.

    The schema is the same dict registered with the tool and sent to
    the model as the tool's ``parameters``, so a mismatch genuinely
    indicates the model didn't follow the contract — not a framework
    bug.  Surfacing specific paths (e.g. ``foo.bar[2]``) and
    passed-value previews keeps the feedback actionable.
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    lines = [
        f"Arguments failed schema validation. You sent: {json.dumps(arguments)}",
        "Errors:",
    ]
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path) or "<root>"
        lines.append(f"  - at {path}: {err.message}")
    lines.append("Look at the tool's `parameters` schema for the correct shape and retry.")
    return "\n".join(lines)


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


async def _trigger_sweep(
    pool: asyncpg.Pool[Any],
    session_id: str,
    bound_log: Any,
) -> None:
    """Run the sweep for this session. Called from the finally block of
    every tool task — both built-in and MCP.

    Brackets the sweep with a ``sweep_start``/``sweep_end`` span pair
    (``site="tail"``). The tail site exercises the full composite sweep
    (ghost repair + find + defer), so ``sweep_end`` carries the real
    ``repaired_ghosts`` and ``woken_sessions`` counts from
    :class:`SweepResult`.
    """
    from aios.harness.sweep import SweepResult, wake_sessions_needing_inference

    sweep_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "sweep_start", "site": "tail"},
    )
    result = SweepResult(repaired_ghosts=0, woken_sessions=0)
    try:
        try:
            result = await wake_sessions_needing_inference(
                pool, runtime.require_task_registry(), session_id=session_id
            )
        except Exception:
            bound_log.warning("tool.sweep_failed")
    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "sweep_end",
                "sweep_start_id": sweep_start.id,
                "repaired_ghosts": result.repaired_ghosts,
                "woken_sessions": result.woken_sessions,
            },
        )


# ── MCP tool dispatch ─────────────────────────────────────────────────────────


def launch_mcp_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
    mcp_server_map: dict[str, str],
    *,
    focal_channel: str | None = None,
) -> None:
    """Launch MCP tool calls as asyncio tasks. Returns immediately.

    ``focal_channel`` is the session's focal at the moment these calls
    were emitted (captured at step top in ``run_session_step`` so a
    concurrent ``switch_channel`` in the same batch cannot race the
    ``chat_id`` injection).  Passed through to each task so
    connection-provided tools can stamp ``_meta`` with the suffix.
    """
    _launch_tasks(
        session_id,
        tool_calls,
        lambda call: _execute_mcp_tool_async(
            pool, session_id, call, mcp_server_map, focal_channel=focal_channel
        ),
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
    *,
    focal_channel: str | None = None,
) -> None:
    """Execute one MCP tool call: connect, invoke, append result, defer wake.

    For connection-provided servers (name in the reserved ``conn_``
    namespace), the focal-channel suffix is stamped into the JSON-RPC
    request's ``_meta`` so the connector can resolve its
    connector-specific chat identifiers without the model having to
    pass them explicitly.  The ``focal_channel`` snapshot is emission-
    time — a concurrent ``switch_channel`` in the same assistant batch
    does not race this injection.
    """
    from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX

    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name: str = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)

    span_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "tool_execute_start",
            "tool_call_id": call_id,
            "tool_name": name,
        },
    )
    is_error = False

    try:
        arguments = _parse_arguments(raw_args)
        if arguments is None:
            bound_log.warning("mcp_tool.bad_arguments")
            is_error = True
            await _append_tool_result(
                pool, session_id, call_id, name, error="arguments were not valid JSON"
            )
            return

        server_name, tool_name = _parse_mcp_tool_name(name)
        url = mcp_server_map.get(server_name)
        if url is None:
            bound_log.warning("mcp_tool.server_not_found", server_name=server_name)
            is_error = True
            await _append_tool_result(
                pool, session_id, call_id, name, error=f"MCP server {server_name!r} not found"
            )
            return

        from aios.harness.channels import FOCAL_CHANNEL_META_KEY, focal_channel_path
        from aios.mcp.client import call_mcp_tool, resolve_auth_for_url

        meta: dict[str, Any] | None = None
        if server_name.startswith(CONNECTION_SERVER_NAME_PREFIX):
            suffix = focal_channel_path(focal_channel)
            if suffix is None:
                # The model shouldn't be able to call a conn_* tool while
                # focal is NULL — loop.py filters them out of the tool
                # list in that state — but defend in depth if it slips
                # through (stale tool_calls, etc.).
                is_error = True
                await _append_tool_result(
                    pool,
                    session_id,
                    call_id,
                    name,
                    error=(
                        "connection-provided tools require a focal channel; "
                        "call switch_channel first"
                    ),
                )
                return
            meta = {FOCAL_CHANNEL_META_KEY: suffix}

        crypto_box = runtime.require_crypto_box()
        headers = await resolve_auth_for_url(pool, crypto_box, session_id, url)
        result = await call_mcp_tool(url, headers, tool_name, arguments, meta=meta)

        content_str = json.dumps(result, ensure_ascii=False)
        mcp_is_error = "error" in result
        event_data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content_str,
        }
        if mcp_is_error:
            event_data["is_error"] = True
            is_error = True

        bound_log.info("mcp_tool.completed", is_error=mcp_is_error)
        await sessions_service.append_event(pool, session_id, "message", event_data)

    except asyncio.CancelledError:
        bound_log.info("mcp_tool.cancelled")
        is_error = True
        await _append_tool_result(pool, session_id, call_id, name, error="cancelled")

    except Exception as err:
        bound_log.exception("mcp_tool.handler_failed")
        is_error = True
        await _append_tool_result(
            pool, session_id, call_id, name, error=f"{type(err).__name__}: {err}"
        )

    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "tool_execute_end",
                "tool_execute_start_id": span_start.id,
                "tool_call_id": call_id,
                "tool_name": name,
                "is_error": is_error,
            },
        )
        await _trigger_sweep(pool, session_id, bound_log)
