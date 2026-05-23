"""Async fire-and-forget tool dispatch (model path).

When the step function gets an assistant message with ``tool_calls``,
it calls :func:`launch_tool_calls` which spawns one ``asyncio.Task``
per tool call. Each task:

1. Invokes the tool via :func:`aios.tools.invoke.invoke_builtin`
   (parse, lookup, validate, call) — the same pure core the
   sandbox CLI broker drives.
2. Appends a tool-role event to the session log (success or error).
3. Triggers the sweep so the next step picks up the result.

The contract: **every task MUST append exactly one tool-role event and
trigger the sweep before returning.** This is enforced by the
:func:`_tool_lifecycle` async context manager, which both dispatch
paths drive. Only a worker SIGKILL can break it — and the periodic
sweep recovers from that.

Tool tasks run on the worker's event loop and outlive the procrastinate
job handler that spawned them. They're tracked in the per-worker
:class:`~aios.harness.task_registry.TaskRegistry` for cancellation
and shutdown support.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg

from aios.harness import runtime
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail, invoke_builtin, parse_arguments
from aios.tools.registry import ToolResult

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


@dataclass
class _ToolCall:
    """Fields extracted from a tool_call dict for use inside the lifecycle."""

    call_id: str
    name: str
    raw_args: Any
    bound_log: Any
    is_error: bool = False


@asynccontextmanager
async def _tool_lifecycle(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    *,
    account_id: str,
    log_prefix: str,
    on_exception: Callable[[str], None] | None = None,
) -> AsyncIterator[_ToolCall]:
    """Bracket a tool call with the ``tool_execute_*`` span pair plus
    try/except/finally + tail sweep (#78).  ``on_exception`` runs on
    generic ``Exception`` only — built-in passes the sandbox-eviction
    hook; MCP leaves it ``None`` because it doesn't use the sandbox.
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
        account_id=account_id,
    )
    tc = _ToolCall(call_id=call_id, name=name, raw_args=raw_args, bound_log=bound_log)
    try:
        yield tc
    except asyncio.CancelledError:
        bound_log.info(f"{log_prefix}.cancelled")
        tc.is_error = True
        await _append_tool_result(
            pool, session_id, call_id, name, account_id=account_id, error="cancelled"
        )
    except ToolBail as err:
        bound_log.info(f"{log_prefix}.bail", error=str(err))
        tc.is_error = True
        await _append_tool_result(
            pool, session_id, call_id, name, account_id=account_id, error=str(err)
        )
    except Exception as err:
        bound_log.exception(f"{log_prefix}.handler_failed")
        tc.is_error = True
        if on_exception is not None:
            on_exception(session_id)
        await _append_tool_result(
            pool,
            session_id,
            call_id,
            name,
            account_id=account_id,
            error=f"{type(err).__name__}: {err}",
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
                "is_error": tc.is_error,
            },
            account_id=account_id,
        )
        await _trigger_sweep(pool, session_id, account_id=account_id)


def launch_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
    *,
    account_id: str,
) -> None:
    """Launch each tool call as an asyncio task. Returns immediately."""
    _launch_tasks(
        session_id,
        tool_calls,
        lambda call: _execute_tool_async(pool, session_id, call, account_id=account_id),
        prefix="tool",
    )


async def _execute_tool_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    *,
    account_id: str,
) -> None:
    """Execute one built-in tool call via the shared invoke core, then
    append the tool-role result event.

    The pure core (parse → lookup → validate → call) is shared with the
    sandbox CLI broker via :func:`aios.tools.invoke.invoke_builtin`;
    only the event-append + sweep + sandbox eviction are model-path
    specific (wrapped here by :func:`_tool_lifecycle`).
    """
    async with _tool_lifecycle(
        pool,
        session_id,
        call,
        account_id=account_id,
        log_prefix="tool",
        on_exception=_evict_session_container,
    ) as tc:
        result = await invoke_builtin(session_id, tc.name, tc.raw_args)
        event_data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tc.call_id,
            "name": tc.name,
        }
        if isinstance(result, ToolResult):
            if isinstance(result.content, (str, list)):
                event_data["content"] = result.content
            else:
                event_data["content"] = json.dumps(result.content, ensure_ascii=False)
            if result.metadata:
                event_data["metadata"] = result.metadata
            if result.is_error:
                event_data["is_error"] = True
                tc.is_error = True
        else:
            event_data["content"] = json.dumps(result, ensure_ascii=False)
        tc.bound_log.info("tool.completed")
        await _append_tool_result_event(
            pool,
            session_id,
            tc.call_id,
            event_data,
            account_id=account_id,
        )


async def _append_tool_result_event(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_call_id: str,
    event_data: dict[str, Any],
    *,
    account_id: str,
) -> None:
    """Append a ``role:"tool"`` event, dedup-guarded on ``tool_call_id``.

    The worker's tool task lifecycle (success-, error-, schema-,
    cancel-paths) all funnel through here so a tool-role event that
    already landed via the operator path (``confirm_tool_deny`` →
    ``services.sessions.append_tool_result``) is not silently
    overwritten by the late worker result.

    Specifically: an ``always_allow`` builtin (or any tool the harness
    dispatches without the always_ask confirmation gate) can have its
    operator deny commit while the task is in-flight; without dedup
    here, the worker's late append lands a SECOND tool-role event for
    the same ``tool_call_id`` and the context builder's
    ``real_results[tcid] = e.data`` keying clobbers the deny — the
    next prompt carries the tool's successful output as if no deny had
    happened (silent failure symmetric to PR #535).

    Transaction + ``SELECT ... FOR UPDATE`` mirrors
    :func:`services.sessions.append_tool_result` so the worker-vs-API
    race serialises through the same session-row lock as the operator
    path; "first commit wins, second commit no-ops" applies to either
    ordering.
    """
    from aios.db import queries

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE",
            session_id,
            account_id,
        )
        existing = await queries.find_tool_result_event(
            conn, session_id, tool_call_id, account_id=account_id
        )
        if existing is not None:
            existing_is_error = bool(existing.data.get("is_error", False))
            log.warning(
                "tool.result_dedup_skip",
                session_id=session_id,
                tool_call_id=tool_call_id,
                existing_is_error=existing_is_error,
                attempted_is_error=bool(event_data.get("is_error", False)),
            )
            return
        await queries.append_event(
            conn,
            session_id=session_id,
            kind="message",
            data=event_data,
            account_id=account_id,
        )


async def _append_tool_result(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call_id: str,
    name: str,
    *,
    account_id: str,
    error: str,
) -> None:
    """Append a tool-role error event (dedup-guarded — see
    :func:`_append_tool_result_event`)."""
    content = json.dumps({"error": error}, ensure_ascii=False)
    await _append_tool_result_event(
        pool,
        session_id,
        call_id,
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
            "is_error": True,
        },
        account_id=account_id,
    )


def _evict_session_container(session_id: str) -> None:
    """Best-effort eviction of the session's cached sandbox container."""
    if runtime.sandbox_registry is None:
        return
    runtime.sandbox_registry.evict(session_id)


async def _trigger_sweep(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> None:
    """Run the sweep for this session. Called from the finally block of
    every tool task — both built-in and MCP.
    """
    from aios.harness.sweep import SweepResult, wake_sessions_needing_inference

    sweep_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "sweep_start", "site": "tail"},
        account_id=account_id,
    )
    result = SweepResult(repaired_ghosts=0, woken_sessions=0)
    try:
        result = await wake_sessions_needing_inference(
            pool, runtime.require_task_registry(), session_id=session_id
        )
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
            account_id=account_id,
        )


# ── MCP tool dispatch ─────────────────────────────────────────────────────────


def launch_mcp_tool_calls(
    pool: asyncpg.Pool[Any],
    session_id: str,
    tool_calls: list[dict[str, Any]],
    mcp_server_map: dict[str, str],
    *,
    account_id: str,
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
            pool,
            session_id,
            call,
            mcp_server_map,
            focal_channel=focal_channel,
            account_id=account_id,
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
    account_id: str,
    focal_channel: str | None = None,
) -> None:
    """Execute one MCP tool call: connect, invoke, append result.

    The focal-channel suffix is stamped into the JSON-RPC request's
    ``_meta`` so connectors that need it can pull the chat-id without
    the model having to pass it; servers that don't care ignore unknown
    ``_meta`` keys per the MCP spec.  The ``focal_channel`` snapshot is
    emission-time — a concurrent ``switch_channel`` in the same
    assistant batch does not race this injection.
    """
    async with _tool_lifecycle(
        pool,
        session_id,
        call,
        account_id=account_id,
        log_prefix="mcp_tool",
    ) as tc:
        arguments = parse_arguments(tc.raw_args)
        if arguments is None:
            raise ToolBail("arguments were not valid JSON")

        try:
            server_name, tool_name = _parse_mcp_tool_name(tc.name)
        except ValueError as err:
            raise ToolBail(str(err)) from err

        from aios.harness.channels import (
            FOCAL_CHANNEL_META_KEY,
            SESSION_ID_META_KEY,
            focal_channel_path,
        )

        meta: dict[str, Any] = {SESSION_ID_META_KEY: session_id}
        suffix = focal_channel_path(focal_channel)
        if suffix is not None:
            meta[FOCAL_CHANNEL_META_KEY] = suffix

        url = mcp_server_map.get(server_name)
        if url is None:
            raise ToolBail(f"MCP server {server_name!r} not found")

        from aios.mcp.client import call_mcp_tool, resolve_auth_for_target_url

        crypto_box = runtime.require_crypto_box()
        vault_id, headers = await resolve_auth_for_target_url(
            pool, crypto_box, session_id, url, account_id=account_id
        )
        result = await call_mcp_tool(url, vault_id, headers, tool_name, arguments, meta=meta)

        mcp_is_error = "error" in result
        event_data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tc.call_id,
            "name": tc.name,
            "content": json.dumps(result, ensure_ascii=False),
        }
        if mcp_is_error:
            event_data["is_error"] = True
            tc.is_error = True

        tc.bound_log.info("mcp_tool.completed", is_error=mcp_is_error)
        await _append_tool_result_event(
            pool, session_id, tc.call_id, event_data, account_id=account_id
        )
