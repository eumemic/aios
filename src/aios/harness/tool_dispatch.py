"""Dispatch assistant tool_calls to tool handlers.

This is the bridge between a LiteLLM response (which has ``tool_calls``
inside the assistant message) and the tool-role message events that
have to appear in the session log before the next LLM call.

The dispatcher embodies the "cattle not pets" container principle from
the Managed Agents blog post: if a tool handler raises an exception —
container died, docker daemon unreachable, exec timeout, anything — the
dispatcher turns the exception into a tool-role message event with an
error payload the model can see and react to. It does NOT propagate the
exception up to the harness loop. The turn continues.

A handler returning a structured dict with normal output (possibly with
a nonzero exit code) is "normal" and gets wrapped as a successful tool
result. Only raised exceptions are treated as errors.

When a handler raises, the dispatcher also evicts the session's cached
:class:`ContainerHandle` from the sandbox registry, on the theory that
the container is probably dead or unusable. The next tool call will
then provision a fresh container lazily.

Concurrency: tool calls within a single assistant message are executed
sequentially, not in parallel. Each handler's result is appended as its
own event before the next handler runs. This is the conservative v1
choice — parallel dispatch has ordering implications for tool-role
messages in the session log. Phase 5+ may revisit.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from aios.harness.lease import append_event_with_fence
from aios.logging import get_logger
from aios.tools.registry import ToolNotFoundError, registry

log = get_logger("aios.harness.tool_dispatch")


async def dispatch_tool_calls(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    worker_id: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Run every tool call in ``tool_calls`` and append their results.

    For each call:

    1. Look up the handler in the global tool registry.
    2. Parse ``arguments`` from its JSON string form.
    3. Invoke the handler.
    4. Append a tool-role message event with the handler's return value
       (wrapped as JSON string per OpenAI's chat-completions shape) via
       :func:`append_event_with_fence`.

    On handler failure:

    * Log the exception with structured context.
    * Evict the session's sandbox container from the registry (the
      container is likely dead or unusable).
    * Append a tool-role message with ``is_error: true`` and a short
      error description, still via the fenced append.
    * Continue to the next tool call — this one's failure does not
      cancel subsequent calls in the same message.

    The fenced append still applies on the error path: if the worker
    has lost the lease, :class:`~aios.harness.lease.LeaseLost` propagates
    up and the harness loop handles it (no release, let the new lease
    holder take over).
    """
    for call in tool_calls:
        await _dispatch_one(
            pool,
            session_id=session_id,
            worker_id=worker_id,
            call=call,
        )


async def _dispatch_one(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    worker_id: str,
    call: dict[str, Any],
) -> None:
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments", "{}")

    bound_log = log.bind(
        session_id=session_id,
        worker_id=worker_id,
        tool_call_id=call_id,
        tool_name=name,
    )

    # Parse arguments. They're a JSON string per OpenAI shape; tolerate
    # dicts too (some providers pass them already-parsed through litellm).
    arguments: dict[str, Any]
    if isinstance(raw_args, dict):
        arguments = raw_args
    else:
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as err:
            bound_log.warning("tool.bad_arguments_json", error=str(err))
            await _append_tool_error(
                pool,
                session_id=session_id,
                worker_id=worker_id,
                call_id=call_id,
                name=name,
                error_message=f"arguments were not valid JSON: {err}",
            )
            return
        if not isinstance(parsed, dict):
            bound_log.warning("tool.arguments_not_object", arguments_type=type(parsed).__name__)
            await _append_tool_error(
                pool,
                session_id=session_id,
                worker_id=worker_id,
                call_id=call_id,
                name=name,
                error_message="arguments must be a JSON object",
            )
            return
        arguments = parsed

    # Look up the tool.
    try:
        tool = registry.get(name)
    except ToolNotFoundError as err:
        bound_log.warning("tool.not_registered")
        await _append_tool_error(
            pool,
            session_id=session_id,
            worker_id=worker_id,
            call_id=call_id,
            name=name,
            error_message=err.message,
        )
        return

    # Run the handler. Any exception → tool error event; evict the
    # container from the sandbox registry since it may be dead.
    try:
        result = await tool.handler(session_id, arguments)
    except asyncio.CancelledError:
        # Cancellation is not an error — propagate so the harness loop
        # can handle interrupt / lease loss cleanly. Phase 5 territory.
        raise
    except Exception as err:
        bound_log.exception("tool.handler_failed")
        _evict_session_container(session_id)
        await _append_tool_error(
            pool,
            session_id=session_id,
            worker_id=worker_id,
            call_id=call_id,
            name=name,
            error_message=f"{type(err).__name__}: {err}",
        )
        return

    # Success path. Append a tool-role message with the handler's
    # JSON-serialized result as the content.
    content_str = json.dumps(result, ensure_ascii=False)
    bound_log.info("tool.dispatched")
    await append_event_with_fence(
        pool,
        session_id=session_id,
        expected_worker_id=worker_id,
        kind="message",
        data={
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content_str,
        },
    )


async def _append_tool_error(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    worker_id: str,
    call_id: str,
    name: str,
    error_message: str,
) -> None:
    """Append a tool-role message that carries an error for the model.

    The shape follows OpenAI's tool message convention: ``role: tool``,
    ``tool_call_id`` linking back to the call, ``content`` as a JSON
    string with an ``error`` field the model can read. We also set a
    top-level ``is_error: true`` flag for easier downstream filtering
    (not part of OpenAI's formal schema but tolerated by LiteLLM).
    """
    content = json.dumps({"error": error_message}, ensure_ascii=False)
    await append_event_with_fence(
        pool,
        session_id=session_id,
        expected_worker_id=worker_id,
        kind="message",
        data={
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
            "is_error": True,
        },
    )


def _evict_session_container(session_id: str) -> None:
    """Best-effort eviction of the session's cached sandbox container.

    If the sandbox registry isn't initialised (e.g. in a unit test that
    exercises tool_dispatch without a full worker), silently skip.
    """
    from aios.harness import runtime

    if runtime.sandbox_registry is None:
        return
    runtime.sandbox_registry.evict(session_id)
