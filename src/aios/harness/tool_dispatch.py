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
from types import EllipsisType
from typing import TYPE_CHECKING, Any

import asyncpg

from aios.errors import AiosError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.agents import McpServerSpec
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail, invoke_builtin, parse_arguments
from aios.tools.registry import ToolResult

if TYPE_CHECKING:
    from aios.services.invocations import ServicerKind

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


def _classify_tool_error(err: BaseException) -> tuple[bool, str]:
    """Decide how a tool handler exception is reported: ``(should_evict, message)``.

    Expected, model-visible refusals do NOT evict the sandbox — the model reads the
    error and self-corrects:
      * a :class:`ToolBail` (bad args, unknown tool, schema mismatch), and
      * a client-class (``status_code < 500``) :class:`AiosError` — a permission denial,
        not-found, conflict, rate-limit, or any tool ``*ArgumentError``.
    A server-class (``>= 500``) ``AiosError`` or any other exception is a genuine failure
    that also evicts the sandbox, which may have been left in a bad state. (The eviction
    itself is still gated on the caller's ``on_exception`` — the MCP path passes none.)
    """
    if isinstance(err, ToolBail):
        return False, str(err)
    if isinstance(err, AiosError):
        # ``to_message`` is total even here in the except clause (see AiosError.to_message).
        return err.status_code >= 500, err.to_message()
    return True, f"{type(err).__name__}: {err}"


@asynccontextmanager
async def _tool_lifecycle(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    *,
    account_id: str,
    log_prefix: str,
    on_exception: Callable[[str], None] | None = None,
    write_start_span: bool = True,
) -> AsyncIterator[_ToolCall]:
    """Bracket a tool call with the ``tool_execute_*`` span pair plus
    try/except/finally + tail sweep (#78).  ``on_exception`` runs on
    generic ``Exception`` only — built-in passes the sandbox-eviction
    hook; MCP leaves it ``None`` because it doesn't use the sandbox.

    ``write_start_span=False`` skips the ``tool_execute_*`` span pair: the
    crash-resume re-park (:func:`_resume_parked_async`) is a pure read of durable
    state, and a fresh ``tool_execute_start`` span would mislead a later ghost
    classification into the side-effect-conservative "may have completed" branch
    (the span is the only side-effect evidence ``sweep.find_and_repair_ghosts``
    has). The dedup-guarded result append + tail sweep are still reused.
    """
    call_id = call.get("id") or "unknown"
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments", "{}")
    # Load-bearing for ghost recovery (#685): the commit of this span is what
    # ``sweep.find_and_repair_ghosts`` reads to distinguish "tool never
    # dispatched" from "tool may have executed (outcome unknown)". Any
    # side-effectful work added ABOVE this append could commit and then be
    # lost to a crash before the span lands, surfacing as "never started"
    # and double-firing on the model's retry. Keep the span as the very
    # first action in this lifecycle — only pure-Python preamble (arg
    # extraction, no I/O) is permitted above.
    #
    # The span is intentionally OUTSIDE the try/except below: an
    # ``asyncio.CancelledError`` arriving mid-await leaves the span possibly
    # committed and the body never entered, surfacing as "may have completed"
    # — an over-pessimistic outcome that the recovery message acknowledges
    # (see the branch comment in ``sweep.find_and_repair_ghosts``).  This is
    # the conservatively-safe direction; the alternative (suppress the span
    # on cancellation) would risk under-pessimism on real side effects.
    span_start = (
        await sessions_service.append_event(
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
        if write_start_span
        else None
    )
    bound_log = log.bind(session_id=session_id, tool_call_id=call_id, tool_name=name)
    tc = _ToolCall(call_id=call_id, name=name, raw_args=raw_args, bound_log=bound_log)
    try:
        yield tc
    except asyncio.CancelledError:
        bound_log.info(f"{log_prefix}.cancelled")
        tc.is_error = True
        await _append_tool_result(
            pool, session_id, call_id, name, account_id=account_id, error="cancelled"
        )
    except Exception as err:
        tc.is_error = True
        evict, message = _classify_tool_error(err)
        if evict:
            bound_log.exception(f"{log_prefix}.handler_failed")
            if on_exception is not None:
                on_exception(session_id)
        else:
            bound_log.info(f"{log_prefix}.refused", error=message)
        await _append_tool_result(
            pool, session_id, call_id, name, account_id=account_id, error=message
        )
    finally:
        if span_start is not None:
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
    parent_focal_at_arrival: str | None | EllipsisType = ...,
) -> None:
    """Launch each tool call as an asyncio task. Returns immediately.

    ``parent_focal_at_arrival`` is the requesting assistant event's locked
    focal stamp; threaded into the success-path result append so
    ``append_event`` skips the in-lock tool-parent lookup (issue #862). The
    live harness dispatch passes it; cold callers (confirmed re-dispatch)
    leave the default ``...`` so the append falls back to the pre-tx lookup.
    """
    _launch_tasks(
        session_id,
        tool_calls,
        lambda call: _execute_tool_async(
            pool,
            session_id,
            call,
            account_id=account_id,
            parent_focal_at_arrival=parent_focal_at_arrival,
        ),
        prefix="tool",
    )


async def _execute_tool_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    *,
    account_id: str,
    parent_focal_at_arrival: str | None | EllipsisType = ...,
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
        result = await invoke_builtin(session_id, tc.name, tc.raw_args, tool_call_id=tc.call_id)
        event_data = _shape_tool_result(tc, result)
        tc.bound_log.info("tool.completed")
        await _append_tool_result_event(
            pool,
            session_id,
            tc.call_id,
            event_data,
            account_id=account_id,
            tool_parent_channel=parent_focal_at_arrival,
        )


def _shape_tool_result(tc: _ToolCall, result: ToolResult | dict[str, Any]) -> dict[str, Any]:
    """Shape a handler result into a ``role:"tool"`` event payload, marking ``tc.is_error``.

    Shared by the live dispatch (:func:`_execute_tool_async`) and the crash-resume re-park
    (:func:`_resume_parked_async`) so both land an identically-shaped tool-role event.
    """
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
    return event_data


def relaunch_parked_invocation(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    call: dict[str, Any],
    servicer_kind: ServicerKind,
    servicer_id: str,
    request_id: str | None,
    output_schema: dict[str, Any] | None,
    account_id: str,
) -> None:
    """Re-park a ``call_*`` invocation whose in-memory park task was lost to a worker
    crash (#1431). Returns immediately; the resume runs as a fire-and-forget tool task.

    Routed through :func:`_launch_tasks` so the resume registers in the ``TaskRegistry``
    synchronously — a concurrent sweep then sees it in-flight (``CANDIDATE`` filter) and
    won't double-launch. The handle ``(servicer_kind, servicer_id, request_id)`` is
    re-derived from the durable servicer edge (``queries.find_parked_servicer``); ``call``
    is the original assistant tool_call dict (carries the ``tool_call_id`` + name the
    tool-role result needs).
    """
    _launch_tasks(
        session_id,
        [call],
        lambda c: _resume_parked_async(
            pool,
            session_id,
            c,
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
            request_id=request_id,
            output_schema=output_schema,
            account_id=account_id,
        ),
        prefix="repark",
    )


async def _resume_parked_async(
    pool: asyncpg.Pool[Any],
    session_id: str,
    call: dict[str, Any],
    *,
    servicer_kind: ServicerKind,
    servicer_id: str,
    request_id: str | None,
    output_schema: dict[str, Any] | None,
    account_id: str,
) -> None:
    """Re-park on the servicer and append the resolved tool-role result.

    Reuses :func:`_tool_lifecycle` (``write_start_span=False``) for the dedup-guarded
    result append + tail sweep — the dedup makes a re-park that races a real result safe
    (first commit wins). :func:`aios.tools.invoke_session._park_and_resolve` is a pure read
    of durable state, so this is side-effect-free apart from that single result append.
    """
    async with _tool_lifecycle(
        pool,
        session_id,
        call,
        account_id=account_id,
        log_prefix="repark",
        write_start_span=False,
    ) as tc:
        # Imported inside the bracket (``invoke_session`` ↔ ``tool_dispatch`` import order)
        # so even an import failure lands a tool-role error result rather than escaping the
        # task as an unobserved exception with no result.
        from aios.tools.invoke_session import _park_and_resolve

        result = await _park_and_resolve(
            pool,
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
            request_id=request_id,
            account_id=account_id,
            output_schema=output_schema,
        )
        event_data = _shape_tool_result(tc, result)
        tc.bound_log.info("repark.completed")
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
    tool_parent_channel: str | None | EllipsisType = ...,
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

    content = event_data.get("content")
    if isinstance(content, str):
        from aios.config import get_settings
        from aios.sandbox.tool_result_spill import (
            cap_tool_result_content,
            record_spill_attachment,
        )

        capped = await cap_tool_result_content(
            session_id, tool_call_id, content, max_chars=get_settings().tool_result_max_chars
        )
        event_data["content"] = capped.content
        # Register any spill file under ``metadata.attachments`` so the
        # attachment GC's referenced-set sees it as live (#1093).  No-op
        # when nothing spilled.
        record_spill_attachment(event_data, capped.attachment)

    # ── Pre-lock precompute (issue #991) ──────────────────────────────────
    # The tokenizer pass (and, on the cold path, the parent-channel JSONB ``@>``
    # scan) must NOT run under the outer ``FOR UPDATE`` dedup lock — that was
    # the ~100 KB tool-result path #862 set out to free but couldn't reach
    # while this caller held the lock across ``append_event``.  Resolve it here,
    # OUTSIDE and BEFORE the lock.  The dedup ``find_tool_result_event`` and the
    # INSERT stay inside the lock (idempotency guard unchanged).
    #
    # The hot builtin/MCP path supplies a concrete ``tool_parent_channel`` and a
    # tool-event token delta is pure ``approx_tokens([data])`` — so it needs NO
    # DB and the precompute runs on a throwaway value.  Only the cold path
    # (``...`` sentinel: operator deny, sweep, ghost-repair re-dispatch) needs a
    # conn for ``_lookup_tool_parent_channel``; it does one brief
    # ``pool.acquire()`` released BEFORE the lock acquire (sequential, no
    # double-hold; the pre-read is race-free by commit-ordering — the parent
    # assistant row commits before any tool result can arrive).
    if tool_parent_channel is ...:
        async with pool.acquire() as precompute_conn:
            precomputed = await queries.precompute_event_append(
                precompute_conn,
                account_id=account_id,
                session_id=session_id,
                kind="message",
                data=event_data,
                tool_parent_channel=tool_parent_channel,
            )
    else:
        # Hot path: concrete channel, pure ``approx_tokens`` delta — no DB.
        precomputed = queries._PrecomputedAppend(
            token_delta=queries._event_token_delta("message", event_data, None, None),
            resolved_tool_channel=tool_parent_channel,
        )

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
            await _dedup_skip(conn, session_id, tool_call_id, existing, event_data, account_id)
            return
        try:
            await queries.append_event(
                conn,
                session_id=session_id,
                kind="message",
                data=event_data,
                account_id=account_id,
                precomputed=precomputed,
            )
        except asyncpg.UniqueViolationError:
            # Structural floor (#1082): the partial UNIQUE index
            # ``events_tool_result_idx`` forbids a second tool-role row for
            # ``(session_id, tool_call_id)``. A racing appender (the operator
            # deny path, or another worker task) committed its row between our
            # read-check above and this INSERT, so ``find_tool_result_event``
            # saw nothing but the index rejects the write. The append runs in
            # ``append_event``'s own ``conn.transaction()``, so the
            # UniqueViolation rolled back the seq increment with it — gapless
            # seq preserved, no duplicate row. Re-read and apply the same
            # id-blind compensation as the read-check path: the winning row is
            # the one already committed.
            existing = await queries.find_tool_result_event(
                conn, session_id, tool_call_id, account_id=account_id
            )
            assert existing is not None  # the UniqueViolation proves a row exists
            await _dedup_skip(conn, session_id, tool_call_id, existing, event_data, account_id)


async def _dedup_skip(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    existing: Any,
    event_data: dict[str, Any],
    account_id: str,
) -> None:
    """Log the dedup-skip and apply the ``open_tool_call_count`` compensation.

    Shared by the read-check fast path and the ``UniqueViolation`` catch in
    :func:`_append_tool_result_event` so both leave identical state: the
    winning row stays, the late worker append no-ops, and the id-blind ``+1``
    that was applied at assistant-turn time (issue #890) is decremented once.
    """
    from aios.db import queries

    existing_is_error = bool(existing.data.get("is_error", False))
    log.warning(
        "tool.result_dedup_skip",
        session_id=session_id,
        tool_call_id=tool_call_id,
        existing_is_error=existing_is_error,
        attempted_is_error=bool(event_data.get("is_error", False)),
    )
    await queries.decrement_open_tool_call_count(conn, session_id, account_id=account_id)


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
    mcp_server_map: dict[str, McpServerSpec],
    *,
    account_id: str,
    focal_channel: str | None = None,
    parent_focal_at_arrival: str | None | EllipsisType = ...,
) -> None:
    """Launch MCP tool calls as asyncio tasks. Returns immediately.

    ``focal_channel`` is the session's focal at the moment these calls
    were emitted (captured at step top in ``run_session_step`` so a
    concurrent ``switch_channel`` in the same batch cannot race the
    ``chat_id`` injection).  Passed through to each task so
    connection-provided tools can stamp ``_meta`` with the suffix.

    ``parent_focal_at_arrival`` is DISTINCT: the requesting assistant
    event's locked focal stamp, threaded into the result append so
    ``append_event`` skips the in-lock tool-parent lookup (issue #862).
    Cold callers (confirmed re-dispatch) leave the default ``...`` →
    pre-tx lookup.
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
            parent_focal_at_arrival=parent_focal_at_arrival,
        ),
        prefix="mcp_tool",
    )


async def _mcp_call_suppressed(
    pool: asyncpg.Pool[Any], session_id: str, account_id: str, qualified_name: str
) -> bool:
    """Whether this MCP call is intercepted by outbound suppression (#710).

    Returns ``False`` when the session is not in suppression mode (the common
    case). Otherwise resolves the per-tool ``read_allow`` opt-in against the
    session's effective agent tools (default-deny). Loaded here so the model
    dispatch path and the sandbox CLI broker make the identical decision.
    """
    from aios.models.agents import mcp_tool_suppressed
    from aios.services import agents as agents_service

    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    if session.outbound_suppression != "on":
        return False
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    return mcp_tool_suppressed(qualified_name, agent.tools)


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
    mcp_server_map: dict[str, McpServerSpec],
    *,
    account_id: str,
    focal_channel: str | None = None,
    parent_focal_at_arrival: str | None | EllipsisType = ...,
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

        spec = mcp_server_map.get(server_name)
        if spec is None:
            raise ToolBail(f"MCP server {server_name!r} not found")
        url = spec.url

        # Outbound suppression (#710): MCP is default-deny. When the session is
        # in suppression mode, every MCP call is intercepted (synthesized
        # success + audit event) unless the per-tool ``read_allow`` opt-in
        # marks it a known-safe read. Same decision the sandbox CLI broker
        # makes — both consult ``mcp_tool_suppressed``.
        from aios.services import outbound_suppression as suppression_service

        if await _mcp_call_suppressed(pool, session_id, account_id, tc.name):
            await suppression_service.record_mcp_suppression(
                pool,
                session_id,
                account_id=account_id,
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
            )
            result = suppression_service.mcp_synthesized_result()
        else:
            from aios.mcp.client import call_mcp_tool, resolve_auth_for_target_url

            crypto_box = runtime.require_crypto_box()
            vault_id, headers = await resolve_auth_for_target_url(
                pool, crypto_box, session_id, url, account_id=account_id
            )
            result = await call_mcp_tool(
                url, vault_id, headers, tool_name, arguments, meta=meta, spec_headers=spec.headers
            )

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
            pool,
            session_id,
            tc.call_id,
            event_data,
            account_id=account_id,
            tool_parent_channel=parent_focal_at_arrival,
        )
