"""Single-step session harness.

Phase 5 replaces the synchronous multi-turn loop with an event-driven
step function. Each procrastinate ``wake_session`` job calls
:func:`run_session_step`, which:

1. Checks whether the model needs to be called (:func:`should_call_model`).
2. Builds the chat-completions message list with pending-result synthesis.
3. Calls LiteLLM exactly once.
4. Appends the assistant message to the session log.
5. Kicks off tool calls as fire-and-forget asyncio tasks (if any).
6. Returns — the procrastinate lock is released immediately.

Tool completion triggers a new ``wake_session`` job, which runs another
step. The "loop" is the job queue re-entering this function.

Mid-turn user injection is free: a new user message is just another
event in the log. The next step's :func:`should_call_model` sees it
and proceeds.
"""

from __future__ import annotations

from typing import Any

from aios.db.sse_lock import has_subscriber
from aios.harness import runtime
from aios.harness.completion import call_litellm, stream_litellm
from aios.harness.step_context import compose_step_context
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.tool_dispatch import launch_mcp_tool_calls, launch_tool_calls
from aios.harness.wake import defer_retry_wake
from aios.logging import get_logger
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service

log = get_logger("aios.harness.loop")


_RETRY_BACKOFF_SECONDS: list[float] = [2, 8, 30, 120]


def _retry_delay_for_attempt(attempt: int) -> float | None:
    """Return the backoff delay for ``attempt``, or ``None`` if the budget is spent."""
    if attempt >= len(_RETRY_BACKOFF_SECONDS):
        return None
    return _RETRY_BACKOFF_SECONDS[attempt]


async def run_session_step(
    session_id: str,
    *,
    cause: str = "message",
    wake_reason: str | None = None,
) -> None:
    """Run one inference step for the session.

    Called by the procrastinate ``wake_session`` task. The procrastinate
    ``lock`` parameter guarantees only one step runs per session at a
    time.

    When ``cause == "scheduled"`` and ``wake_reason`` is set (the
    ``schedule_wake`` tool's delayed wake), a user-role marker is
    appended before the sweep guard so the model has something to
    react to on this step.
    """
    pool = runtime.require_pool()
    task_registry = runtime.require_task_registry()

    if cause == "scheduled" and wake_reason:
        await sessions_service.append_event(
            pool,
            session_id,
            "message",
            {
                "role": "user",
                "content": f"[Your scheduled wake fired. Reason: {wake_reason}]",
            },
        )

    # Sweep-based guard: does this session actually need work?
    # Prevents wasted DB/model calls from stale or duplicate wakes.
    needs = await find_sessions_needing_inference(pool, task_registry, session_id=session_id)
    if session_id not in needs:
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return

    session = await sessions_service.get_session(pool, session_id)

    # Load agent config — pinned version or latest.
    from aios.models.agents import Agent, AgentVersion

    agent: Agent | AgentVersion
    if session.agent_version is not None:
        agent = await agents_service.get_agent_version(
            pool, session.agent_id, session.agent_version
        )
    else:
        agent = await agents_service.get_agent(pool, session.agent_id)

    from aios.harness.channels import connection_server_name, list_bindings_and_connections

    bindings, connections = await list_bindings_and_connections(pool, session_id)

    mcp_server_map: dict[str, str] = {s.name: s.url for s in agent.mcp_servers}
    for c in connections:
        mcp_server_map[connection_server_name(c)] = c.mcp_url

    # Read windowed message events for this session.
    events = await sessions_service.read_windowed_events(
        pool, session_id, window_min=agent.window_min, window_max=agent.window_max
    )

    # Check for confirmed-but-undispatched tool calls (always_ask → allow).
    # The sweep's case (c) ensures we passed the guard above.
    pending = await _dispatch_confirmed_tools(pool, session_id, events)
    if pending:
        pending_builtin = [tc for tc in pending if not _is_mcp_tool(_tc_name(tc))]
        pending_mcp = [tc for tc in pending if _is_mcp_tool(_tc_name(tc))]
        if pending_builtin:
            launch_tool_calls(pool, session_id, pending_builtin)
        if pending_mcp:
            launch_mcp_tool_calls(
                pool,
                session_id,
                pending_mcp,
                mcp_server_map,
                focal_channel=session.focal_channel,
            )
        log.info(
            "step.confirmed_tools_dispatched",
            session_id=session_id,
            count=len(pending),
        )
        return

    # Span the remainder of the prologue so "why is the step slow?"
    # can separate context-build cost from model-call cost (issue #78).
    # Bracketing starts AFTER the dispatch early-return so every start
    # has a matching end; on failure we still emit the end with
    # ``is_error: True`` and re-raise, matching the ``model_request_*``
    # symmetry.
    context_build_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "context_build_start"},
    )

    try:
        step_ctx = await compose_step_context(
            pool,
            session_id,
            session=session,
            agent=agent,
            bindings=bindings,
            connections=connections,
            events=events,
        )
    except Exception:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "context_build_end",
                "context_build_start_id": context_build_start.id,
                "is_error": True,
            },
        )
        raise

    messages = step_ctx.messages
    tools = step_ctx.tools

    # Provision skill files to workspace (idempotent, host-side writes).
    if step_ctx.skill_versions:
        from aios.harness.skills import provision_skill_files

        await provision_skill_files(session_id, step_ctx.skill_versions)

    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "context_build_end",
            "context_build_start_id": context_build_start.id,
            "is_error": False,
            "event_count_read": len(events),
            "message_count": len(messages),
            "tools_count": len(tools),
        },
    )

    # Dump the exact chat-completions payload we're about to send to LiteLLM
    # when AIOS_DUMP_CONTEXT is set — useful for debugging prompt construction
    # (header inlining, system-prompt augmentation, tool list shape).
    await _dump_context_if_enabled(session_id, agent.model, messages, tools)

    # Mark session as running.
    await sessions_service.set_session_status(pool, session_id, "running")

    # Emit span start so consumers can measure inference latency.
    start_event = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "model_request_start"},
    )

    # Call the model exactly once.  Stream deltas via pg_notify only when
    # an SSE subscriber is attached (issue #81); otherwise run the faster
    # non-streaming path.  OpenRouter-style proxies can be 2-3x slower on
    # the streaming path when nobody is consuming the deltas.
    subscribed = await has_subscriber(pool, session_id)
    try:
        if subscribed:
            assistant_msg, usage, cost_usd = await stream_litellm(
                model=agent.model,
                messages=messages,
                tools=tools if tools else None,
                extra=agent.litellm_extra or None,
                pool=pool,
                session_id=session_id,
            )
        else:
            assistant_msg, usage, cost_usd = await call_litellm(
                model=agent.model,
                messages=messages,
                tools=tools if tools else None,
                extra=agent.litellm_extra or None,
            )
    except Exception:
        log.exception("step.litellm_failed", session_id=session_id)
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "model_request_end",
                "model_request_start_id": start_event.id,
                "is_error": True,
                "model_usage": {},
                "cost_usd": None,
            },
        )

        attempt = await _count_consecutive_rescheduling(pool, session_id)
        delay = _retry_delay_for_attempt(attempt)
        if delay is not None:
            await sessions_service.set_session_status(
                pool, session_id, "rescheduling", stop_reason={"type": "rescheduling"}
            )
            await _append_lifecycle(pool, session_id, "turn_ended", "rescheduling", "rescheduling")
            await defer_retry_wake(session_id, delay_seconds=delay)
            return

        await sessions_service.set_session_status(
            pool, session_id, "idle", stop_reason={"type": "error"}
        )
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "error")
        raise

    # Emit span end with per-request token usage and LiteLLM-reported cost.
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "model_request_end",
            "model_request_start_id": start_event.id,
            "is_error": False,
            "model_usage": usage,
            "cost_usd": cost_usd,
        },
    )

    # Increment cumulative session-level token counters.
    await sessions_service.increment_usage(
        pool,
        session_id,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
    )

    if bindings:
        from aios.harness.channels import apply_monologue_prefix

        assistant_msg = apply_monologue_prefix(assistant_msg)

    # Inject reacting_to so should_call_model knows what this response
    # was based on. This is the seq of the latest user/tool event in the
    # context — events after this seq are "new" on the next wake.
    assistant_msg["reacting_to"] = step_ctx.reacting_to

    # Append assistant message to the session log (unfenced — procrastinate
    # lock provides mutual exclusion).
    await sessions_service.append_event(pool, session_id, "message", assistant_msg)

    # Check for tool calls — partition into four buckets:
    #   immediate       — built-in with always_allow (execute now)
    #   mcp_immediate   — MCP with always_allow (execute now via MCP client)
    #   needs_confirm   — built-in or MCP with always_ask (hold for confirmation)
    #   custom          — not in registry and not MCP (hold for client execution)
    tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []

    if tool_calls:
        from aios.tools.registry import registry as tool_registry

        immediate: list[dict[str, Any]] = []
        mcp_immediate: list[dict[str, Any]] = []
        needs_confirm: list[dict[str, Any]] = []
        custom: list[dict[str, Any]] = []

        for tc in tool_calls:
            name = _tc_name(tc)
            if _is_mcp_tool(name):
                # MCP tools default to always_ask (unlike built-in always_allow).
                perm = resolve_mcp_permission(name, agent.tools)
                if perm == "always_allow":
                    mcp_immediate.append(tc)
                else:
                    needs_confirm.append(tc)
            elif not tool_registry.has(name):
                custom.append(tc)
            elif resolve_permission(name, agent.tools) == "always_ask":
                needs_confirm.append(tc)
            else:
                immediate.append(tc)

        if immediate:
            launch_tool_calls(pool, session_id, immediate)
            log.info(
                "step.tools_launched",
                session_id=session_id,
                count=len(immediate),
                tool_names=[_tc_name(tc) for tc in immediate],
            )

        if mcp_immediate:
            launch_mcp_tool_calls(
                pool,
                session_id,
                mcp_immediate,
                mcp_server_map,
                focal_channel=session.focal_channel,
            )
            log.info(
                "step.mcp_tools_launched",
                session_id=session_id,
                count=len(mcp_immediate),
                tool_names=[_tc_name(tc) for tc in mcp_immediate],
            )

        if needs_confirm or custom:
            confirm_ids = [tc.get("id") for tc in needs_confirm if tc.get("id")]
            custom_ids = [tc.get("id") for tc in custom if tc.get("id")]
            all_pending_ids = confirm_ids + custom_ids

            stop_reason: dict[str, Any] = {
                "type": "requires_action",
                "event_ids": all_pending_ids,
            }
            if confirm_ids:
                stop_reason["confirmations"] = confirm_ids
            if custom_ids:
                stop_reason["custom_tools"] = custom_ids

            await sessions_service.set_session_status(
                pool,
                session_id,
                "idle",
                stop_reason=stop_reason,
            )
            await _append_lifecycle(pool, session_id, "turn_ended", "idle", "requires_action")
            log.info(
                "step.tools_pending",
                session_id=session_id,
                confirmations=confirm_ids,
                custom_tools=custom_ids,
            )
    else:
        # No tool calls — the model's turn is done.
        await sessions_service.set_session_status(
            pool, session_id, "idle", stop_reason={"type": "end_turn"}
        )
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "end_turn")
        log.info("step.turn_ended", session_id=session_id, cause=cause)


def _switch_channel_tool_spec() -> dict[str, Any]:
    """Build the chat-completions tool entry for the ``switch_channel`` built-in.

    Injected unconditionally into the tool list when the session has
    any active bindings (see ``run_session_step``).  Agents don't need
    to list it in their ``tools`` declaration — it's focal-machinery
    scope, not agent scope.
    """
    from aios.tools.registry import registry as tool_registry

    tool = tool_registry.get("switch_channel")
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def _hide_conn_tools_when_phone_down(
    mcp_tools: list[dict[str, Any]], focal_channel: str | None
) -> list[dict[str, Any]]:
    """Filter connection-provided MCP tools out when focal is NULL.

    Those tools resolve their ``chat_id`` from focal (injected into
    ``_meta`` at dispatch time); a "phone down" state has no focal to
    inject, so the model shouldn't be offered them.  Agent-declared
    MCP tools are untouched.
    """
    from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX

    if focal_channel is not None:
        return mcp_tools
    prefix = f"mcp__{CONNECTION_SERVER_NAME_PREFIX}"
    return [t for t in mcp_tools if not t.get("function", {}).get("name", "").startswith(prefix)]


async def _dump_context_if_enabled(
    session_id: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Write the chat-completions payload to disk when ``AIOS_DUMP_CONTEXT`` is set.

    Debug aid: inspect exactly what reaches LiteLLM (post header-inlining,
    post system-prompt augmentation, with the full tool list).
    """
    import os as _os

    if not _os.environ.get("AIOS_DUMP_CONTEXT"):
        return
    import asyncio as _asyncio
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    dump_dir = _Path(_os.environ.get("AIOS_DUMP_CONTEXT_DIR", "/tmp/aios-context-dumps"))
    ts = int(_time.time() * 1000)
    path = dump_dir / f"{ts}_{session_id}.json"
    payload = {
        "session_id": session_id,
        "model": model,
        "messages": messages,
        "tools": tools,
    }

    def _write() -> None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            _json.dump(payload, f, indent=2)

    await _asyncio.to_thread(_write)
    log.info("step.context_dumped", path=str(path))


def _tc_name(tc: dict[str, Any]) -> str:
    """Extract the function name from a tool_call dict."""
    name: str = (tc.get("function") or {}).get("name", "")
    return name


def _is_mcp_tool(name: str) -> bool:
    """Return True if the tool name is an MCP-namespaced tool."""
    return name.startswith("mcp__")


def resolve_permission(name: str, agent_tools: list[ToolSpec]) -> str | None:
    """Look up the permission policy for a built-in or custom tool by name."""
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.permission
    return None


def resolve_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> str | None:
    """Look up the permission policy for an MCP tool.

    Connection-provided tools (server name in the reserved ``conn_``
    namespace) default to ``always_allow`` — the session's channel
    binding is already explicit routing consent; gating every reply on
    a confirmation prompt would defeat the connector autonomy story.

    For agent-declared servers, finds the ``mcp_toolset`` entry whose
    ``mcp_server_name`` matches the server portion of the namespaced
    tool name, then returns the ``default_config.permission_policy.type``
    or ``None`` (which callers treat as ``always_ask``).
    """
    from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX

    server_name = name.split("__", 2)[1]
    if server_name.startswith(CONNECTION_SERVER_NAME_PREFIX):
        return "always_allow"
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            cfg = spec.default_config
            if cfg and cfg.permission_policy:
                return cfg.permission_policy.type
            return spec.permission
    return None


async def discover_session_mcp_tools(
    pool: Any,
    session_id: str,
    agent: Any,
    connections: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Discover MCP tools from agent-declared servers (filtered by enabled
    ``mcp_toolset`` entries) unioned with connection-provided servers.

    Returns ``(tools, instructions_by_server)`` where the second element
    maps ``server_name`` → the server's ``InitializeResult.instructions``
    string.  Servers that supplied no instructions (or ``""``) are
    omitted from the dict — the harness uses presence in the dict as
    the trigger for rendering a per-connector affordance block.
    """
    import asyncio

    from aios.harness.channels import connection_server_name
    from aios.mcp.client import discover_mcp_tools, resolve_auth_for_url

    servers: list[tuple[str, str]] = []

    enabled_server_names: set[str] = set()
    for spec in agent.tools:
        if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name:
            enabled_server_names.add(spec.mcp_server_name)
    for s in agent.mcp_servers:
        if s.name in enabled_server_names:
            servers.append((s.name, s.url))

    for c in connections:
        servers.append((connection_server_name(c), c.mcp_url))

    if not servers:
        return [], {}

    crypto_box = runtime.require_crypto_box()

    async def _discover_one(name: str, url: str) -> tuple[list[dict[str, Any]], str | None]:
        headers = await resolve_auth_for_url(pool, crypto_box, session_id, url)
        return await discover_mcp_tools(url, name, headers)

    results = await asyncio.gather(*[_discover_one(n, u) for n, u in servers])
    tools: list[dict[str, Any]] = [
        tool for (tool_list, _instructions) in results for tool in tool_list
    ]
    instructions_by_server: dict[str, str] = {
        name: instructions
        for (name, _url), (_tools, instructions) in zip(servers, results, strict=True)
        if instructions
    }
    return tools, instructions_by_server


async def _dispatch_confirmed_tools(
    pool: Any,
    session_id: str,
    message_events: list[Any],
) -> list[dict[str, Any]]:
    """Find tool calls that have been confirmed (allow) but not yet dispatched.

    Returns the original tool call dicts ready for ``launch_tool_calls``,
    or an empty list if nothing to dispatch.
    """
    # Find the latest assistant message with tool_calls.
    asst_tool_calls: list[dict[str, Any]] = []
    for e in reversed(message_events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            tcs = e.data.get("tool_calls")
            if tcs:
                asst_tool_calls = tcs
                break

    if not asst_tool_calls:
        return []

    # Build set of tool_call_ids that already have a tool-role result.
    completed: set[str] = set()
    for e in message_events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                completed.add(tcid)

    # Read lifecycle events to find tool_confirmed allow events.
    lifecycle_events = await sessions_service.read_events(
        pool,
        session_id,
        kind="lifecycle",
    )
    confirmed: set[str] = set()
    for e in lifecycle_events:
        if e.data.get("event") == "tool_confirmed" and e.data.get("result") == "allow":
            tcid = e.data.get("tool_call_id")
            if tcid:
                confirmed.add(tcid)

    # Return tool calls that are confirmed but have no result yet.
    pending = [
        tc for tc in asst_tool_calls if tc.get("id") in confirmed and tc.get("id") not in completed
    ]
    return pending


async def _count_consecutive_rescheduling(pool: Any, session_id: str) -> int:
    """Count consecutive rescheduling lifecycle events at the tail of the log.

    Returns the number of consecutive ``turn_ended`` lifecycle events
    with ``stop_reason == "rescheduling"`` at the end of the lifecycle
    event sequence. A non-rescheduling event breaks the streak.
    """
    lifecycle_events = await sessions_service.read_events(pool, session_id, kind="lifecycle")
    count = 0
    for e in reversed(lifecycle_events):
        if e.data.get("event") == "turn_ended" and e.data.get("stop_reason") == "rescheduling":
            count += 1
        else:
            break
    return count


async def _append_lifecycle(
    pool: Any,
    session_id: str,
    event: str,
    status: str,
    stop_reason: str,
) -> None:
    """Append a lifecycle event."""
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": event, "status": status, "stop_reason": stop_reason},
    )
