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

import asyncio
from typing import TYPE_CHECKING, Any

from aios.db.sse_lock import has_subscriber
from aios.harness import runtime
from aios.harness.completion import call_litellm, stream_litellm
from aios.harness.step_context import compose_step_context, compute_step_prelude
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.tokens import approx_tokens
from aios.harness.tool_dispatch import launch_mcp_tool_calls, launch_tool_calls
from aios.harness.wake import defer_retry_wake
from aios.logging import get_logger
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    import asyncpg

    from aios.harness.task_registry import TaskRegistry
    from aios.models.memory_stores import MemoryStoreResourceEcho

log = get_logger("aios.harness.loop")


_RETRY_BACKOFF_SECONDS: list[float] = [2, 8, 30, 120]

# Wall-clock cap on a single ``run_session_step`` invocation. The harness's
# zero-hang guarantee: per-call timeouts (LiteLLM, MCP, tool dispatch, etc.)
# are the precise instruments, but if any future code path bypasses them
# this cap fires and forces a clean rescheduling. Sized to fit the longest
# legitimate single-turn use (300s = matches the ``_REQUEST_TIMEOUT_S`` in
# ``completion.py`` so the model call alone can occupy almost the whole
# budget).
_JOB_TIMEOUT_S = 300.0


def _retry_delay_for_attempt(attempt: int) -> float | None:
    """Return the backoff delay for ``attempt``, or ``None`` if the budget is spent."""
    if attempt >= len(_RETRY_BACKOFF_SECONDS):
        return None
    return _RETRY_BACKOFF_SECONDS[attempt]


async def refresh_session_mount_state(
    pool: asyncpg.Pool[Any], session_id: str
) -> list[MemoryStoreResourceEcho]:
    """Return the freshly-loaded echoes so the step body can skip a second DB query."""
    from aios.db import queries

    async with pool.acquire() as conn:
        echoes = await queries.list_session_memory_store_echoes(conn, session_id)
    runtime.set_session_memory_mounts(session_id, echoes)
    if runtime.sandbox_registry is not None:
        await runtime.sandbox_registry.release_if_mounts_changed(session_id, echoes)
    return echoes


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

    # Outermost span pair: brackets the entire step (issue #131).  Emitted
    # before the sweep guard so early-outs are also measured — a "wasted
    # wake" cost shows up as a ``step_start``/``step_end`` pair with no
    # ``context_build_*`` inside.  ``step_start_id`` backpointer on the
    # end event matches the ``context_build_start_id`` convention.
    step_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "step_start", "cause": cause},
    )
    retry_delay: float | None = None
    try:
        try:
            retry_delay = await asyncio.wait_for(
                _run_session_step_body(
                    pool, task_registry, session_id, cause=cause, wake_reason=wake_reason
                ),
                timeout=_JOB_TIMEOUT_S,
            )
        except TimeoutError:
            # Job-level safety net: a per-call timeout was missing or didn't
            # fire. Force a reschedulable error state so the next wake can
            # proceed (matches what the body's litellm-error handler does).
            log.exception("step.job_timeout", session_id=session_id, timeout=_JOB_TIMEOUT_S)
            retry_delay = await _handle_step_timeout(pool, session_id)
    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {"event": "step_end", "step_start_id": step_start.id},
        )

    # Fire retry deferral AFTER ``step_end`` so its ``wake_deferred`` lands
    # in step N+1's temporal window, not step N's. Under the "all
    # wake_deferred since previous step_end" pairing rule, emitting
    # inside the body would make the reschedule invisible to the next
    # step's queue-latency calculation — the one path where delay is
    # a known quantity (the retry backoff).
    if retry_delay is not None:
        await defer_retry_wake(pool, session_id, delay_seconds=retry_delay)


async def _run_session_step_body(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    session_id: str,
    *,
    cause: str,
    wake_reason: str | None,
) -> float | None:
    """Returns the retry backoff delay when the model errored and the
    outer function should ``defer_retry_wake`` after ``step_end``; ``None``
    otherwise.  Keeping the actual ``defer_retry_wake`` call outside the
    body is what makes the reschedule's ``wake_deferred`` land in the
    next step's temporal window."""
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
    #
    # Bracket with a ``sweep_start``/``sweep_end`` span pair (site="entry").
    # Only ``find_sessions_needing_inference`` runs here — no ghost repair,
    # no ``defer_wake`` — so ``repaired_ghosts`` is always 0. ``woken_sessions``
    # at ``site="entry"`` is 0 or 1: it records whether the guard determined
    # this specific session had work. 0 indicates a wasted wake.
    sweep_start = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "sweep_start", "site": "entry"},
    )
    needs: set[str] = set()
    try:
        needs = await find_sessions_needing_inference(pool, task_registry, session_id=session_id)
    finally:
        await sessions_service.append_event(
            pool,
            session_id,
            "span",
            {
                "event": "sweep_end",
                "sweep_start_id": sweep_start.id,
                "repaired_ghosts": 0,
                "woken_sessions": 1 if session_id in needs else 0,
            },
        )
    if session_id not in needs:
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return None

    session = await sessions_service.get_session(pool, session_id)

    from aios.models.agents import Agent, AgentVersion
    from aios.services.channels import list_session_channels

    agent_task: asyncio.Task[Agent | AgentVersion]
    if session.agent_version is not None:
        agent_task = asyncio.create_task(
            agents_service.get_agent_version(pool, session.agent_id, session.agent_version)
        )
    else:
        agent_task = asyncio.create_task(agents_service.get_agent(pool, session.agent_id))
    agent, channels, memory_echoes = await asyncio.gather(
        agent_task,
        list_session_channels(pool, session_id),
        refresh_session_mount_state(pool, session_id),
    )

    mcp_server_map: dict[str, str] = {s.name: s.url for s in agent.mcp_servers}

    # Build the events-independent prelude (system prompt + tools)
    # before windowing so its overhead can be subtracted from the
    # window budget — otherwise the sent prompt can exceed window_max
    # by exactly that overhead.
    prelude = await compute_step_prelude(
        pool,
        session_id,
        session=session,
        agent=agent,
        channels=channels,
        memory_store_echoes=memory_echoes,
    )
    overhead_local = (
        approx_tokens(
            [{"role": "system", "content": prelude.system_prompt}],
            tools=prelude.tools,
        )
        + prelude.tail_block_upper_bound_local
    )

    # Read windowed message events for this session.
    events = await sessions_service.read_windowed_events(
        pool,
        session_id,
        window_min=agent.window_min,
        window_max=agent.window_max,
        model=agent.model,
        overhead_local=overhead_local,
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
        return None

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
            session=session,
            agent=agent,
            channels=channels,
            prelude=prelude,
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
            return delay

        await sessions_service.set_session_status(
            pool, session_id, "idle", stop_reason={"type": "error"}
        )
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "error")
        raise

    # ``local_tokens`` costs the full payload (messages + tools) so it
    # matches what the provider counts.  The error branch above stays
    # un-stamped; its ``is_error=True`` alone is enough to keep it out of
    # calibration reads (the partial index and the aggregate query both
    # filter on ``is_error=false``).
    local_tokens = approx_tokens(messages, tools=tools)
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
            "local_tokens": local_tokens,
            "model": agent.model,
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

    if channels:
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
        immediate: list[dict[str, Any]] = []
        mcp_immediate: list[dict[str, Any]] = []
        needs_confirm: list[dict[str, Any]] = []
        custom: list[dict[str, Any]] = []
        unknown_mcp: list[dict[str, Any]] = []

        for tc in tool_calls:
            kind = _classify_tool_call(tc, agent, mcp_server_map)
            if kind == "immediate":
                immediate.append(tc)
            elif kind == "mcp_immediate":
                mcp_immediate.append(tc)
            elif kind == "needs_confirm":
                needs_confirm.append(tc)
            elif kind == "custom":
                custom.append(tc)
            else:  # "unknown_mcp"
                unknown_mcp.append(tc)

        if immediate:
            launch_tool_calls(pool, session_id, immediate)
            log.info(
                "step.tools_launched",
                session_id=session_id,
                count=len(immediate),
                tool_names=[_tc_name(tc) for tc in immediate],
            )

        # Unknown-MCP tools route through the regular MCP dispatcher,
        # bypassing the permission gate.  ``_execute_mcp_tool_async``
        # already detects unknown servers and appends a tool_error
        # event (``mcp_tool.server_not_found``); the prior code path
        # parked the session in ``requires_action`` and dispatched
        # only after a confirmation, which never came.  Routing them
        # to immediate dispatch lets the model see the error in the
        # next step and self-correct.
        immediate_mcp = mcp_immediate + unknown_mcp
        if immediate_mcp:
            launch_mcp_tool_calls(
                pool,
                session_id,
                immediate_mcp,
                mcp_server_map,
                focal_channel=session.focal_channel,
            )
            log.info(
                "step.mcp_tools_launched",
                session_id=session_id,
                count=len(immediate_mcp),
                tool_names=[_tc_name(tc) for tc in immediate_mcp],
                unknown_count=len(unknown_mcp),
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
    return None


def _switch_channel_tool_spec() -> dict[str, Any]:
    """Build the chat-completions tool entry for the ``switch_channel`` built-in.

    Injected unconditionally into the tool list when the session has
    any bound channels (see ``run_session_step``).  Agents don't need
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


def _is_known_mcp_server(server_name: str, mcp_server_map: dict[str, str]) -> bool:
    """Return True if ``server_name`` resolves to a registered MCP server.

    A server is "known" if either:

    * It's a connector instance currently registered with the
      :class:`~aios.harness.connector_supervisor.ConnectorSubprocessRegistry`,
      or
    * It appears in ``mcp_server_map`` (the agent-declared HTTP MCP
      servers, keyed by ``McpServerSpec.name``).

    Used by :func:`_classify_tool_call` to short-circuit hallucinated
    tool names before the permission gate, so the model gets a tool
    error in one turn instead of parking in ``requires_action`` forever
    waiting on a confirmation that would dispatch into
    ``mcp_tool.server_not_found`` anyway.
    """
    registry = runtime.connector_subprocess_registry
    if registry is not None and registry.states_for_connector(server_name):
        return True
    return server_name in mcp_server_map


def _classify_tool_call(
    tool_call: dict[str, Any],
    agent: Any,
    mcp_server_map: dict[str, str],
) -> str:
    """Classify a tool call into a dispatch bucket.

    Returns one of:

    * ``"immediate"`` — built-in tool, run synchronously.
    * ``"mcp_immediate"`` — known MCP tool, ``always_allow``.
    * ``"needs_confirm"`` — built-in or MCP tool gated on
      ``always_ask`` confirmation.
    * ``"custom"`` — client-executed custom tool (the harness holds
      the call until the client posts a tool-result).
    * ``"unknown_mcp"`` — MCP-namespaced tool whose server is not
      registered.  Routed to immediate tool-error so the model can
      self-correct rather than parking in ``requires_action``.
    """
    from aios.tools.registry import registry as tool_registry

    function = tool_call.get("function") or {}
    name: str = function.get("name") or ""

    if _is_mcp_tool(name):
        try:
            server_name = name.split("__", 2)[1]
        except IndexError:
            server_name = ""
        if not server_name or not _is_known_mcp_server(server_name, mcp_server_map):
            return "unknown_mcp"
        perm = resolve_mcp_permission(name, agent.tools)
        if perm is None:
            # Fallback to the operator-configured default for unmounted
            # toolsets (``AIOS_DEFAULT_MCP_PERMISSION_POLICY``).  Unset
            # → ``always_ask`` (safe default); explicit per-toolset
            # policies on ``agent.tools`` always win and are returned
            # above without reaching this branch.
            from aios.config import get_settings

            perm = get_settings().default_mcp_permission_policy
        if perm == "always_allow":
            return "mcp_immediate"
        return "needs_confirm"

    if not tool_registry.has(name):
        return "custom"

    if resolve_permission(name, agent.tools) == "always_ask":
        return "needs_confirm"

    return "immediate"


def resolve_permission(name: str, agent_tools: list[ToolSpec]) -> str | None:
    """Look up the permission policy for a built-in or custom tool by name."""
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.permission
    return None


def resolve_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> str | None:
    """Look up the permission policy for an MCP tool.

    Finds the ``mcp_toolset`` entry whose ``mcp_server_name`` matches
    the server portion of the namespaced tool name, then returns the
    ``default_config.permission_policy.type`` or ``None`` (which callers
    treat as ``always_ask``).
    """
    server_name = name.split("__", 2)[1]
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
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Discover MCP tools from agent-declared servers, filtered by enabled
    ``mcp_toolset`` entries.

    Returns ``(tools, instructions_by_server)`` where the second element
    maps ``server_name`` → the server's ``InitializeResult.instructions``
    string.  Servers that supplied no instructions (or ``""``) are
    omitted from the dict.
    """
    from aios.mcp.client import discover_mcp_tools, resolve_auth_for_url

    enabled_server_names: set[str] = set()
    for spec in agent.tools:
        if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name:
            enabled_server_names.add(spec.mcp_server_name)
    servers: list[tuple[str, str]] = [
        (s.name, s.url) for s in agent.mcp_servers if s.name in enabled_server_names
    ]
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

    # Read the recent tail; on long sessions the default ASC scan would read
    # the oldest 200 lifecycle events and miss any fresh tool_confirmed.
    lifecycle_events = await sessions_service.read_events(
        pool,
        session_id,
        kind="lifecycle",
        newest_first=True,
        limit=200,
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


async def _handle_step_timeout(pool: Any, session_id: str) -> float | None:
    """Synthesize a reschedulable error state when the job-level cap fires.

    Mirrors the rescheduling logic in the litellm-error handler so the
    session ends each step in a clean status regardless of which path
    surfaced the failure. Returns the retry delay (seconds) when the
    backoff budget allows, ``None`` for a terminal failure.
    """
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "step_timeout", "timeout_seconds": _JOB_TIMEOUT_S},
    )
    attempt = await _count_consecutive_rescheduling(pool, session_id)
    delay = _retry_delay_for_attempt(attempt)
    if delay is not None:
        await sessions_service.set_session_status(
            pool, session_id, "rescheduling", stop_reason={"type": "rescheduling"}
        )
        await _append_lifecycle(pool, session_id, "turn_ended", "rescheduling", "rescheduling")
        return delay
    await sessions_service.set_session_status(
        pool, session_id, "idle", stop_reason={"type": "error"}
    )
    await _append_lifecycle(pool, session_id, "turn_ended", "idle", "error")
    return None


async def _count_consecutive_rescheduling(pool: Any, session_id: str) -> int:
    """Count consecutive rescheduling lifecycle events at the tail of the log.

    Returns the number of consecutive ``turn_ended`` lifecycle events
    with ``stop_reason == "rescheduling"`` at the end of the lifecycle
    event sequence. A non-rescheduling event breaks the streak.
    """
    # Only the tail matters; reading ASC with the default LIMIT would miss the
    # recent streak entirely on a session with >limit lifecycle events.
    lifecycle_events = await sessions_service.read_events(
        pool,
        session_id,
        kind="lifecycle",
        newest_first=True,
        limit=len(_RETRY_BACKOFF_SECONDS) + 1,
    )
    count = 0
    for e in lifecycle_events:
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
