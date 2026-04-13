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

from aios.harness import runtime
from aios.harness.completion import stream_litellm
from aios.harness.context import build_messages, should_call_model
from aios.harness.tool_dispatch import launch_tool_calls
from aios.logging import get_logger
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools.registry import to_openai_tools

log = get_logger("aios.harness.loop")


async def run_session_step(session_id: str, *, cause: str = "message") -> None:
    """Run one inference step for the session.

    Called by the procrastinate ``wake_session`` task. The procrastinate
    ``lock`` parameter guarantees only one step runs per session at a
    time.
    """
    pool = runtime.require_pool()

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

    # Read all message events for this session.
    events = await sessions_service.read_message_events(pool, session_id)

    # Check for confirmed-but-undispatched tool calls (always_ask → allow).
    # This must run before should_call_model because the confirmed tool has
    # no result in the log yet — should_call_model would return False.
    pending = await _dispatch_confirmed_tools(pool, session_id, events)
    if pending:
        launch_tool_calls(pool, session_id, pending)
        log.info(
            "step.confirmed_tools_dispatched",
            session_id=session_id,
            count=len(pending),
        )
        return

    # Early-out: nothing actionable for the model.
    if not should_call_model(events):
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return

    # Build context with pending-result synthesis.
    tools = to_openai_tools(agent.tools)
    ctx = build_messages(
        events,
        system_prompt=agent.system,
        window_min=agent.window_min,
        window_max=agent.window_max,
        model=agent.model,
    )

    # Mark session as running.
    await sessions_service.set_session_status(pool, session_id, "running")

    # Emit span start so consumers can measure inference latency.
    start_event = await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {"event": "model_request_start"},
    )

    # Call the model exactly once (streaming — deltas go to SSE via pg_notify).
    try:
        assistant_msg, usage = await stream_litellm(
            model=agent.model,
            messages=ctx.messages,
            tools=tools if tools else None,
            pool=pool,
            session_id=session_id,
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
            },
        )

        # Count consecutive rescheduling lifecycle events to decide
        # whether to retry or give up.
        consecutive = await _count_consecutive_rescheduling(pool, session_id)
        if consecutive < 2:
            # Retry: set status to rescheduling and defer a delayed wake.
            await sessions_service.set_session_status(
                pool, session_id, "rescheduling", stop_reason={"type": "rescheduling"}
            )
            await _append_lifecycle(pool, session_id, "turn_ended", "rescheduling", "rescheduling")
            from aios.harness.procrastinate_app import app as procrastinate_app

            try:
                await procrastinate_app.configure_task("harness.wake_session").defer_async(
                    session_id=session_id,
                    cause="reschedule",
                    schedule_in={"seconds": 5},
                )
            except Exception:
                log.exception("step.reschedule_defer_failed", session_id=session_id)
            return
        else:
            # 3rd consecutive error — give up.
            await sessions_service.set_session_status(
                pool, session_id, "idle", stop_reason={"type": "error"}
            )
            await _append_lifecycle(pool, session_id, "turn_ended", "idle", "error")
            raise

    # Emit span end with per-request token usage.
    await sessions_service.append_event(
        pool,
        session_id,
        "span",
        {
            "event": "model_request_end",
            "model_request_start_id": start_event.id,
            "is_error": False,
            "model_usage": usage,
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

    # Inject reacting_to so should_call_model knows what this response
    # was based on. This is the seq of the latest user/tool event in the
    # context — events after this seq are "new" on the next wake.
    assistant_msg["reacting_to"] = ctx.reacting_to

    # Append assistant message to the session log (unfenced — procrastinate
    # lock provides mutual exclusion).
    await sessions_service.append_event(pool, session_id, "message", assistant_msg)

    # Check for tool calls — partition into three buckets:
    #   immediate       — built-in with always_allow (execute now)
    #   needs_confirm   — built-in with always_ask (hold for client confirmation)
    #   custom          — not in registry (hold for client execution)
    tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []

    if tool_calls:
        from aios.tools.registry import registry as tool_registry

        def _tc_name(tc: dict[str, Any]) -> str:
            name: str = (tc.get("function") or {}).get("name", "")
            return name

        immediate: list[dict[str, Any]] = []
        needs_confirm: list[dict[str, Any]] = []
        custom: list[dict[str, Any]] = []

        for tc in tool_calls:
            name = _tc_name(tc)
            if not tool_registry.has(name):
                custom.append(tc)
            elif _resolve_permission(name, agent.tools) == "always_ask":
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


def _resolve_permission(name: str, agent_tools: list[ToolSpec]) -> str | None:
    """Look up the permission policy for a tool by name."""
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.permission
    return None


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
