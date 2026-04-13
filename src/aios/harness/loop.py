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
from aios.harness.completion import call_litellm
from aios.harness.context import build_messages, should_call_model
from aios.harness.tool_dispatch import launch_tool_calls
from aios.logging import get_logger
from aios.services import agents as agents_service
from aios.services import credentials as credentials_service
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
    vault = runtime.require_vault()

    session = await sessions_service.get_session(pool, session_id)
    agent = await agents_service.get_agent(pool, session.agent_id)

    # Read all message events for this session.
    events = await sessions_service.read_message_events(pool, session_id)

    # Early-out: nothing actionable for the model.
    if not should_call_model(events):
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return

    # Decrypt credential. Plaintext lives only on this stack frame.
    api_key: str | None = None
    if agent.credential_id is not None:
        api_key = await credentials_service.decrypt_credential(pool, vault, agent.credential_id)

    # Build context with pending-result synthesis.
    tools = to_openai_tools(agent.tools)
    messages = build_messages(
        events,
        system_prompt=agent.system,
        window_min=agent.window_min,
        window_max=agent.window_max,
        model=agent.model,
    )

    # Mark session as running.
    await sessions_service.set_session_status(pool, session_id, "running")

    # Call the model exactly once.
    try:
        assistant_msg = await call_litellm(
            model=agent.model,
            messages=messages,
            tools=tools if tools else None,
            api_key=api_key,
        )
    except Exception:
        log.exception("step.litellm_failed", session_id=session_id)
        await sessions_service.set_session_status(pool, session_id, "idle", stop_reason="error")
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "error")
        raise

    # Append assistant message to the session log (unfenced — procrastinate
    # lock provides mutual exclusion).
    await sessions_service.append_event(pool, session_id, "message", assistant_msg)

    # Check for tool calls.
    tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []

    if tool_calls:
        # Fire-and-forget: each task appends its result and defers a wake.
        launch_tool_calls(pool, session_id, tool_calls)
        log.info(
            "step.tools_launched",
            session_id=session_id,
            count=len(tool_calls),
            tool_names=[(tc.get("function") or {}).get("name", "?") for tc in tool_calls],
        )
    else:
        # No tool calls — the model's turn is done.
        await sessions_service.set_session_status(pool, session_id, "idle", stop_reason="end_turn")
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "end_turn")
        log.info("step.turn_ended", session_id=session_id, cause=cause)


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
