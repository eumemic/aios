"""Single-step session harness.

Each procrastinate ``wake_session`` job calls :func:`run_session_step`,
which calls the model exactly once, appends the assistant message, kicks
off tool calls as async tasks, and returns.

Tool results are appended AFTER the step finishes (via an asyncio.Event
gate), so the event log always reflects the order in which the model
perceived events. This is critical for mid-turn injection: if a tool
completes during inference, the model's response was based on a "pending"
snapshot, so the tool result must appear after the assistant message in
the log. The next step then picks it up as a genuinely "new" event.
"""

from __future__ import annotations

import asyncio
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

# Per-session gate: tool tasks wait on this event before appending
# results, ensuring tool_result seqs come after the assistant message.
_step_done: dict[str, asyncio.Event] = {}


async def run_session_step(session_id: str, *, cause: str = "message") -> None:
    """Run one inference step for the session."""
    pool = runtime.require_pool()
    vault = runtime.require_vault()

    session = await sessions_service.get_session(pool, session_id)
    agent = await agents_service.get_agent(pool, session.agent_id)

    events = await sessions_service.read_message_events(pool, session_id)

    if not should_call_model(events):
        log.debug("step.early_out", session_id=session_id, cause=cause)
        return

    api_key: str | None = None
    if agent.credential_id is not None:
        api_key = await credentials_service.decrypt_credential(pool, vault, agent.credential_id)

    tools = to_openai_tools(agent.tools)
    messages = build_messages(
        events,
        system_prompt=agent.system,
        window_min=agent.window_min,
        window_max=agent.window_max,
        model=agent.model,
    )

    await sessions_service.set_session_status(pool, session_id, "running")

    # Create the gate BEFORE calling the model. Tool tasks that complete
    # during inference will wait on this event before appending results.
    done_event = asyncio.Event()
    _step_done[session_id] = done_event

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
    finally:
        # ALWAYS open the gate, even on error. Otherwise tool tasks hang.
        done_event.set()

    await sessions_service.append_event(pool, session_id, "message", assistant_msg)

    tool_calls: list[dict[str, Any]] = assistant_msg.get("tool_calls") or []

    if tool_calls:
        launch_tool_calls(pool, session_id, tool_calls)
        log.info(
            "step.tools_launched",
            session_id=session_id,
            count=len(tool_calls),
            tool_names=[(tc.get("function") or {}).get("name", "?") for tc in tool_calls],
        )
    else:
        await sessions_service.set_session_status(pool, session_id, "idle", stop_reason="end_turn")
        await _append_lifecycle(pool, session_id, "turn_ended", "idle", "end_turn")
        log.info("step.turn_ended", session_id=session_id, cause=cause)


def get_step_done_event(session_id: str) -> asyncio.Event | None:
    """Return the step-done gate for a session, or None if no step is active."""
    return _step_done.get(session_id)


async def _append_lifecycle(
    pool: Any,
    session_id: str,
    event: str,
    status: str,
    stop_reason: str,
) -> None:
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": event, "status": status, "stop_reason": stop_reason},
    )
