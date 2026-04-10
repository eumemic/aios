"""Phase 1 inline harness loop.

This is the simplest possible runnable agent loop:

1. Read the agent record (system prompt, model, credential, windowing config)
2. Read every message-kind event for the session
3. Apply the chunked stable-prefix windowing
4. Build a chat-completions request: ``[system] + windowed messages``
5. Call LiteLLM
6. Persist the assistant message as a new event
7. Emit a turn_ended lifecycle event and set the session to idle

Phase 1 has no tools, no Docker sandbox, no worker, no cancellation. The
caller (the API request handler) awaits this function inline. Phases 2-5
extract this into an async worker, add tool dispatch, and add interrupt /
resume semantics.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.crypto.vault import Vault
from aios.harness.completion import call_litellm
from aios.harness.tokens import token_count_for_event
from aios.harness.window import select_window
from aios.services import agents as agents_service
from aios.services import credentials as credentials_service
from aios.services import sessions as sessions_service


async def run_session_turn(
    pool: asyncpg.Pool[Any],
    vault: Vault,
    session_id: str,
) -> None:
    """Run a single turn of the session loop until the model is idle.

    This advances the session through one or more LLM calls, persisting each
    response as a message event. In Phase 1 (no tools), the loop terminates
    after the first response — there are no tool_calls to dispatch.
    """
    session = await sessions_service.get_session(pool, session_id)
    agent = await agents_service.get_agent(pool, session.agent_id)

    # Read all message events for this session, ordered by seq.
    msg_events = await sessions_service.read_message_events(pool, session_id)

    # Apply the windowing function.
    windowed = select_window(
        msg_events,
        min_tokens=agent.window_min,
        max_tokens=agent.window_max,
        token_counter=lambda e: token_count_for_event(e, model=agent.model),
    )

    # Build the chat-completions messages array: system + windowed message bodies.
    messages: list[dict[str, Any]] = []
    if agent.system:
        messages.append({"role": "system", "content": agent.system})
    for evt in windowed:
        messages.append(evt.data)

    # Decrypt credential if any. Plaintext lives only on this stack frame.
    api_key: str | None = None
    if agent.credential_id is not None:
        api_key = await credentials_service.decrypt_credential(pool, vault, agent.credential_id)

    # Mark session running, then call litellm.
    await sessions_service.set_session_status(pool, session_id, "running")
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {"event": "turn_started", "status": "running"},
    )

    try:
        message_dict = await call_litellm(
            model=agent.model,
            messages=messages,
            tools=None,  # Phase 1: no tools
            api_key=api_key,
        )
    except Exception as exc:
        await sessions_service.append_event(
            pool,
            session_id,
            "lifecycle",
            {
                "event": "turn_ended",
                "status": "idle",
                "stop_reason": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        await sessions_service.set_session_status(pool, session_id, "idle", "error")
        raise

    # Persist the assistant message exactly as litellm returned it.
    await sessions_service.append_event(pool, session_id, "message", message_dict)

    # Phase 1: no tool_calls handling. End the turn.
    await sessions_service.append_event(
        pool,
        session_id,
        "lifecycle",
        {
            "event": "turn_ended",
            "status": "idle",
            "stop_reason": "end_turn",
        },
    )
    await sessions_service.set_session_status(pool, session_id, "idle", "end_turn")
