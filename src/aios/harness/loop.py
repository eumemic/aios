"""Harness loop.

The core function :func:`run_session_turn` runs a single turn of the session
loop: read the agent record, window the events, call LiteLLM, persist the
response. It's been the same shape since Phase 1 — pure I/O on top of the DB.

Phase 2 introduces :func:`run_session_turn_with_lease`, which wraps
``run_session_turn`` in lease management:

* acquire the lease (reschedule if another worker holds it)
* start a background refresh task with a cancel event
* check the "last message was assistant" early-return guard
* run the turn
* on success: release the lease cleanly
* on :class:`~aios.harness.lease.LeaseLost`: skip the release (the new
  lease holder owns the session now), exit the task

Phase 2 only checks the cancel event between turns, not mid-LiteLLM-call.
Phase 5 will thread the cancel event into ``call_litellm`` for true mid-call
cancellation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from aios.crypto.vault import Vault
from aios.harness import runtime
from aios.harness.completion import call_litellm
from aios.harness.lease import (
    LeaseLost,
    acquire_lease,
    lease_refresher,
    release_lease,
)
from aios.harness.tokens import token_count_for_event
from aios.harness.window import select_window
from aios.logging import get_logger
from aios.services import agents as agents_service
from aios.services import credentials as credentials_service
from aios.services import sessions as sessions_service

log = get_logger("aios.harness.loop")


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


# ─── Phase 2: lease-wrapped variant ──────────────────────────────────────────


async def run_session_turn_with_lease(session_id: str, *, cause: str = "message") -> None:
    """Run a single turn under a row-level lease.

    This is the entrypoint procrastinate's ``wake_session`` task calls. It:

    1. Acquires the lease (if another worker holds it, defers a retry job and
       returns).
    2. Starts a background refresh task that extends the lease periodically.
    3. Checks the early-return guard: if the most recent message in the log
       is already from the assistant, there's nothing for this turn to do
       (a duplicate wake job, or a coalesced wake that arrived after the
       previous turn already addressed all pending messages). Release the
       lease and exit.
    4. Calls :func:`run_session_turn` to do the actual work.
    5. On normal completion, releases the lease cleanly.
    6. On :class:`LeaseLost` (the fenced append detected another worker took
       over), skips the release because the new lease holder owns the
       session now.

    All long-lived state (pool, vault, worker_id) is read from
    :mod:`aios.harness.runtime`. The api process never calls this function;
    only the worker does.

    The ``cause`` parameter is observability-only — never branch loop
    behavior on it. The loop must be fully recoverable from the events log
    alone.
    """
    pool = runtime.require_pool()
    vault = runtime.require_vault()
    worker_id = runtime.require_worker_id()

    bound_log = log.bind(session_id=session_id, worker_id=worker_id, cause=cause)

    # 1. Acquire the lease.
    last_seq = await acquire_lease(pool, session_id, worker_id)
    if last_seq is None:
        # Another worker holds it. Defer a reschedule with a delay so we
        # don't busy-loop while waiting for the held lease to expire. The
        # delay is keyed off `lease_reschedule_delay_seconds` from settings.
        from aios.config import get_settings
        from aios.harness.procrastinate_app import app as procrastinate_app

        delay = get_settings().lease_reschedule_delay_seconds
        bound_log.info("lease.busy", reschedule_seconds=delay)
        await procrastinate_app.configure_task(
            "harness.wake_session",
            schedule_in={"seconds": delay},
        ).defer_async(
            session_id=session_id,
            cause="reschedule",
        )
        return

    bound_log.info("lease.acquired", last_event_seq=last_seq)
    cancel = asyncio.Event()
    try:
        async with lease_refresher(pool, session_id, worker_id, cancel):
            # 2. Early-return guard: if the most recent message is already
            # from the assistant, this wake is a duplicate or a coalesced
            # spillover. Nothing for this turn to do.
            msg_events = await sessions_service.read_message_events(pool, session_id)
            if msg_events and msg_events[-1].data.get("role") == "assistant":
                bound_log.info(
                    "loop.no_op_duplicate_wake",
                    last_seq=msg_events[-1].seq,
                )
                # Fall through to the release in `finally`.
                return

            if cancel.is_set():
                bound_log.warning("lease.lost_before_turn")
                return

            # 3. Run the turn. Inherits all of Phase 1's logic.
            try:
                await run_session_turn(pool, vault, session_id)
            except LeaseLost:
                bound_log.warning("lease.lost_during_turn")
                # Skip the release; the new lease holder owns the session.
                # Suppress the LeaseLost so the procrastinate task ends
                # cleanly without retry.
                return
            except Exception:
                bound_log.exception("loop.unexpected_error")
                raise
    finally:
        if not cancel.is_set():
            # Normal completion: release the lease cleanly.
            await release_lease(
                pool,
                session_id,
                worker_id,
                new_status="idle",
                stop_reason="end_turn",
            )


# Re-exported for backwards compatibility with tests/integrations that pass
# the pool and vault explicitly. The Phase 2 worker entrypoint uses the
# `_with_lease` variant; Phase 1 tests still call `run_session_turn` directly.
__all__ = ["run_session_turn", "run_session_turn_with_lease"]


# Avoid an unused-import warning while keeping `asyncpg`/`Vault` imports
# meaningful: the module-level type annotations on `run_session_turn` use them.
_ = asyncpg
_ = Vault
