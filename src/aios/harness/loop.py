"""Harness loop.

The core function :func:`run_session_turn` runs a single turn of the
session loop: read the agent record, window the events, call LiteLLM,
dispatch any tool calls, repeat until the model returns no more tool
calls. Phase 1 left this as a single LLM call; Phase 3 turns it into the
in-process assistant→tool→assistant chain that makes agents actually
useful.

Phase 2 introduces :func:`run_session_turn_with_lease`, which wraps
``run_session_turn`` in lease management:

* acquire the lease (reschedule if another worker holds it)
* start a background refresh task with a cancel event
* check the "last message was assistant" early-return guard
* run the turn
* on success: release the lease cleanly
* on :class:`~aios.harness.lease.LeaseLost`: skip the release (the new
  lease holder owns the session now), exit the task

Phase 3 additions:

* The inner loop runs assistant → tool dispatch → assistant → … until
  the model returns a response with no ``tool_calls``. Lease refresh
  runs in the background throughout; a long tool chain does not require
  any new checkpoint machinery.
* Assistant and tool messages are appended via
  :func:`append_event_with_fence` so they respect the lease. User
  messages are still appended via the unfenced ``append_event`` on the
  API side (user input must always land).
* After the lease is released, the worker's sandbox registry tears down
  the session's container (if one was ever provisioned). Chat-only
  sessions never created one, so teardown is a no-op.

Phase 2 only checks the cancel event between turns, not mid-LiteLLM-call.
Phase 5 will thread the cancel event into ``call_litellm`` for true
mid-call cancellation.
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
    append_event_with_fence,
    lease_refresher,
    release_lease,
)
from aios.harness.tokens import token_count_for_event
from aios.harness.tool_dispatch import dispatch_tool_calls
from aios.harness.window import select_window
from aios.logging import get_logger
from aios.models.agents import Agent
from aios.services import agents as agents_service
from aios.services import credentials as credentials_service
from aios.services import sessions as sessions_service
from aios.tools.registry import to_openai_tools

log = get_logger("aios.harness.loop")

# Safety cap on the number of assistant → tool → assistant iterations per
# turn. A well-behaved agent stops on its own after a handful of steps.
# A broken one could otherwise loop until the lease expires. 50 is well
# past any reasonable real-world tool chain; bump if needed.
MAX_LOOP_ITERATIONS = 50


async def run_session_turn(
    pool: asyncpg.Pool[Any],
    vault: Vault,
    session_id: str,
    *,
    worker_id: str | None = None,
) -> None:
    """Run a session turn: model → tools → model → … until no tool_calls.

    ``worker_id`` is required for fenced appends — assistant and tool
    messages must carry the lease's worker id so a lost lease is
    detected at the DB boundary. Phase 1 tests call this function
    without a worker_id; in that path the fenced append is skipped in
    favour of the unfenced ``append_event`` (matching Phase 1 semantics
    for unit tests that don't exercise the lease protocol).
    """
    session = await sessions_service.get_session(pool, session_id)
    agent = await agents_service.get_agent(pool, session.agent_id)

    # Decrypt credential if any. Plaintext lives only on this stack frame.
    api_key: str | None = None
    if agent.credential_id is not None:
        api_key = await credentials_service.decrypt_credential(pool, vault, agent.credential_id)

    # Translate the agent's declared tools to the LiteLLM tools parameter.
    # An empty tools list (Phase 1 chat-only agent) is fine; call_litellm
    # drops the kwarg.
    openai_tools = to_openai_tools(agent.tools) if agent.tools else None

    # Mark session running and emit the turn_started lifecycle event.
    await sessions_service.set_session_status(pool, session_id, "running")
    await _append_lifecycle(
        pool,
        session_id=session_id,
        worker_id=worker_id,
        data={"event": "turn_started", "status": "running"},
    )

    stop_reason = "end_turn"
    try:
        for _iteration in range(MAX_LOOP_ITERATIONS):
            # Build the window fresh each iteration. This is correct
            # (captures tool messages we just appended) and keeps the
            # loop stateless — the event log is the source of truth.
            messages = await _build_messages(pool, agent, session_id)

            try:
                message_dict = await call_litellm(
                    model=agent.model,
                    messages=messages,
                    tools=openai_tools,
                    api_key=api_key,
                )
            except Exception as exc:
                await _append_lifecycle(
                    pool,
                    session_id=session_id,
                    worker_id=worker_id,
                    data={
                        "event": "turn_ended",
                        "status": "idle",
                        "stop_reason": "error",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                )
                await sessions_service.set_session_status(pool, session_id, "idle", "error")
                raise

            # Append the assistant message as-is (LiteLLM's dict round-trips
            # reasoning_content / thinking_blocks / tool_calls untouched).
            await _append_message(
                pool,
                session_id=session_id,
                worker_id=worker_id,
                data=message_dict,
            )

            tool_calls = message_dict.get("tool_calls") or []
            if not tool_calls:
                stop_reason = "end_turn"
                break

            if worker_id is None:
                # Without a lease we can't safely run tools — the fenced
                # append inside dispatch_tool_calls would fail.
                raise RuntimeError(
                    "run_session_turn received tool_calls but no worker_id; "
                    "tool dispatch requires the lease-wrapped entrypoint"
                )

            await dispatch_tool_calls(
                pool,
                session_id=session_id,
                worker_id=worker_id,
                tool_calls=tool_calls,
            )
            # Continue the loop — next iteration calls the model with
            # the tool messages now in the log.
        else:
            # Safety cap tripped. Append a lifecycle event explaining why
            # and bail. The session goes idle; the agent can continue by
            # being woken again.
            log.warning(
                "loop.max_iterations_reached",
                session_id=session_id,
                max=MAX_LOOP_ITERATIONS,
            )
            stop_reason = "max_iterations"
    except Exception:
        # call_litellm failure above already emitted turn_ended. Any
        # other exception (including LeaseLost) propagates without
        # emitting a duplicate turn_ended.
        raise

    await _append_lifecycle(
        pool,
        session_id=session_id,
        worker_id=worker_id,
        data={
            "event": "turn_ended",
            "status": "idle",
            "stop_reason": stop_reason,
        },
    )
    await sessions_service.set_session_status(pool, session_id, "idle", stop_reason)


async def _build_messages(
    pool: asyncpg.Pool[Any],
    agent: Agent,
    session_id: str,
) -> list[dict[str, Any]]:
    """Read the session's message events, window them, add the system prompt."""
    msg_events = await sessions_service.read_message_events(pool, session_id)
    windowed = select_window(
        msg_events,
        min_tokens=agent.window_min,
        max_tokens=agent.window_max,
        token_counter=lambda e: token_count_for_event(e, model=agent.model),
    )
    messages: list[dict[str, Any]] = []
    if agent.system:
        messages.append({"role": "system", "content": agent.system})
    for evt in windowed:
        messages.append(evt.data)
    return messages


async def _append_message(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    worker_id: str | None,
    data: dict[str, Any],
) -> None:
    """Append a ``message`` event, fenced if we have a worker_id."""
    if worker_id is None:
        await sessions_service.append_event(pool, session_id, "message", data)
        return
    await append_event_with_fence(
        pool,
        session_id=session_id,
        expected_worker_id=worker_id,
        kind="message",
        data=data,
    )


async def _append_lifecycle(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    worker_id: str | None,
    data: dict[str, Any],
) -> None:
    """Append a ``lifecycle`` event, fenced if we have a worker_id."""
    if worker_id is None:
        await sessions_service.append_event(pool, session_id, "lifecycle", data)
        return
    await append_event_with_fence(
        pool,
        session_id=session_id,
        expected_worker_id=worker_id,
        kind="lifecycle",
        data=data,
    )


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

            # 3. Run the turn. Phase 3 adds the in-process tool loop
            # inside run_session_turn — we pass worker_id so fenced
            # appends can enforce the lease.
            try:
                await run_session_turn(pool, vault, session_id, worker_id=worker_id)
            except LeaseLost:
                bound_log.warning("lease.lost_during_turn")
                # Skip the release; the new lease holder owns the session.
                # Suppress the LeaseLost so the procrastinate task ends
                # cleanly without retry. The new lease holder will tear
                # down any container that worker provisioned when their
                # turn completes.
                return
            except Exception:
                bound_log.exception("loop.unexpected_error")
                raise
    finally:
        # Tear down the session's sandbox container (if one was
        # provisioned during this turn). Chat-only sessions never
        # created one, so this is a cheap dict lookup and return.
        # Release is a best-effort teardown; we log and continue on
        # failure to make sure the lease release still runs.
        if runtime.sandbox_registry is not None:
            try:
                await runtime.sandbox_registry.release(session_id)
            except Exception:
                bound_log.exception("sandbox.release_failed")

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
