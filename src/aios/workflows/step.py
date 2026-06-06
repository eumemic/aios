"""``run_workflow_step`` — one durable wake of a workflow run.

The re-entrant step driven by procrastinate (``lock=run_id``), structurally the
dual of ``run_session_step`` but with a deterministic-Python subprocess where the
model would be. It is the **sole writer** of ``wf_run_events``.

One wake:

1. Load the run + its journal + its signals; rebuild ``memo`` (resolved results)
   and ``inflight`` (opened-but-unresolved capabilities) from the log.
2. On the first wake, append ``run_started`` and flip to ``running``.
3. **Pre-replay harvest**: for any inflight capability that now has a signal,
   journal its ``call_result`` and fold it into ``memo`` — so replay sees a
   maximal memo and fast-forwards past it.
4. Drive one wake of the pinned script in the credential-free subprocess.
5. Raised → ``errored`` (except a transient ``script_host_spawn_failed``, which
   re-raises so the sweep retries — an infra hiccup must not terminally error the
   run). On a *real* replay (returned/suspended) a replay-prefix check fails
   closed on divergence. Returned → ``run_completed`` + ``completed`` (one txn).
   Suspended → open each *new* frontier capability and park as ``suspended``: a
   gate gets a nonce; an ``agent`` spawns a deterministic, idempotent child
   session, delivers its input, and journals ``call_started``. The run re-wakes
   on the next signal / child completion or the periodic sweep.
"""

from __future__ import annotations

import secrets
from typing import Any

import asyncpg

from aios.db import queries as db_queries
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.workflows import WfRun
from aios.services.sessions import create_child_session
from aios.services.wake import defer_run_wake, defer_wake
from aios.workflows.child_id import child_session_id
from aios.workflows.host_launcher import EmittedCapability, run_script_host

log = get_logger("aios.workflows.step")

_TERMINAL = ("completed", "errored")


async def run_workflow_step(run_id: str) -> None:
    pool = runtime.require_pool()

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        if run is None or run.status in _TERMINAL:
            return  # vanished or already terminal — a stray/duplicate wake is a no-op
        account_id = run.account_id

        events = await wf_queries.list_run_events(conn, run_id)
        signals = {s.call_key: s for s in await wf_queries.list_run_signals(conn, run_id)}
        memo: dict[str, Any] = {
            e.call_key: e.payload.get("result")
            for e in events
            if e.type == "call_result" and e.call_key is not None
        }
        inflight: set[str] = {
            e.call_key
            for e in events
            if e.type == "call_started" and e.call_key is not None and e.call_key not in memo
        }

        if not events:
            await wf_queries.append_run_event(
                conn,
                account_id=account_id,
                run_id=run_id,
                type="run_started",
                payload={"input": run.input},
            )
            await wf_queries.set_run_status(conn, run_id, "running", account_id=account_id)

        # Pre-replay harvest: resolve any inflight capability that has a signal.
        for call_key in list(inflight):
            sig = signals.get(call_key)
            if sig is None:
                continue
            await wf_queries.append_run_event(
                conn,
                account_id=account_id,
                run_id=run_id,
                type="call_result",
                call_key=call_key,
                payload={"result": sig.result, "is_error": False},
            )
            memo[call_key] = sig.result
            inflight.discard(call_key)

    # Drive one wake in the credential-free subprocess (no DB conn held).
    outcome = await run_script_host(source=run.script, input=run.input, memo=memo)

    needs_rewake = False
    async with pool.acquire() as conn:
        if outcome.kind == "raised":
            if outcome.error_kind == "script_host_spawn_failed":
                # Failure to *spawn* the host (EAGAIN/ENOMEM at fork) is a transient
                # worker-infra fault, not the script's — never terminally error the
                # run for it. Fail the step so the lock releases and the periodic
                # sweep re-wakes the (still non-terminal) run when capacity returns.
                raise RuntimeError(f"workflow {run_id}: {outcome.error_repr}")
            await _complete_run(
                conn, run, output=outcome.error_repr, is_error=True, error_kind=outcome.error_kind
            )
            return

        # The host completed a real replay (suspended or returned), so every
        # still-open capability MUST have been re-reached. Replay-prefix assertion
        # (fail-closed): if one wasn't re-emitted, the script diverged
        # (nondeterminism) — error loudly rather than orphan the old call_started.
        # Gated on a non-crash outcome: a crashed/killed host emits nothing, and
        # that emptiness is a crash (handled above), never divergence.
        emitted_keys = {cap.call_key for cap in outcome.emitted}
        diverged = sorted(k for k in inflight if k not in emitted_keys)
        if diverged:
            await _complete_run(
                conn,
                run,
                output=f"nondeterministic replay: open capabilities {diverged} were not re-emitted",
                is_error=True,
                error_kind="nondeterministic_replay",
            )
            return

        if outcome.kind == "returned":
            await _complete_run(conn, run, output=outcome.value, is_error=False)
            return

        # Suspended: open any *new* frontier capability, then park.
        for cap in outcome.emitted:
            if cap.call_key in memo or cap.call_key in inflight:
                continue  # already resolved, or already open (an idempotent re-emit)
            if cap.capability_id == "gate":
                nonce = secrets.token_urlsafe(32)
                await wf_queries.append_run_event(
                    conn,
                    account_id=account_id,
                    run_id=run_id,
                    type="call_started",
                    call_key=cap.call_key,
                    payload={"capability": "gate", "gate_nonce": nonce},
                )
                # A resume signal can land before this call_started is journaled
                # (the call_key is derivable); the pre-replay harvest only sees
                # gates already inflight, so a self-wake harvests this freshly
                # opened gate promptly instead of waiting for the periodic sweep.
                if cap.call_key in signals:
                    needs_rewake = True
            elif cap.capability_id == "agent":
                if await _open_agent_capability(conn, pool, run, cap):
                    return  # the frontier was terminally rejected (bad call) — run errored
            else:
                # parallel/pipeline reach the step as their `agent` leaves (B2.G);
                # any other capability id is unknown.
                await _complete_run(
                    conn,
                    run,
                    output=f"capability {cap.capability_id!r} is not supported yet",
                    is_error=True,
                    error_kind="not_implemented",
                )
                return
        await wf_queries.set_run_status(conn, run_id, "suspended", account_id=account_id)

    # Outside the txn (after the suspend commits): a self-wake so the next step
    # harvests an already-delivered resume for a gate opened this wake.
    if needs_rewake:
        await defer_run_wake(run_id)


async def _open_agent_capability(
    conn: asyncpg.Connection[Any],
    pool: asyncpg.Pool[Any],
    run: WfRun,
    cap: EmittedCapability,
) -> bool:
    """Spawn (or, on replay/C1', re-attach) the ``agent()`` child for a frontier,
    then journal ``call_started{child_session_id, child_agent_version}``.

    Idempotent + crash-safe: the child id is deterministic, ``create_child_session``
    is ``ON CONFLICT`` (delivers the input atomically with the row on first spawn),
    ``call_started`` dedups on the memo, and ``defer_wake`` dedups on its queueing
    lock. On replay the child id already exists → re-attach, journaling the *row's*
    pinned version (not a re-resolved one). Returns ``True`` iff the call was
    terminally rejected (the run was errored — the caller should stop the step).
    """
    account_id = run.account_id
    spec = cap.spec if isinstance(cap.spec, dict) else {}
    if spec.get("output_schema") is not None:
        await _complete_run(
            conn,
            run,
            output="agent() output_schema is not supported in v1 (lands in Block 3)",
            is_error=True,
            error_kind="not_implemented",
        )
        return True
    agent_id = spec.get("agent_id")
    if not isinstance(agent_id, str):
        await _complete_run(
            conn,
            run,
            output=f"agent() requires a string agent_id, got {agent_id!r}",
            is_error=True,
            error_kind="bad_agent_call",
        )
        return True

    child_id = child_session_id(run.id, cap.call_key)
    try:
        pinned = (await db_queries.get_agent(conn, agent_id, account_id=account_id)).version
    except NotFoundError:
        await _complete_run(
            conn,
            run,
            output=f"agent {agent_id!r} not found",
            is_error=True,
            error_kind="agent_not_found",
        )
        return True

    created = await create_child_session(
        pool,
        session_id=child_id,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=run.environment_id,
        agent_version=pinned,
        parent_run_id=run.id,
        input=spec.get("input"),
    )
    # On replay the row already carries its first-spawn version — journal THAT, so
    # call_started.child_agent_version always matches the version the child runs under.
    child_version = (
        pinned
        if created
        else (
            await db_queries.get_session_bare(conn, child_id, account_id=account_id)
        ).agent_version
    )
    await wf_queries.append_run_event(
        conn,
        account_id=account_id,
        run_id=run.id,
        type="call_started",
        call_key=cap.call_key,
        payload={
            "capability": "agent",
            "child_session_id": child_id,
            "child_agent_version": child_version,
        },
    )
    # Prompt wake of the child; the session sweep is the durable backstop (the
    # child's first user message makes it is-wakeable regardless).
    await defer_wake(pool, child_id, account_id=account_id, cause="workflow_child_spawn")
    return False


async def _complete_run(
    conn: asyncpg.Connection[Any],
    run: WfRun,
    *,
    output: Any,
    is_error: bool,
    error_kind: str | None = None,
) -> None:
    """Append ``run_completed`` + flip status (+ store output) atomically."""
    payload: dict[str, Any] = {"output": output, "is_error": is_error}
    if error_kind is not None:
        payload["error"] = {"kind": error_kind}
    async with conn.transaction():
        await wf_queries.append_run_event(
            conn, account_id=run.account_id, run_id=run.id, type="run_completed", payload=payload
        )
        await wf_queries.set_run_terminal(
            conn,
            run.id,
            status="errored" if is_error else "completed",
            output=None if is_error else output,
            account_id=run.account_id,
        )
