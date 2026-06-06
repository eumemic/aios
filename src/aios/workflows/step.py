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
5. Returned → ``run_completed`` + ``completed`` (one txn). Raised → ``errored``.
   Suspended → open any *new* frontier capability (a gate gets a nonce +
   ``call_started``) and park as ``suspended``; the run re-wakes on the next
   signal or sweep.
"""

from __future__ import annotations

import secrets
from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.workflows import WfRun
from aios.workflows.host_launcher import run_script_host

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
        inflight: dict[str, dict[str, Any]] = {
            e.call_key: e.payload
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
            del inflight[call_key]

    # Drive one wake in the credential-free subprocess (no DB conn held).
    outcome = await run_script_host(source=run.script, input=run.input, memo=memo)

    async with pool.acquire() as conn:
        # Replay-prefix assertion (fail-closed): every still-open capability must
        # be re-reached by this replay. If one wasn't, the script emitted a
        # divergent sequence (nondeterminism) — error loudly rather than orphan
        # the old call_started and silently open a fresh one.
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
        if outcome.kind == "raised":
            await _complete_run(
                conn, run, output=outcome.error_repr, is_error=True, error_kind=outcome.error_kind
            )
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
            else:
                # agent / parallel / pipeline are not openable in Block 1.
                await _complete_run(
                    conn,
                    run,
                    output=f"capability {cap.capability_id!r} is not supported yet (lands in Block 2)",
                    is_error=True,
                    error_kind="not_implemented",
                )
                return
        await wf_queries.set_run_status(conn, run_id, "suspended", account_id=account_id)


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
        if not is_error:
            await wf_queries.set_run_output(conn, run.id, output, account_id=run.account_id)
        await wf_queries.set_run_status(
            conn, run.id, "errored" if is_error else "completed", account_id=run.account_id
        )
