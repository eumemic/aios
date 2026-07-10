"""``run_workflow_step`` — one durable wake of a workflow run.

The re-entrant step driven by procrastinate (``lock=run_id``), structurally the
dual of ``run_session_step`` but with a deterministic-Python subprocess where the
model would be. It is the **sole writer** of ``wf_run_events``.

One wake:

1. Load the run + its journal + its signals; rebuild ``memo`` (resolved results)
   and ``inflight`` (opened-but-unresolved capabilities) from the log.
2. On the first wake, append ``run_started`` and flip to ``running``.
3. **Pre-replay harvest**: for any inflight capability now done — a gate with a
   delivered resume signal, or an agent child with a ``request_response`` for the
   call's request — journal its ``call_result`` and fold it into ``memo`` so replay
   sees a maximal memo and fast-forwards past it.
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

import contextlib
import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NamedTuple

import asyncpg
import jsonschema
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012
from structlog.contextvars import bind_contextvars, clear_contextvars

from aios.config import get_settings
from aios.db import queries as db_queries
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError, RateLimitedError
from aios.harness import runtime
from aios.jobs.app import defer_run_wake, defer_trigger_fire, defer_wake
from aios.logging import get_logger
from aios.models.attenuation import api_base_of, surface_of
from aios.models.sessions import Err, Outcome
from aios.models.workflows import TERMINAL_RUN_STATUSES, WfRun, WfRunEvent, WfRunStatus
from aios.services import attenuation as attenuation_service
from aios.services.model_binding_authz import is_workflow_binding
from aios.services.sessions import (
    AskNewSession,
    create_child_session,
    fail_open_child_requests_conn,
    seed_outbound_cancel_conn,
    write_gate_opened,
)
from aios.tools.registry import tool_executes_class
from aios.workflows import run_llm, run_sandbox, run_tools
from aios.workflows.child_id import child_session_id
from aios.workflows.child_run_id import child_run_id
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from aios.workflows.host_launcher import EmittedCapability, run_script_host
from aios.workflows.service import WorkflowRunDepthExceededError, create_run

log = get_logger("aios.workflows.step")


def _collect_refs(node: Any) -> list[str]:
    """Every ``$ref`` / ``$dynamicRef`` / ``$recursiveRef`` string anywhere in a schema."""
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$dynamicRef", "$recursiveRef") and isinstance(value, str):
                refs.append(value)
            refs.extend(_collect_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_collect_refs(item))
    return refs


def _unresolvable_ref(schema: dict[str, Any]) -> str | None:
    """The first reference in ``schema`` that does not resolve within the schema itself,
    or ``None`` if every reference resolves.

    A dangling local ``$ref`` or a remote ``$ref`` passes ``check_schema`` but raises a
    referencing error when the child *applies* the schema — bricking the child (it can
    never ``return`` a conforming value) until its wall-clock deadline. Rejecting it at
    the spawn gate turns that into a clean author-facing error. Remote refs are
    unresolvable by design: a sandboxed child must not fetch a schema over the network.
    Valid self-contained refs (``#/$defs/…``) resolve and are left alone.
    """
    refs = _collect_refs(schema)
    if not refs:
        return None
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    resolver = Registry().with_resource("", resource).resolver(base_uri="")
    for ref in refs:
        try:
            resolver.lookup(ref)
        except Unresolvable:
            return ref
    return None


def _validate_output_against_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Validate a run's terminal ``output`` against the request's ``output_schema``.

    ``None`` on success; otherwise a human-readable message enumerating every
    failure (the same shape the session ``return`` tool produces, minus the
    self-correct hint — a run does NOT bounce-and-retry, it fails loud). Drives
    the run-target ``output_schema_violation`` error-arm in :func:`_complete_run`.
    """
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(value),
        key=lambda e: list(e.absolute_path),
    )
    if not errors:
        return None
    lines = [f"run output does not match the request's required schema: {json.dumps(value)}"]
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path)
        lines.append(f"  - at {'output.' + path if path else 'output'}: {err.message}")
    return "\n".join(lines)


def _usage_payload(usage: wf_queries.RunChildrenUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cost_usd": usage.cost_microusd / 1_000_000,
    }


def _budget_view(
    run: WfRun, usage: wf_queries.RunChildrenUsage, call_llm_cost_microusd: int = 0
) -> dict[str, float] | None:
    if run.budget_usd is None:
        return None
    # The script-facing spend must match what the over-budget gate enforces: the
    # child-session rollup PLUS the run's own call_llm inference meter (#1633).
    spent = (usage.cost_microusd + call_llm_cost_microusd) / 1_000_000
    return {
        "total_usd": run.budget_usd,
        "spent_usd": spent,
        "remaining_usd": max(run.budget_usd - spent, 0.0),
    }


async def _journal_agent_rejection(
    conn: asyncpg.Connection[Any],
    *,
    run: WfRun,
    call_key: str,
    kind: str,
    message: str,
) -> None:
    await wf_queries.append_run_event(
        conn,
        account_id=run.account_id,
        run_id=run.id,
        type="call_result",
        call_key=call_key,
        payload={"result": None, "is_error": True, "error": {"kind": kind, "message": message}},
    )


# Post-replay step disposition (#1548): a single discriminated kind replacing the
# old (owed_drive, needs_rewake) boolean pair. The arms are ordered by strength —
# `owed_drive` ⊇ `harvest_now` ⊇ `settled` — so the illegal "owed a drive but don't
# self-wake" state is unrepresentable: `"owed_drive"` is one token that BOTH selects
# the 'running' flip AND satisfies `!= "settled"` (self-wake).
#   - "settled"      → park 'suspended', no self-wake.
#   - "harvest_now"  → park 'suspended', self-wake (a freshly-opened gate/child already
#                      carries its signal/answer; harvest it next step, not next sweep).
#   - "owed_drive"   → hand the lease back 'running' AND self-wake (a same-wake call_result
#                      was journaled — budget read or catchable rejection — so the replay
#                      must re-drive to throw the AgentError at the await).
type StepDisposition = Literal["settled", "harvest_now", "owed_drive"]

_RANK: dict[StepDisposition, int] = {"settled": 0, "harvest_now": 1, "owed_drive": 2}


def _escalate(cur: StepDisposition, to: StepDisposition) -> StepDisposition:
    """Monotone join over the disposition arms: a frontier that needs only a rewake
    never downgrades one that owes a drive, and a later settled cap never clears an
    earlier escalation."""
    return to if _RANK[to] > _RANK[cur] else cur


def _memo_outcome(outcome: Outcome) -> dict[str, Any]:
    """Map an ``Outcome`` to the host memo's tagged shape: a value the driver
    fast-forwards into the await (``{"ok": value}``), or an error it throws there as
    ``AgentError`` (``{"error": {...}}``). The discriminated union lets the host tell
    "the agent answered" from "the agent failed" even when the answer is itself a
    dict — the real value always nests one level under ``"ok"``. The kind is the
    discriminator; no ``is_error`` re-derivation."""
    if outcome.kind == "ok":
        return {"ok": outcome.result}
    return {"error": outcome.error}


async def _resolve_agent_call(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    child_id: str,
    request_id: str,
    started_at: datetime,
    now: datetime,
    deadline: timedelta,
) -> Outcome | None:
    """The outcome of one inflight ``agent()`` call, or ``None`` if it is still
    pending and within its wall-clock deadline.

    ``derive_response`` returns the child's written response or a ``child_gone``
    outcome, else ``None`` (live and unanswered). A still-pending call past its
    deadline is force-resolved with a ``timeout`` error response (H3): a child that
    never goes idle — e.g. it loops on tools forever — never trips R4's quiescence
    nudge, so without this its caller would suspend forever. The write is
    exactly-once and we re-derive, so a real response that raced in wins; and a child
    archived/deleted in the derive→write window surfaces as ``child_gone`` on the
    re-derive (``write_response_if_absent`` raises ``NotFoundError`` against the gone
    row — the same graceful resolution the non-timeout harvest path gives a gone
    child, never a crash)."""
    resolved = await db_queries.derive_response(
        conn, child_id, account_id=account_id, request_id=request_id
    )
    if resolved is not None:
        return resolved
    if now - started_at < deadline:
        return None  # pending, within deadline
    with contextlib.suppress(NotFoundError):
        async with conn.transaction():
            wrote = await db_queries.write_response_if_absent(
                conn,
                child_id,
                account_id=account_id,
                request_id=request_id,
                outcome=Err(error={"kind": "timeout"}),
            )
            if wrote and get_settings().cancel_cascade_enabled:
                await db_queries.insert_session_cancel_marker(
                    conn, session_id=child_id, request_id=request_id, account_id=account_id
                )
    resolved = await db_queries.derive_response(
        conn, child_id, account_id=account_id, request_id=request_id
    )
    assert resolved is not None  # a response now exists (timeout/child) or child_gone
    return resolved


async def _enrich_agent_result(
    conn: asyncpg.Connection[Any],
    payload: dict[str, Any],
    *,
    account_id: str,
    child_id: str,
    started_at: datetime,
    now: datetime,
) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT input_tokens, output_tokens, cache_read_input_tokens,
               cache_creation_input_tokens, cost_microusd
          FROM sessions
         WHERE id = $1 AND account_id = $2
        """,
        child_id,
        account_id,
    )
    if row is None:
        return {
            **payload,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cost_usd": 0.0,
            },
            "duration_ms": max(0, int((now - started_at).total_seconds() * 1000)),
            "tool_calls": 0,
        }
    tool_calls = await conn.fetchval(
        "SELECT count(*) FROM events WHERE session_id = $1 AND kind = 'message' AND data->>'role' = 'tool'",
        child_id,
    )
    return {
        **payload,
        "usage": {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cache_read_input_tokens": row["cache_read_input_tokens"],
            "cache_creation_input_tokens": row["cache_creation_input_tokens"],
            "cost_usd": row["cost_microusd"] / 1_000_000,
        },
        "duration_ms": max(0, int((now - started_at).total_seconds() * 1000)),
        "tool_calls": int(tool_calls or 0),
    }


async def run_workflow_step(run_id: str) -> None:
    pool = runtime.require_pool()

    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        if run is None or run.status in TERMINAL_RUN_STATUSES:
            return  # vanished or already terminal — a stray/duplicate wake is a no-op
        account_id = run.account_id
    # Bind run/tenant onto structlog contextvars so every line this step emits is
    # attributable; cleared in the finally as defensive hygiene (contextvars are
    # task-scoped and procrastinate runs each job in a fresh asyncio task, but the
    # clear keeps the bindings from outliving the step regardless). Bound AFTER the
    # gone/terminal guard above — that path is an idempotent no-op with no account to
    # attribute. ``cause`` is omitted: workflows have no per-wake cause concept
    # (unlike a session step's message/reschedule/etc).
    bind_contextvars(run_id=run_id, account_id=account_id)
    try:
        await _run_workflow_step_body(pool, run_id, run, account_id)
    finally:
        clear_contextvars()


async def _run_workflow_step_body(
    pool: asyncpg.Pool[Any], run_id: str, run: WfRun, account_id: str
) -> None:
    async with pool.acquire() as conn:
        # ``running`` is the step's LEASE (#780): flipped on EVERY wake before any
        # journal write — not just the first — so a crash anywhere mid-step
        # (including the deliberate spawn-failed re-raise below) leaves a state the
        # needs-step sweep matches unconditionally. Without this, a crash after the
        # harvest journals its last call_result but before the re-drive parks/ends
        # the run would leave a 'suspended' row no filter clause ever wakes again.
        # The step's closing write — park to 'suspended' or a terminal — hands the
        # lease back. (``run.status`` is the entry snapshot, read just above.)
        if run.status != "running":
            await wf_queries.set_run_status(conn, run_id, "running", account_id=account_id)

        events = await wf_queries.list_run_events(conn, run_id)
        signals = {s.call_key: s for s in await wf_queries.list_run_signals(conn, run_id)}

        # User-requested cancel (signal-driven): the cancel API records a ``cancel``
        # side-marker + wakes the run, and we finalize it here — under the
        # procrastinate lock, so the journal keeps its single writer (appending a
        # terminal event from the request handler would race the gapless-seq
        # allocation). Checked before replay so a pending/suspended/running run is
        # stopped without driving the script. A cancel that lost the race to a
        # natural terminal already returned at the ``TERMINAL_RUN_STATUSES`` guard above.
        if (cancel_sig := signals.get(wf_queries.CANCEL_SIGNAL_CALL_KEY)) is not None:
            await _cancel_run(conn, run, reason=cancel_sig.result)
            return

        if run.host_semantics_epoch != HOST_SEMANTICS_EPOCH:
            await _complete_run(
                conn,
                run,
                output=(
                    "engine semantics changed: this run was created under "
                    f"host-semantics epoch {run.host_semantics_epoch} but the worker now "
                    f"runs epoch {HOST_SEMANTICS_EPOCH}; relaunch the workflow to run it "
                    "under the current engine"
                ),
                is_error=True,
                error_kind="engine_semantics_changed",
            )
            return

        memo: dict[str, Any] = {
            e.call_key: _memo_outcome(db_queries.outcome_from_jsonb(e.payload))
            for e in events
            if e.type == "call_result" and e.call_key is not None
        }
        # call_key -> the call_started event, so the harvest can read a gate's signal
        # or an agent child's response, and measure an agent call's age against its
        # wall-clock deadline (off the event's created_at).
        inflight: dict[str, WfRunEvent] = {
            e.call_key: e
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

        # Pre-replay harvest: resolve any inflight capability that is now done — a
        # gate with a delivered resume signal, or an agent child with a response (or a
        # past-deadline timeout) — and fold its result into memo so replay
        # fast-forwards past it.
        harvested = False  # any call_result journaled this step
        now = datetime.now(UTC)
        agent_deadline = timedelta(seconds=get_settings().workflow_agent_deadline_seconds)
        for call_key, cap_event in list(inflight.items()):
            cap_payload = cap_event.payload
            if cap_payload.get("capability") == "agent":
                # The child was invoked with request_id = call_key (the agent() call
                # IS the request), so its outcome is derived under that same id.
                call_outcome = await _resolve_agent_call(
                    conn,
                    account_id=account_id,
                    child_id=cap_payload["child_session_id"],
                    request_id=call_key,
                    started_at=cap_event.created_at,
                    now=now,
                    deadline=agent_deadline,
                )
                if call_outcome is None:
                    continue  # still pending, within deadline — stay suspended
                # Serialize the kind to the flat call_result JSONB once (the journal
                # shape is unchanged), then merge the agent-only usage enrichment.
                result_payload = await _enrich_agent_result(
                    conn,
                    db_queries.outcome_to_jsonb(call_outcome),
                    account_id=account_id,
                    child_id=cap_payload["child_session_id"],
                    started_at=cap_event.created_at,
                    now=now,
                )
            elif cap_payload.get("capability") == "invoke_workflow":
                # The sub-run was spawned with request_id = call_key (the
                # invoke_workflow() call IS the request), so its terminal outcome
                # is resolved under that same id through the SAME single resolver
                # the agent arm uses — only its run-kind branch (#1126). No live
                # await: a replay re-drives this fold from the journaled
                # call_result (memo already carries the outcome; this arm is not
                # reached), and a fresh wake resolves it from the durable log.
                call_outcome = await wf_queries.derive_run_response(
                    conn,
                    cap_payload["child_run_id"],
                    account_id=account_id,
                )
                if call_outcome is None:
                    continue  # sub-run still in-flight and unanswered — stay suspended
                result_payload = db_queries.outcome_to_jsonb(call_outcome)
            elif cap_payload.get("capability") == "tool":
                sig = signals.get(call_key)
                if sig is None and not run_tools.has_inflight(run_id, call_key):
                    # No signal in the snapshot AND no live task — but the snapshot can be
                    # stale: a task running outside the run lock commits its signal *before*
                    # it leaves the registry, so "not in-flight" may mean "just finished" as
                    # easily as "crashed". A fresh point-read disambiguates — so a tool that
                    # already ran is never re-dispatched (which would double-execute a
                    # non-idempotent http_request).
                    sig = await wf_queries.read_run_signal(conn, run_id, call_key)
                    if sig is None:
                        # Cold re-dispatch routes by the tool's execution class:
                        # a sandbox tool (bash) re-launches against the run's
                        # provisioned container, everything else on the worker.
                        # Both register in the SHARED run_tools._INFLIGHT, so the
                        # has_inflight guard above is class-agnostic.
                        if tool_executes_class(cap_payload["tool_name"]) == "sandbox":
                            run_sandbox.launch_sandbox_task(
                                pool,
                                run,
                                call_key=call_key,
                                tool_name=cap_payload["tool_name"],
                                tool_input=cap_payload.get("input"),
                            )
                        else:
                            run_tools.launch_tool_task(
                                pool,
                                run,
                                call_key=call_key,
                                tool_name=cap_payload["tool_name"],
                                tool_input=cap_payload.get("input"),
                            )
                        continue
                if sig is None:
                    continue  # a live task is still running — stay suspended, it will signal
                result_payload = {"result": sig.result, "is_error": False}
            elif cap_payload.get("capability") == "call_llm":
                # call_llm mirrors the tool park-and-harvest: its worker task writes a
                # ``tool_result`` signal (and charges the run's inference meter in the same
                # txn). Cold re-dispatch (no signal, no live task, no crashed-but-committed
                # signal) re-launches the inference — at-least-once, a second billable call.
                sig = signals.get(call_key)
                if sig is None and not run_llm.has_inflight(run_id, call_key):
                    sig = await wf_queries.read_run_signal(conn, run_id, call_key)
                    if sig is None:
                        run_llm.launch_call_llm_task(
                            pool,
                            run,
                            call_key=call_key,
                            spec=cap_payload.get("spec") or {},
                        )
                        continue
                if sig is None:
                    continue  # a live task is still running — stay suspended, it will signal
                result_payload = {"result": sig.result, "is_error": False}
            else:  # gate: the resume value lives in the signal
                sig = signals.get(call_key)
                if sig is None:
                    continue
                result_payload = {"result": sig.result, "is_error": False}
            await wf_queries.append_run_event(
                conn,
                account_id=account_id,
                run_id=run_id,
                type="call_result",
                call_key=call_key,
                payload=result_payload,
            )
            # memo carries the host's tagged outcome (R3): an `{ok}` value to
            # fast-forward into the await, or an `{error}` to throw as AgentError.
            memo[call_key] = _memo_outcome(db_queries.outcome_from_jsonb(result_payload))
            del inflight[call_key]
            harvested = True

        # QUIET WAKE early-exit (#780): the run entered parked ('suspended' means
        # the previous step's closing park committed, so the journal is a complete
        # suspension) and the harvest resolved nothing — by determinism the replay
        # would re-emit the same frontier and re-park, so the O(memo) reship +
        # replay is a provable no-op. Park back and return. This is what keeps a
        # sweep wake of a heavy parked run O(1): without it, a replay that outlives
        # the sweep interval re-arms the next tick's wake forever (lease/sweep
        # resonance), and a slow tool's run pays a full replay per tick. A
        # 'pending' or 'running' entry MUST drive: first wake, or a crashed step
        # whose frontier may be only part-journaled.
        if run.status == "suspended" and not harvested:
            await wf_queries.set_run_status(conn, run_id, "suspended", account_id=account_id)
            return

    # Drive one wake in the credential-free subprocess (no DB conn held).
    outcome = await run_script_host(source=run.script, input=run.input, memo=memo)

    # Post-replay step disposition (#1548). A same-wake call_result journaled this wake
    # (budget read or catchable agent error) owes one more drive so the replay throws the
    # AgentError at the await — so we DON'T settle into a quiet 'suspended' (which the next
    # wake's quiet-wake guard would early-exit), we hand the lease back as 'running' and let
    # the guaranteed self-wake drive it. One-shot by construction: a rejected call lands in
    # memo, so its capability is skipped on replay (never re-journaled), and the driven wake
    # re-suspends/completes normally.
    disposition: StepDisposition = "settled"
    # (call_key, tool_name, input) for tool frontiers opened this wake — launched AFTER
    # the txn commits (below), so call_started is visible before a task can signal+wake.
    tools_to_launch: list[tuple[str, str, Any]] = []
    # Same shape + same post-commit launch discipline, for tool frontiers whose tool
    # runs in the run's sandbox (bash) rather than on the worker.
    sandboxes_to_launch: list[tuple[str, str, Any]] = []
    # (call_key, spec) for call_llm frontiers opened this wake — the worker-side raw
    # inference task, launched post-commit like the tool launchers above (#1633).
    call_llm_to_launch: list[tuple[str, dict[str, Any]]] = []
    async with pool.acquire() as conn:
        # Journal this wake's progress annotations (log()/phase()) FIRST — before the
        # raised/returned/suspended dispatch — so a line emitted just before a crash or
        # a return is durably captured, ordered ahead of this wake's call_started /
        # run_completed. The run is still 'running' here (the lease flip above), so the
        # append's status guard admits them. Emit-once across replays is the memo's job:
        # a re-emitted annotation hits ON CONFLICT (run_id, call_key, type) — type
        # 'annotation' — and no-ops, preserving the original seq.
        for ann in outcome.annotations:
            await wf_queries.append_run_event(
                conn,
                account_id=account_id,
                run_id=run_id,
                type="annotation",
                call_key=ann.call_key,
                payload=ann.payload,
            )

        if outcome.kind == "raised":
            if outcome.error_kind == "script_host_spawn_failed":
                # Failure to *spawn* the host (EAGAIN/ENOMEM at fork) is a transient
                # worker-infra fault, not the script's — never terminally error the
                # run for it. Fail the step so the lock releases and the periodic
                # sweep re-wakes the (still non-terminal) run when capacity returns.
                raise RuntimeError(f"workflow {run_id}: {outcome.error_repr}")
            await _complete_run(
                conn,
                run,
                output=outcome.error_repr,
                is_error=True,
                error_kind=outcome.error_kind,
                error_traceback=outcome.error_traceback,
            )
            return

        # The host completed a real replay (suspended or returned), so every
        # still-open capability MUST have been re-reached. Replay-prefix assertion
        # (fail-closed): if one wasn't re-emitted, the script diverged
        # (nondeterminism) — error loudly rather than orphan the old call_started.
        # Gated on a non-crash outcome: a crashed/killed host emits nothing, and
        # that emptiness is a crash (handled above), never divergence.
        emitted_keys = {cap.call_key for cap in outcome.emitted}
        # A call_key with a `frontier_deferred` marker but no `call_started` (and not
        # yet resolved into memo) is a WAITING-to-be-admitted agent frontier. On a
        # completed replay the script MUST re-emit it (it has neither started nor
        # resolved); if it didn't, the run diverged just as surely as a vanished
        # inflight call — fail closed rather than strand a frontier that will never
        # be admitted. A key that later gets a `call_started` (admitted) is tracked
        # by `inflight` instead, and one that resolved is in `memo` — both excluded.
        # A key is no longer waiting iff it is open (`inflight`: has a `call_started`
        # without a `call_result`) or resolved (`memo`: has a `call_result` — either
        # a completed call, or a spawn REJECTION journaled directly as a catchable
        # error result with no `call_started` at all). Both sets together are exactly
        # the not-pending keys — no separate scan of `events` needed.
        started_keys = set(inflight) | set(memo)
        deferred_pending = {
            e.call_key
            for e in events
            if e.type == "frontier_deferred"
            and e.call_key is not None
            and e.call_key not in started_keys
        }
        diverged = sorted(
            {k for k in inflight if k not in emitted_keys}
            | {k for k in deferred_pending if k not in emitted_keys}
        )
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

        # Lifetime agent-call cap (H1): bound a runaway that spawns children in an
        # unbounded loop. Counted from the monotonic call_started journal (prior real
        # spawns) plus each child this step ACTUALLY creates — enforced per-spawn at the
        # gate (see ``_open_agent_capability``), the moment before a child is created, so
        # the over-cap child is never created. Per-spawn (not a batch pre-count) because a
        # rejected cap (agent_not_found / bad_agent_call) spawns no child and consumes no
        # slot, so it must NOT count toward the cap: a batch count over the frontier would
        # pessimistically tip the quota on caps that never spawn, terminating a run whose
        # only real over-cap pressure is illusory. (Good children spawned earlier in an
        # over-cap frontier do persist as orphans the quiescence sweep reclaims — same
        # terminal outcome, no correctness cost.) A harvest-only re-suspend (no new spawns)
        # thus naturally never errors — H1 gates NEW real spawns only, so lowering the cap
        # mid-flight can't retroactively kill a run whose children are already all inflight.
        # (parallel()'s width is bounded separately in the host.)
        prior_agent_calls = sum(
            1 for e in events if e.type == "call_started" and e.payload.get("capability") == "agent"
        )
        max_agent_calls = get_settings().workflow_max_agent_calls
        # The running tally of real agent children: prior journaled spawns + this step's.
        agent_spawns = prior_agent_calls

        # Per-run wave admission (#784): bound the number of concurrently in-flight
        # agent() children. Count this run's currently in-flight agents (harvested
        # `inflight` map), and admit at most `slots` NEW agent frontiers this wake;
        # journal the rest as `frontier_deferred` (no spawn, no call_started). Freed
        # slots admit deferred frontiers on the next wake — the existing child-
        # completion re-wake re-runs this step, no new machinery. Gate and tool
        # frontiers are UNTHROTTLED. Orthogonal to H1: the lifetime cap is enforced
        # per-spawn at the gate (above), and only ADMITTED frontiers reach the gate —
        # a deferred frontier consumes no lifetime quota until admitted (no spawn, no
        # call_started), and when admitted on a later wake it is counted exactly once,
        # so the wave gate neither masks nor double-counts the lifetime cap.
        inflight_agents = sum(
            1 for e in inflight.values() if e.payload.get("capability") == "agent"
        )
        slots = max(0, get_settings().workflow_max_inflight_children_per_run - inflight_agents)
        budget_usage = (
            await wf_queries.run_children_usage(conn, run_id, account_id=account_id)
            if run.budget_usd is not None
            else None
        )
        # The run's OWN call_llm inference spend (#1633): raw inference runs on the worker
        # at the run's own inference site, so it has no child-session row — it lives in the
        # run-level meter. The budget gate is the SUM: child-session rollup + this meter.
        call_llm_spent_microusd = (
            await wf_queries.get_run_call_llm_cost_microusd(conn, run_id, account_id=account_id)
            if run.budget_usd is not None
            else 0
        )
        budget_total_microusd = (
            round(run.budget_usd * 1_000_000) if run.budget_usd is not None else None
        )
        budget_spent_microusd = (
            budget_usage.cost_microusd + call_llm_spent_microusd if budget_usage is not None else 0
        )
        over_budget = (
            budget_usage is not None
            and budget_total_microusd is not None
            and budget_spent_microusd >= budget_total_microusd
        )

        # Suspended: open any *new* frontier capability, then park.
        for cap in outcome.emitted:
            if cap.call_key in memo or cap.call_key in inflight:
                continue  # already resolved, or already open (an idempotent re-emit)
            if cap.capability_id == "gate":
                nonce = secrets.token_urlsafe(32)
                # Persist the gate's structured spec (e.g. the dev_pipeline's
                # kind/tier/reason/sha) into the journal alongside the nonce, mirroring
                # the agent arm's persistence of its structured annotations. This is
                # append-only enrichment: the nonce and existing resume path are
                # untouched, and journal readers (e.g. the ops-agent gate-PREMISE
                # auditor) can re-derive a gate's premise from its recorded spec instead
                # of being blind to it. (aios#1660)
                await wf_queries.append_run_event(
                    conn,
                    account_id=account_id,
                    run_id=run_id,
                    type="call_started",
                    call_key=cap.call_key,
                    payload={"capability": "gate", "gate_nonce": nonce, "spec": cap.spec},
                )
                if run.launcher_session_id is not None:
                    await write_gate_opened(
                        conn,
                        run.launcher_session_id,
                        account_id=account_id,
                        request_id=cap.call_key,
                        run_id=run_id,
                        gate_nonce=nonce,
                    )
                # A resume signal can land before this call_started is journaled
                # (the call_key is derivable); the pre-replay harvest only sees
                # gates already inflight, so a self-wake harvests this freshly
                # opened gate promptly instead of waiting for the periodic sweep.
                if cap.call_key in signals:
                    disposition = _escalate(disposition, "harvest_now")
            elif cap.capability_id == "agent":
                if over_budget:
                    assert budget_usage is not None and run.budget_usd is not None
                    await _journal_agent_rejection(
                        conn,
                        run=run,
                        call_key=cap.call_key,
                        kind="budget_exceeded",
                        message=(
                            f"run budget exhausted: spent ${budget_usage.cost_microusd / 1_000_000:.2f} "
                            f"of ${run.budget_usd:.2f} — new agent() calls are refused"
                        ),
                    )
                    disposition = _escalate(disposition, "owed_drive")
                    continue
                if slots > 0:
                    spawn = await _open_agent_capability(
                        conn,
                        pool,
                        run,
                        cap,
                        agent_spawns=agent_spawns,
                        max_agent_calls=max_agent_calls,
                    )
                    # Lifetime cap hit (H1): this cap passed every rejection gate and WOULD
                    # create a real child, but that child would exceed the cap — terminate the
                    # run (terminal, not catchable). Checked at the spawn point, before the
                    # child exists, so no orphan is created; any good children already spawned
                    # earlier in this frontier share the run's terminal outcome.
                    if spawn.quota_exceeded:
                        await _complete_run(
                            conn,
                            run,
                            output=(
                                f"workflow exceeded the {max_agent_calls}-agent call cap "
                                f"({agent_spawns + 1} total agent calls attempted)"
                            ),
                            is_error=True,
                            error_kind="too_many_agents",
                        )
                        return
                    # rejected: this call's catchable error was journaled — self-wake so the
                    # run replays and throws the AgentError at the await. A spawn error on ONE
                    # capability must NOT abort spawning the others, so continue the loop.
                    # needs_rewake (C1'/C4): a re-attached child already has its marker.
                    if spawn.rejected or spawn.needs_rewake:
                        disposition = _escalate(disposition, "harvest_now")
                    if spawn.rejected:
                        disposition = _escalate(disposition, "owed_drive")
                    else:
                        # A real child was created (or re-attached): it counts toward the
                        # lifetime tally AND occupies a wave slot. A REJECTED cap consumes
                        # neither — no child exists, so it neither tips the lifetime quota
                        # nor blocks admission of the remaining frontier.
                        agent_spawns += 1
                        slots -= 1
                else:
                    # Over the per-run wave cap: defer this frontier. Journal a
                    # `frontier_deferred` marker (idempotent on (run_id, call_key,
                    # type)) so the divergence guard sees a waiting agent. Do NOT
                    # spawn and do NOT journal call_started. A later wake re-emits this
                    # same frontier (no call_started ⇒ still "new"); when a child
                    # resolves and frees a slot, it is admitted then. Deferral is a
                    # WAIT, never an error — it must never reach _open_agent_capability,
                    # so it can never route through the catchable spawn-failure path
                    # (#779): rejection means an ATTEMPTED spawn failed; deferral means
                    # the spawn was POSTPONED, invisible to the script. No stall: slots
                    # can only be exhausted while ≥1 real child is in flight, and that
                    # child's completion re-wake re-drives admission.
                    await wf_queries.append_run_event(
                        conn,
                        account_id=account_id,
                        run_id=run_id,
                        type="frontier_deferred",
                        call_key=cap.call_key,
                        payload={"capability": "agent"},
                    )
            elif cap.capability_id == "invoke_workflow":
                # Run-caller surface (#1129): spawn a sub-run, journal call_started.
                # Modelled on the agent arm minus the wave/lifetime caps (those are
                # agent-call specific); the run→run depth + fan-out caps live in
                # create_run and apply for free. A bad output_schema / target
                # rejects as a CATCHABLE author error (call_result error journaled),
                # so self-wake to replay and throw the AgentError at the await.
                spawn = await _open_invoke_workflow_capability(conn, pool, run, cap)
                if spawn.rejected:
                    disposition = _escalate(disposition, "owed_drive")
                elif spawn.needs_rewake:
                    disposition = _escalate(disposition, "harvest_now")
            elif cap.capability_id == "budget":
                usage = (
                    await wf_queries.run_children_usage(conn, run_id, account_id=account_id)
                    if run.budget_usd is not None
                    else wf_queries.RunChildrenUsage(0, 0, 0, 0, 0)
                )
                # Include the run's own call_llm inference meter (#1633) so the author's
                # budget() view reports the same spend the over-budget gate enforces.
                meter = (
                    await wf_queries.get_run_call_llm_cost_microusd(
                        conn, run_id, account_id=account_id
                    )
                    if run.budget_usd is not None
                    else 0
                )
                await wf_queries.append_run_event(
                    conn,
                    account_id=account_id,
                    run_id=run_id,
                    type="call_result",
                    call_key=cap.call_key,
                    payload={"result": _budget_view(run, usage, meter), "is_error": False},
                )
                disposition = _escalate(disposition, "owed_drive")
            elif cap.capability_id == "tool":
                spec = cap.spec if isinstance(cap.spec, dict) else {}
                tool_name = spec.get("tool_name")
                if not isinstance(tool_name, str):
                    # A malformed spec is a deterministic author bug (replay-identical),
                    # so it terminally errors the run — unlike an undeclared/unknown tool
                    # name, which the dispatcher returns as a recoverable error value.
                    await _complete_run(
                        conn,
                        run,
                        output=f"tool() requires a string tool name, got {tool_name!r}",
                        is_error=True,
                        error_kind="bad_tool_call",
                    )
                    return
                await wf_queries.append_run_event(
                    conn,
                    account_id=account_id,
                    run_id=run_id,
                    type="call_started",
                    call_key=cap.call_key,
                    payload={
                        "capability": "tool",
                        "tool_name": tool_name,
                        "input": spec.get("input"),
                    },
                )
                # Route by execution class: a sandbox tool (bash) launches against
                # the run's provisioned container, everything else on the worker.
                # The journaled payload is identical (bash rides the `tool`
                # capability); only the launcher differs.
                if tool_executes_class(tool_name) == "sandbox":
                    sandboxes_to_launch.append((cap.call_key, tool_name, spec.get("input")))
                else:
                    tools_to_launch.append((cap.call_key, tool_name, spec.get("input")))
            elif cap.capability_id == "call_llm":
                # Raw inference (#1633). Charges the run's call_llm meter, which the
                # budget gate reads — so an exhausted budget refuses it here. Unlike
                # agent()'s budget refusal (a catchable AgentError), call_llm errors are
                # VALUES: the refusal is journaled as the call's recoverable {"error": …}
                # result, which the script branches on (matching call_llm's "errors
                # resolve" contract). The call_started/result pair fully resolves it —
                # no worker task launches, so no inference (and no spend) occurs.
                llm_spec = cap.spec if isinstance(cap.spec, dict) else {}
                if over_budget:
                    assert run.budget_usd is not None
                    await wf_queries.append_run_event(
                        conn,
                        account_id=account_id,
                        run_id=run_id,
                        type="call_started",
                        call_key=cap.call_key,
                        payload={"capability": "call_llm", "spec": llm_spec},
                    )
                    await wf_queries.append_run_event(
                        conn,
                        account_id=account_id,
                        run_id=run_id,
                        type="call_result",
                        call_key=cap.call_key,
                        payload={
                            "result": {
                                "error": (
                                    f"run budget exhausted: spent "
                                    f"${budget_spent_microusd / 1_000_000:.2f} of "
                                    f"${run.budget_usd:.2f} — call_llm is refused"
                                )
                            },
                            "is_error": False,
                        },
                    )
                    disposition = _escalate(disposition, "owed_drive")
                    continue
                await wf_queries.append_run_event(
                    conn,
                    account_id=account_id,
                    run_id=run_id,
                    type="call_started",
                    call_key=cap.call_key,
                    payload={"capability": "call_llm", "spec": llm_spec},
                )
                call_llm_to_launch.append((cap.call_key, llm_spec))
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
        # Hand the lease back. A reject journaled this wake leaves the run 'running' (a
        # drive is owed: the self-wake replays and throws the catchable AgentError);
        # otherwise park as a settled 'suspended'.
        await wf_queries.set_run_status(
            conn,
            run_id,
            "running" if disposition == "owed_drive" else "suspended",
            account_id=account_id,
        )

    # Outside the txn (after the suspend commits): launch this wake's tool tasks — now
    # that their call_started rows are committed, a task that signals+wakes lands a step
    # that sees them (no double-dispatch) — then self-wake so the next step harvests an
    # already-delivered gate resume opened this wake.
    for launch_key, launch_name, launch_input in tools_to_launch:
        run_tools.launch_tool_task(
            pool, run, call_key=launch_key, tool_name=launch_name, tool_input=launch_input
        )
    for launch_key, launch_name, launch_input in sandboxes_to_launch:
        run_sandbox.launch_sandbox_task(
            pool, run, call_key=launch_key, tool_name=launch_name, tool_input=launch_input
        )
    for launch_key, launch_spec in call_llm_to_launch:
        run_llm.launch_call_llm_task(pool, run, call_key=launch_key, spec=launch_spec)
    if disposition != "settled":
        await defer_run_wake(run_id)


class _SpawnResult(NamedTuple):
    """Outcome of opening one ``agent()`` frontier."""

    # this call's catchable error was journaled — self-wake so the run replays and throws
    rejected: bool
    needs_rewake: bool  # a re-attached child already has its marker — self-wake to harvest
    # the cap passed every rejection gate but creating its child would exceed the lifetime
    # agent-call cap — the caller terminates the run (``too_many_agents``); no child created
    quota_exceeded: bool = False


async def _reject_invalid_output_schema(
    output_schema: Any,
    *,
    call_name: str,
    reject_kind: str,
    reject: Callable[[str, str], Awaitable[_SpawnResult]],
) -> _SpawnResult | None:
    """The shared author-facing output_schema validity gate for the agent() and
    invoke_workflow() spawn arms — the single definition of "is this a usable
    structured-output schema". Returns a rejecting _SpawnResult (already journaled
    via ``reject``) on the first failing branch, or None if the schema is usable.

    Three sequentially-dependent author-facing failures, each an early reject:
    a non-object schema (a bare boolean is valid JSON Schema, but ``false``
    rejects every value and ``true`` disables enforcement), a structurally-invalid
    schema, and one with an unresolvable reference (passes check_schema but raises
    at validation time, bricking the child).
    """
    if not isinstance(output_schema, dict):
        return await reject(
            reject_kind,
            f"{call_name} output_schema must be a JSON object schema, "
            f"got {type(output_schema).__name__}",
        )
    try:
        jsonschema.Draft202012Validator.check_schema(output_schema)
    except jsonschema.SchemaError as exc:
        return await reject(
            reject_kind, f"{call_name} output_schema is not a valid JSON Schema: {exc.message}"
        )
    if (bad_ref := _unresolvable_ref(output_schema)) is not None:
        return await reject(
            reject_kind,
            f"{call_name} output_schema has an unresolvable reference {bad_ref!r} "
            "(references must resolve within the schema; remote refs are unsupported)",
        )
    return None


async def _open_agent_capability(
    conn: asyncpg.Connection[Any],
    pool: asyncpg.Pool[Any],
    run: WfRun,
    cap: EmittedCapability,
    *,
    agent_spawns: int,
    max_agent_calls: int,
) -> _SpawnResult:
    """Spawn (or, on replay/C1', re-attach) the ``agent()`` child for a frontier,
    then journal ``call_started{child_session_id, child_agent_version}``.

    Idempotent + crash-safe: the child id is deterministic, ``create_child_session``
    is ``ON CONFLICT`` (delivers the input atomically with the row on first spawn),
    ``call_started`` dedups on the memo. On replay the child id already exists →
    re-attach, journaling the *row's* pinned version (not a re-resolved one).

    Returns ``rejected=True`` iff this call was rejected and its catchable error was
    journaled (a ``call_result`` error for ``cap.call_key``): the caller self-wakes so
    the run replays and ``_agent_error_from`` throws an ``AgentError`` at the
    ``await``, where the author can catch it. ``defer_wake`` fires **only on
    first spawn** (``created``): on a re-attach the child is already sweep-wakeable
    via its own first user message and may even be terminal, so appending a wake
    span to it would crash. ``needs_rewake=True`` asks the caller for a self-wake
    when a re-attached child *already* has its completion marker (C1'/C4: it
    finished before this wake journaled ``call_started``, so the pre-replay harvest
    — which only sees inflight ``call_started`` events — missed it).

    The lifetime agent-call cap (H1) is enforced here, at the spawn point: ``agent_spawns``
    is the count of real children already created (prior journaled + this step's so far).
    A cap is checked AFTER its rejection gates and BEFORE ``create_child_session`` — a
    rejected cap creates no child and never counts, and a cap that would create the
    (``max_agent_calls`` + 1)-th child returns ``quota_exceeded=True`` so the caller
    terminates the run before that child exists.
    """
    account_id = run.account_id

    async def _reject(kind: str, message: str) -> _SpawnResult:
        # Journal the rejection as a CATCHABLE call_result error for this call_key.
        await _journal_agent_rejection(
            conn, run=run, call_key=cap.call_key, kind=kind, message=message
        )
        return _SpawnResult(rejected=True, needs_rewake=False)

    spec = cap.spec if isinstance(cap.spec, dict) else {}
    # agent() carries output_schema as a canonical JSON *string* (so a schema's floats
    # survive the call_key hash); reconstruct the dict. None means no schema demanded.
    output_schema_raw = spec.get("output_schema")
    output_schema = (
        json.loads(output_schema_raw) if isinstance(output_schema_raw, str) else output_schema_raw
    )
    if (
        output_schema is not None
        and (
            rejected := await _reject_invalid_output_schema(
                output_schema,
                call_name="agent()",
                reject_kind="bad_agent_call",
                reject=_reject,
            )
        )
        is not None
    ):
        return rejected
    agent_id = spec.get("agent_id")
    if agent_id is not None and not isinstance(agent_id, str):
        return await _reject(
            "bad_agent_call", f"agent() requires agent_id to be a string or None, got {agent_id!r}"
        )
    model = spec.get("model")
    if model is not None and not isinstance(model, str):
        return await _reject("bad_agent_call", f"agent() model must be a string, got {model!r}")

    child_id = child_session_id(run.id, cap.call_key)
    pinned: int | None
    stamped_model: str | None = model
    # #823: the child's model identity (litellm_extra, api_base foremost) — the SECOND
    # authority axis. A generic agentless child carries none (the run's model on the
    # default endpoint); a named child inherits the agent's, clamped + frozen below.
    child_litellm_extra: dict[str, Any] = {}
    if agent_id is None:
        pinned = None
        stamped_model = model or run.default_child_model
        if stamped_model is None:
            return await _reject(
                "bad_agent_call",
                "generic agent() call has no model: pass model= or set the run's default child model",
            )
        child_surface = surface_of(run)
    else:
        try:
            # The full agent (head row) — both the pinned version AND its declared surface,
            # which the run clamps. One query, same NotFoundError contract as before.
            agent = await db_queries.get_agent(conn, agent_id, account_id=account_id)
        except NotFoundError:
            return await _reject("agent_not_found", f"agent {agent_id!r} not found")
        pinned = agent.version
        # #794: named children wield agent ∩ run, frozen at spawn; vaults are the run's.
        child_surface = attenuation_service.clamp(surface_of(agent), surface_of(run))
        # #823: model-identity clamp at the spawn edge. The agent's litellm_extra can
        # carry api_base, which redirects the child's model call — sending its entire
        # prompt context to that endpoint on the first inference, no tool call required.
        # The surface meet (#794) does nothing about this orthogonal axis. Admit the
        # child's effective api_base iff it equals the launcher's (a workflow run has
        # none → the default operator endpoint) OR sits in the operator trusted-endpoint
        # allowlist; else FAIL CLOSED with a catchable rejection, before any child row
        # exists. Both clamps are needed and independent: the create-time
        # surface-attenuation clamp (#1470, services.agents._enforce_surface_attenuation)
        # bounds what a self-authoring agent may WRITE (declared ⊆ creator), while this
        # spawn-edge re-clamp bounds what a given RUN may wield (agent ∩ run) and freezes
        # model identity. With the create-time clamp in place this edge stays sound even
        # with native self-management — it no longer rests on the old "create_agent is
        # operator-only" precondition.
        child_litellm_extra = agent.litellm_extra or {}
        if not attenuation_service.model_identity_trusted(child_litellm_extra, None):
            redirect = api_base_of(child_litellm_extra)
            return await _reject(
                "untrusted_api_base",
                f"agent {agent_id!r} routes model calls to an untrusted inference "
                f"endpoint ({redirect!r}); add it to the operator "
                f"trusted_inference_api_bases allowlist to permit this spawn",
            )
    # #1636: the ``workflow:`` model-binding privilege at the spawn-edge dispatch seam,
    # keyed on the RUN's owning principal — operator iff the run is operator/HTTP-launched
    # (no launcher session). Covers BOTH unnamed paths: the per-call ``agent(model=…)``
    # override and the generic agentless child's resolved model (``stamped_model``); a
    # NAMED child additionally inherits the agent's stored ``model`` when no override is
    # given, so that string is checked too. A self-authoring run may neither select nor
    # bind a ``workflow:`` model — a catchable rejection, before any child row exists.
    # Orthogonal to the #823 api_base clamp above (that bounds *where* inference routes;
    # this bounds *whether* it may route through a workflow at all).
    is_operator_run = run.launcher_session_id is None
    selected_model = stamped_model if agent_id is None else (model or agent.model)
    if not is_operator_run and is_workflow_binding(selected_model):
        return await _reject(
            "workflow_model_forbidden",
            f"selecting a workflow: model ({selected_model!r}) is operator-only; this "
            "self-authoring run may not route a child's inference through a workflow",
        )
    # Lifetime cap (H1) enforced HERE — past every rejection gate, before the child is
    # created — so only caps that genuinely spawn count. ``agent_spawns`` already excludes
    # rejected caps; this child would be the (agent_spawns + 1)-th, so cap it on strict ``>``.
    if agent_spawns + 1 > max_agent_calls:
        return _SpawnResult(rejected=False, needs_rewake=False, quota_exceeded=True)
    # #1124: the run→session ``agent()`` edge spends one unit of the run's
    # DOWN-counting trusted depth budget. Refuse BEFORE the child is written when
    # the run has none left — a journaled, author-catchable rejection (raising here
    # would crash the deterministic step), so no over-budget child row/edge exists.
    # The child's edge carries ``run.depth - 1``; the decrement IS the cycle bound.
    if run.depth <= 0:
        # Late import: ``aios.workflows.service`` transitively pulls in the tools
        # package, a cycle when step.py is on the import path.
        from aios.workflows.service import INVOKE_MAX_DEPTH

        return await _reject(
            "invoke_depth_exceeded",
            f"agent() would exceed the trusted invoke-depth budget ({INVOKE_MAX_DEPTH})",
        )
    run_vaults = await wf_queries.get_run_vault_ids(conn, run.id, account_id=account_id)

    created = await create_child_session(
        pool,
        AskNewSession(
            session_id=child_id,
            agent_id=agent_id,
            environment_id=run.environment_id,
            agent_version=pinned,
            model=stamped_model,
            parent_run_id=run.id,
            surface=child_surface,
            vault_ids=run_vaults,
            request_id=cap.call_key,  # the agent() call IS the request the child must answer
            input=spec.get("input"),
            output_schema=output_schema,
            depth=run.depth - 1,
            litellm_extra=child_litellm_extra,  # #823: frozen, clamped model identity
        ),
        account_id=account_id,
    )
    # On replay the row already carries its first-spawn version — journal THAT, so
    # call_started.child_agent_version always matches the version the child runs under.
    child_version: int | None
    needs_rewake = False
    if created:
        child_version = pinned
    else:
        child = await db_queries.get_session_bare(conn, child_id, account_id=account_id)
        child_version = child.agent_version
        # C1'/C4: the child reached a terminal outcome before we journaled
        # call_started, so the pre-replay harvest (which only scans inflight
        # call_started events) could not have seen it. Ask for a self-wake so the
        # next step harvests it now, rather than waiting one periodic sweep tick.
        # Routed through the same derive_response seam as the harvest, so an already-
        # gone child (not just an answered one) triggers the prompt re-wake too.
        needs_rewake = (
            await db_queries.derive_response(
                conn, child_id, account_id=account_id, request_id=cap.call_key
            )
            is not None
        )
    cap_payload: dict[str, Any] = {
        "capability": "agent",
        "child_session_id": child_id,
        "child_agent_version": child_version,
    }
    if "label" in cap.annotations:
        cap_payload["label"] = cap.annotations["label"]
    if output_schema is not None:
        cap_payload["output_schema"] = output_schema  # audit: the shape this call demanded
    await wf_queries.append_run_event(
        conn,
        account_id=account_id,
        run_id=run.id,
        type="call_started",
        call_key=cap.call_key,
        payload=cap_payload,
    )
    # Prompt wake of the child ONLY on first spawn. On a re-attach the child is
    # already sweep-wakeable via its own first user message (delivered with the
    # row) and may already be terminal — appending a wake span to an archived
    # child would crash on append_event's archived guard.
    if created:
        await defer_wake(pool, child_id, account_id=account_id, cause="workflow_child_spawn")
    return _SpawnResult(rejected=False, needs_rewake=needs_rewake)


async def _open_invoke_workflow_capability(
    conn: asyncpg.Connection[Any],
    pool: asyncpg.Pool[Any],
    run: WfRun,
    cap: EmittedCapability,
) -> _SpawnResult:
    """Open one ``invoke_workflow()`` frontier: spawn a sub-run, journal ``call_started``.

    The run dual of :func:`_open_agent_capability` — same shape (validate →
    create-or-reattach the servicer under a DETERMINISTIC id → journal
    ``call_started`` carrying the servicer id), keyed by ``workflow_id`` instead of
    ``agent_id``. Differences from the agent arm:

    * The servicer is a sub-RUN, created by ``create_run`` (which applies the #794
      surface clamp, the run→run depth cap, and the per-account fan-out cap for
      free), spawned with ``request_id = cap.call_key`` (the call IS the request),
      ``caller = {kind:'run', id:run.id}``, and the request's ``output_schema``.
    * There is no lifetime "agent-call" cap (that is agent-call-specific); a
      doomed-by-quota sub-run is refused by ``create_run`` and surfaces as a
      catchable author error, not a run-terminating ``too_many_agents``.

    A bad ``output_schema`` / missing workflow / cap breach is journaled as a
    CATCHABLE ``call_result`` error (``rejected=True``) so the run self-wakes,
    replays, and throws ``AgentError`` at the ``await``. ``needs_rewake=True`` when
    a re-attached sub-run already carries its answer (it finished before this wake
    journaled ``call_started``, so the pre-replay harvest missed it).
    """
    account_id = run.account_id

    async def _reject(kind: str, message: str) -> _SpawnResult:
        await _journal_agent_rejection(
            conn, run=run, call_key=cap.call_key, kind=kind, message=message
        )
        return _SpawnResult(rejected=True, needs_rewake=False)

    spec = cap.spec if isinstance(cap.spec, dict) else {}
    workflow_id = spec.get("workflow_id")
    if not isinstance(workflow_id, str):
        return await _reject(
            "bad_invoke_workflow",
            f"invoke_workflow() requires workflow_id to be a string, got {workflow_id!r}",
        )
    # output_schema rides the wire as a canonical JSON *string* (mirror agent());
    # reconstruct the dict and apply the SAME author-facing validity gates.
    output_schema_raw = spec.get("output_schema")
    output_schema = (
        json.loads(output_schema_raw) if isinstance(output_schema_raw, str) else output_schema_raw
    )
    if (
        output_schema is not None
        and (
            rejected := await _reject_invalid_output_schema(
                output_schema,
                call_name="invoke_workflow()",
                reject_kind="bad_invoke_workflow",
                reject=_reject,
            )
        )
        is not None
    ):
        return rejected

    sub_run_id = child_run_id(run.id, cap.call_key)
    run_vaults = await wf_queries.get_run_vault_ids(conn, run.id, account_id=account_id)
    # Spawn (or idempotently re-attach) the sub-run. ``create_run`` owns its own
    # transaction on a separate pooled connection (like ``create_child_session``);
    # its create-or-reattach + caps are all internal. A 404 (workflow gone /
    # archived) or a cap breach (depth / fan-out) is a catchable author error.
    try:
        sub_run = await create_run(
            pool,
            account_id=account_id,
            workflow_id=workflow_id,
            environment_id=run.environment_id,
            input=spec.get("input"),
            vault_ids=run_vaults,
            run_id=sub_run_id,
            parent_run_id=run.id,
            # #1653: propagate the ORIGINATING principal down the ``parent_run_id``
            # lineage. Without this the sub-run's ``launcher_session_id`` defaults to
            # NULL, so ``is_operator_run`` (this module) and the launcher surface clamp
            # (service.py) both mis-read it as an edgeless operator/HTTP run — letting a
            # self-authoring agent (1) bind the operator-only ``workflow:`` model for a
            # grandchild and (2) run the sub-run un-attenuated on the tool axis. The
            # parent run already carries the originating principal: a NON-NULL session
            # for an agent- or trigger-launched chain (the sub-run inherits it and is
            # correctly non-operator), or NULL for a genuine operator/HTTP root (the
            # sub-run stays operator, like the parent). Inheriting it verbatim is the
            # whole fix — it reflects the originator at every depth of a nested chain.
            launcher_session_id=run.launcher_session_id,
            request_id=cap.call_key,  # the invoke_workflow() call IS the request
            caller={"kind": "run", "id": run.id, "awaited": True},
            request_output_schema=output_schema,
        )
    except NotFoundError:
        return await _reject("workflow_not_found", f"workflow {workflow_id!r} not found")
    except ConflictError as exc:
        return await _reject("bad_invoke_workflow", str(exc))
    except (WorkflowRunDepthExceededError, RateLimitedError, ForbiddenError) as exc:
        return await _reject("invoke_workflow_refused", str(exc))

    cap_payload: dict[str, Any] = {
        "capability": "invoke_workflow",
        "child_run_id": sub_run.id,
        "workflow_id": workflow_id,
    }
    if "label" in cap.annotations:
        cap_payload["label"] = cap.annotations["label"]
    if output_schema is not None:
        cap_payload["output_schema"] = output_schema  # audit: the shape this call demanded
    await wf_queries.append_run_event(
        conn,
        account_id=account_id,
        run_id=run.id,
        type="call_started",
        call_key=cap.call_key,
        payload=cap_payload,
    )
    # If the sub-run already carries its answer (re-attach to a finished/gone run),
    # ask the caller to self-wake so its next step harvests now rather than after a
    # sweep tick — resolved through the SAME ``derive_run_response`` seam the harvest
    # uses, so an already-gone sub-run triggers the prompt re-wake too.
    needs_rewake = (
        await wf_queries.derive_run_response(conn, sub_run.id, account_id=account_id) is not None
    )
    return _SpawnResult(rejected=False, needs_rewake=needs_rewake)


async def _complete_run(
    conn: asyncpg.Connection[Any],
    run: WfRun,
    *,
    output: Any,
    is_error: bool,
    error_kind: str | None = None,
    error_traceback: str | None = None,
) -> None:
    """Append ``run_completed`` + flip status (+ store output) atomically.

    Child reclaim (archiving the run's spawned children) is deliberately NOT done
    here: at run-completion the just-answered child is still finishing its own step
    (it wrote its response and woke us *before* appending its tool_result), so
    archiving it now would race that in-flight append into ``append_event``'s
    archived guard. Safe reclaim is quiescence-gated — a child archives only once it
    is genuinely idle — which is the deferred ``schedule-archival-on-quiescence``
    mechanism. Until then a finished child lingers idle (harmless; no sandbox until
    used). The run's correctness never depended on reclaim. The
    ``run_session_step`` archived-guard makes that future reclaim crash-free.
    """
    # Run-target error-arm (#1126 / #790 schema half): a run completing IN SERVICE
    # OF A REQUEST validates its terminal output against the REQUEST's per-call
    # output_schema and FAILS LOUD on mismatch (error_kind=output_schema_violation).
    # No bounce-and-retry — the script already ran (unlike an agent, which the
    # return-tool bounces). Only a *successful* output is schema-checked; an already-
    # errored completion passes through (the error already explains the outcome).
    if (
        not is_error
        and run.request_id is not None
        and run.request_output_schema is not None
        and (schema_error := _validate_output_against_schema(output, run.request_output_schema))
        is not None
    ):
        output = schema_error
        is_error = True
        error_kind = "output_schema_violation"

    usage = await wf_queries.run_children_usage(conn, run.id, account_id=run.account_id)
    payload: dict[str, Any] = {
        "output": output,
        "is_error": is_error,
        "usage": _usage_payload(usage),
        "duration_ms": max(0, int((datetime.now(UTC) - run.created_at).total_seconds() * 1000)),
    }
    if error_kind is not None:
        error: dict[str, Any] = {"kind": error_kind}
        if output is not None:
            error["message"] = str(output)
        if error_traceback:
            error["traceback"] = error_traceback
        payload["error"] = error
    await _commit_terminal_and_dispatch(
        conn,
        run,
        status="errored" if is_error else "completed",
        payload=payload,
        output=output,
    )


async def _cancel_run(conn: asyncpg.Connection[Any], run: WfRun, *, reason: Any = None) -> None:
    """Finalize a run as ``cancelled`` — the terminal path for a user cancel.

    Structurally :func:`_complete_run` for cancellation: a non-error
    ``run_completed`` bookend (``cancelled: True``, so a live ``/stream`` closes on
    the event) + a ``cancelled`` terminal status, atomically. Reached only from the
    pre-replay cancel harvest, so it runs under the lock as the journal's single
    writer. A cancel is a terminal completion too — watchers with ``cancelled`` in
    their statuses filter fire. Like a natural completion, child reclaim is left to
    the deferred quiescence sweep (see :func:`_complete_run`).
    """
    payload: dict[str, Any] = {"output": None, "is_error": False, "cancelled": True}
    if reason is not None:
        payload["reason"] = reason
    await _commit_terminal_and_dispatch(conn, run, status="cancelled", payload=payload, output=None)


async def _fail_child_requests_for_terminal_error(
    conn: asyncpg.Connection[Any], run: WfRun, *, error_kind: str
) -> list[str]:
    """Close each live child's open requests with ``error_kind`` AND seed the cancel
    marker those closures would otherwise orphan (event-path parity, spec §5).

    This runs BEFORE ``seed_outbound_cancel_conn``'s open-edge enumeration in the same
    terminal transaction, so the edges it closes are invisible to the C1 seeder — each
    child must get its cancel-marker HERE, in the SAME transaction, or the
    ``engine_semantics_changed`` / ``nondeterministic_replay`` cohort never runs its
    lifecycle leaf until the durable sweep (the backstop silently becoming the primary
    path). The marker is the wake carrier only; the child's own leaf classifies the
    already-written ``error_kind`` (∈ ``REVOCATION_KINDS``) under its own lock. Returns
    the marked session ids so the post-commit loop can prompt each child's leaf.
    """
    marked: list[str] = []
    rows = await conn.fetch(
        "SELECT id FROM sessions "
        "WHERE parent_run_id = $1 AND account_id = $2 AND archived_at IS NULL",
        run.id,
        run.account_id,
    )
    cascade_enabled = get_settings().cancel_cascade_enabled
    for row in rows:
        # Snapshot the open set BEFORE the closures — these are exactly the requests
        # ``fail_open_child_requests_conn`` answers (same conn, same transaction).
        open_ids = await db_queries.get_open_request_ids(conn, row["id"], account_id=run.account_id)
        await fail_open_child_requests_conn(
            conn,
            row["id"],
            account_id=run.account_id,
            error={"kind": error_kind},
        )
        if not cascade_enabled or not open_ids:
            continue
        for request_id in open_ids:
            await db_queries.insert_session_cancel_marker(
                conn, session_id=row["id"], request_id=request_id, account_id=run.account_id
            )
        marked.append(row["id"])
    return marked


async def _commit_terminal_and_dispatch(
    conn: asyncpg.Connection[Any],
    run: WfRun,
    *,
    status: WfRunStatus,
    payload: dict[str, Any],
    output: Any,
) -> None:
    """THE terminal chokepoint: ``run_completed`` + status flip + run_completion
    trigger dispatch (#819), one transaction; the fire defers post-commit.

    Exactly-once dispatch gate: the journal memo (``UNIQUE NULLS NOT DISTINCT
    (run_id, call_key, type)``) guarantees exactly one ``run_completed`` insert
    per run EVER commits — under procrastinate dual execution the loser's
    append returns ``None`` and dispatches nothing. The pending carrier rows
    commit atomically with the terminal transition ("the run completed" and
    "these fires are owed" are one fact). The post-commit defers are
    best-effort by design — they ride procrastinate's separate psycopg pool,
    so a loss (worker crash, broker blip) leaves durable ``pending`` carrier
    rows the periodic sweep re-defers; never a silently dropped event fire.
    """
    fires: list[db_queries.TriggerFireRef] = []
    cascade_children: list[db_queries.ChildNode] = []
    terminal_marked: list[str] = []
    # The CALLER run to answer, or None — a run servicing a RUN caller (single-sourced so
    # the durable signal-write and the prompt wake below can never desync on the gate).
    run_caller_id = (
        run.caller["id"]
        if (
            run.request_id is not None
            and isinstance(run.caller, dict)
            and run.caller.get("kind") == "run"
            and isinstance(run.caller.get("id"), str)
        )
        else None
    )
    async with conn.transaction():
        inserted = await wf_queries.append_run_event(
            conn, account_id=run.account_id, run_id=run.id, type="run_completed", payload=payload
        )
        if status == "errored" and (error := payload.get("error")) is not None:
            kind = error.get("kind")
            if kind in {"engine_semantics_changed", "nondeterministic_replay"}:
                terminal_marked = await _fail_child_requests_for_terminal_error(
                    conn, run, error_kind=kind
                )
        # A run in service of a request answers via its terminal record itself (#1126):
        # the ``run_completed`` bookend (above) + the ``status`` flip (below) ARE the
        # answer, read back by ``derive_run_response``. No separate ``request_response``
        # event is written — the run is singly-inbound, so its terminal state already
        # carries the one outcome (§3.6); this also lets a cancelled run resolve as
        # ``cancelled`` rather than the ``child_gone`` a gated-off response implied.
        await wf_queries.set_run_terminal(
            conn, run.id, status=status, output=output, account_id=run.account_id
        )
        cascade_children = (
            await seed_outbound_cancel_conn(
                conn, caller_kind="run", caller_id=run.id, account_id=run.account_id
            )
            if inserted is not None
            else []
        )
        # Durable run-caller wake (C4/C5): a run answering a RUN caller writes a
        # ``child_done`` signal into the CALLER's signal side-table (never its journal —
        # single-writer-safe), keyed by this run's ``request_id`` (= the caller's
        # ``invoke_workflow`` ``call_key``). It is the run-side analog of
        # ``write_child_response``'s session→run seam: it makes the best-effort post-commit
        # ``defer_run_wake`` below recoverable via the unharvested-signal sweep clause —
        # ``invoke_workflow`` maps to NULL in the staleness CASE, so it has NO other
        # backstop. Fires for EVERY terminal, including ``cancelled``: this is what stops a
        # cancelled sub-run stranding its parent (the 6b liveness gap). The harvest journals
        # the matching ``call_result``, clearing the signal — no hot-loop.
        if run_caller_id is not None:
            assert run.request_id is not None  # implied by run_caller_id's gate
            await wf_queries.insert_run_signal(
                conn, run_id=run_caller_id, call_key=run.request_id, kind="child_done"
            )
        # Inline-script runs carry no ``workflow_id`` (#1466): a ``run_completion``
        # trigger always keys off a concrete ``workflow_id``, so a workflow-less run
        # can never match one. Skip the query entirely — this also satisfies the
        # ``insert_run_completion_fires`` ``workflow_id: str`` contract.
        if inserted is not None and run.workflow_id is not None:
            fires = await db_queries.insert_run_completion_fires(
                conn,
                account_id=run.account_id,
                workflow_id=run.workflow_id,
                run_id=run.id,
                status=status,
            )
    # Prompt the CALLER run to harvest this answer on its next step rather than waiting
    # out the periodic sweep — mirroring how an agent() child wakes its run. Only for a
    # run caller (caller.kind=='run'); a session/api caller resumes through its own poll
    # path. Best-effort + post-commit: the answer is durable and the ``child_done`` signal
    # written above recovers a lost wake, so this fires for EVERY terminal — a cancelled
    # sub-run wakes its parent just like a completed one.
    if inserted is not None and run_caller_id is not None:
        with contextlib.suppress(Exception):
            await defer_run_wake(run_caller_id, batch=True)
    # Prompt each seeded child to run its cancel leaf now instead of waiting out the
    # durable backstops (the C2 marker sweep / the run cancel-signal sweep clause).
    # Session children use the same pool/deferred-wake plumbing ``archive_session``
    # rides after its own outbound pass; the worker's runtime pool is always set here
    # (``run_workflow_step`` required it at step entry).
    for child in cascade_children:
        if child.kind == "run":
            await defer_run_wake(child.id, batch=True)
        else:
            await defer_wake(
                runtime.require_pool(), child.id, cause="cancel", account_id=run.account_id
            )
    for marked_session_id in terminal_marked:
        await defer_wake(
            runtime.require_pool(), marked_session_id, cause="cancel", account_id=run.account_id
        )
    for fire in fires:
        try:
            await defer_trigger_fire(fire.trigger_id, fire.trigger_run_id)
        except Exception:
            log.exception("trigger.fire_defer_failed", trigger_run_id=fire.trigger_run_id)
    # Release the run's ephemeral sandbox now the run is terminal (#988). Fires on
    # EVERY terminal state — completed / errored / cancelled all route through here
    # — and is best-effort: ``teardown_run_sandbox`` schedules a fire-and-forget
    # ``release_run`` and NEVER raises, so a teardown hiccup can't corrupt the
    # just-committed terminal transition. If a sandbox was never provisioned (the
    # common no-bash run), ``release_run`` is a no-op.
    run_sandbox.teardown_run_sandbox(run.id)
