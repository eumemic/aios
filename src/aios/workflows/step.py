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
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import asyncpg
import jsonschema
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012
from structlog.contextvars import bind_contextvars, clear_contextvars

from aios.config import get_settings
from aios.db import queries as db_queries
from aios.db.queries import workflows as wf_queries
from aios.errors import NotFoundError
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.attenuation import surface_of
from aios.models.workflows import TERMINAL_RUN_STATUSES, WfRun, WfRunEvent
from aios.services import attenuation as attenuation_service
from aios.services.sessions import create_child_session
from aios.services.wake import defer_run_wake, defer_trigger_fire, defer_wake
from aios.workflows import run_tools
from aios.workflows.child_id import child_session_id
from aios.workflows.host_launcher import EmittedCapability, run_script_host

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


def _memo_outcome(call_result_payload: dict[str, Any]) -> dict[str, Any]:
    """Map a ``call_result`` payload to the host memo's tagged outcome: a value the
    driver fast-forwards into the await (``{"ok": value}``), or an error it throws
    there as ``AgentError`` (``{"error": {...}}``). The discriminated union lets the
    host tell "the agent answered" from "the agent failed" even when the answer is
    itself a dict — the real value always nests one level under ``"ok"``."""
    if call_result_payload.get("is_error"):
        return {"error": call_result_payload.get("error")}
    return {"ok": call_result_payload.get("result")}


async def _resolve_agent_call(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    child_id: str,
    request_id: str,
    started_at: datetime,
    now: datetime,
    deadline: timedelta,
) -> dict[str, Any] | None:
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
        await db_queries.write_response_if_absent(
            conn,
            child_id,
            account_id=account_id,
            request_id=request_id,
            is_error=True,
            result=None,
            error={"kind": "timeout"},
        )
    resolved = await db_queries.derive_response(
        conn, child_id, account_id=account_id, request_id=request_id
    )
    assert resolved is not None  # a response now exists (timeout/child) or child_gone
    return resolved


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

        memo: dict[str, Any] = {
            e.call_key: _memo_outcome(e.payload)
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
                result_payload = await _resolve_agent_call(
                    conn,
                    account_id=account_id,
                    child_id=cap_payload["child_session_id"],
                    request_id=call_key,
                    started_at=cap_event.created_at,
                    now=now,
                    deadline=agent_deadline,
                )
                if result_payload is None:
                    continue  # still pending, within deadline — stay suspended
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
            memo[call_key] = _memo_outcome(result_payload)
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

    needs_rewake = False
    # A spawn rejection journaled this wake (a catchable agent() error). The run owes one
    # more drive so the replay throws the AgentError at the await — so we DON'T settle into
    # a quiet 'suspended' (which the next wake's quiet-wake guard would early-exit), we hand
    # the lease back as 'running' and let the guaranteed self-wake (needs_rewake) drive it.
    # One-shot by construction: a rejected call lands in memo, so its capability is skipped
    # on replay (never re-journaled), and the driven wake re-suspends/completes normally.
    rejected_any = False
    # (call_key, tool_name, input) for tool frontiers opened this wake — launched AFTER
    # the txn commits (below), so call_started is visible before a task can signal+wake.
    tools_to_launch: list[tuple[str, str, Any]] = []
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
                spawn = await _open_agent_capability(
                    conn, pool, run, cap, agent_spawns=agent_spawns, max_agent_calls=max_agent_calls
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
                            f"({prior_agent_calls} started before this step)"
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
                    needs_rewake = True
                if spawn.rejected:
                    rejected_any = True
                else:
                    agent_spawns += 1  # a real child was created (or re-attached) — it counts
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
                tools_to_launch.append((cap.call_key, tool_name, spec.get("input")))
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
            conn, run_id, "running" if rejected_any else "suspended", account_id=account_id
        )

    # Outside the txn (after the suspend commits): launch this wake's tool tasks — now
    # that their call_started rows are committed, a task that signals+wakes lands a step
    # that sees them (no double-dispatch) — then self-wake so the next step harvests an
    # already-delivered gate resume opened this wake.
    for launch_key, launch_name, launch_input in tools_to_launch:
        run_tools.launch_tool_task(
            pool, run, call_key=launch_key, tool_name=launch_name, tool_input=launch_input
        )
    if needs_rewake:
        await defer_run_wake(run_id)


class _SpawnResult(NamedTuple):
    """Outcome of opening one ``agent()`` frontier."""

    # this call's catchable error was journaled — self-wake so the run replays and throws
    rejected: bool
    needs_rewake: bool  # a re-attached child already has its marker — self-wake to harvest
    # the cap passed every rejection gate but creating its child would exceed the lifetime
    # agent-call cap — the caller terminates the run (``too_many_agents``); no child created
    quota_exceeded: bool = False


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
        # Journal the rejection as a CATCHABLE call_result error for this call_key —
        # the same channel every other agent failure uses — instead of terminally
        # erroring the run. The descriptive ``message`` is the author's only source
        # (``_AGENT_ERROR_DEFAULT_MESSAGES`` has no entry for these kinds), so it is
        # preserved verbatim. The caller self-wakes; the next step's replay throws
        # ``_agent_error_from`` at the ``await``, where the author can try/except it.
        await wf_queries.append_run_event(
            conn,
            account_id=account_id,
            run_id=run.id,
            type="call_result",
            call_key=cap.call_key,
            payload={"result": None, "is_error": True, "error": {"kind": kind, "message": message}},
        )
        return _SpawnResult(rejected=True, needs_rewake=False)

    spec = cap.spec if isinstance(cap.spec, dict) else {}
    # agent() carries output_schema as a canonical JSON *string* (so a schema's floats
    # survive the call_key hash); reconstruct the dict. None means no schema demanded.
    output_schema_raw = spec.get("output_schema")
    output_schema = (
        json.loads(output_schema_raw) if isinstance(output_schema_raw, str) else output_schema_raw
    )
    if output_schema is not None:
        # Structured output: the child must return a value matching this schema (the
        # return tool enforces it; see workflow_completion). Reject a bad schema here —
        # author-facing — rather than letting it brick the child when it applies the
        # schema. Three sequentially-dependent author-facing failures, each an early
        # reject: a non-object schema (a bare boolean is valid JSON Schema, but `false`
        # rejects every value and `true` disables enforcement), a structurally-invalid
        # schema, and one with an unresolvable reference (passes check_schema but raises
        # at validation time).
        if not isinstance(output_schema, dict):
            return await _reject(
                "bad_agent_call",
                f"agent() output_schema must be a JSON object schema, "
                f"got {type(output_schema).__name__}",
            )
        try:
            jsonschema.Draft202012Validator.check_schema(output_schema)
        except jsonschema.SchemaError as exc:
            return await _reject(
                "bad_agent_call", f"agent() output_schema is not a valid JSON Schema: {exc.message}"
            )
        if (bad_ref := _unresolvable_ref(output_schema)) is not None:
            return await _reject(
                "bad_agent_call",
                f"agent() output_schema has an unresolvable reference {bad_ref!r} "
                "(references must resolve within the schema; remote refs are unsupported)",
            )
    agent_id = spec.get("agent_id")
    if not isinstance(agent_id, str):
        return await _reject(
            "bad_agent_call", f"agent() requires a string agent_id, got {agent_id!r}"
        )

    child_id = child_session_id(run.id, cap.call_key)
    try:
        # The full agent (head row) — both the pinned version AND its declared surface,
        # which the run clamps. One query, same NotFoundError contract as before.
        agent = await db_queries.get_agent(conn, agent_id, account_id=account_id)
    except NotFoundError:
        return await _reject("agent_not_found", f"agent {agent_id!r} not found")
    # Lifetime cap (H1) enforced HERE — past every rejection gate, before the child is
    # created — so only caps that genuinely spawn count. ``agent_spawns`` already excludes
    # rejected caps; this child would be the (agent_spawns + 1)-th, so cap it on strict ``>``.
    if agent_spawns + 1 > max_agent_calls:
        return _SpawnResult(rejected=False, needs_rewake=False, quota_exceeded=True)
    pinned = agent.version
    # #794: the child wields agent ∩ run, frozen at spawn; its vaults are the run's.
    child_surface = attenuation_service.clamp(surface_of(agent), surface_of(run))
    run_vaults = await wf_queries.get_run_vault_ids(conn, run.id, account_id=account_id)

    created = await create_child_session(
        pool,
        session_id=child_id,
        account_id=account_id,
        agent_id=agent_id,
        environment_id=run.environment_id,
        agent_version=pinned,
        parent_run_id=run.id,
        surface=child_surface,
        vault_ids=run_vaults,
        request_id=cap.call_key,  # the agent() call IS the request the child must answer
        input=spec.get("input"),
        output_schema=output_schema,
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


async def _complete_run(
    conn: asyncpg.Connection[Any],
    run: WfRun,
    *,
    output: Any,
    is_error: bool,
    error_kind: str | None = None,
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
    payload: dict[str, Any] = {"output": output, "is_error": is_error}
    if error_kind is not None:
        payload["error"] = {"kind": error_kind}
    await _commit_terminal_and_dispatch(
        conn,
        run,
        status="errored" if is_error else "completed",
        payload=payload,
        output=None if is_error else output,
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


async def _commit_terminal_and_dispatch(
    conn: asyncpg.Connection[Any],
    run: WfRun,
    *,
    status: str,
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
    async with conn.transaction():
        inserted = await wf_queries.append_run_event(
            conn, account_id=run.account_id, run_id=run.id, type="run_completed", payload=payload
        )
        await wf_queries.set_run_terminal(
            conn, run.id, status=status, output=output, account_id=run.account_id
        )
        if inserted is not None:
            fires = await db_queries.insert_run_completion_fires(
                conn,
                account_id=run.account_id,
                workflow_id=run.workflow_id,
                run_id=run.id,
                status=status,
            )
    for fire in fires:
        try:
            await defer_trigger_fire(fire.trigger_id, fire.trigger_run_id)
        except Exception:
            log.exception("trigger.fire_defer_failed", trigger_run_id=fire.trigger_run_id)
