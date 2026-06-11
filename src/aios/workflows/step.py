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
        # A call_key with a `frontier_deferred` marker but no `call_started` (and not
        # yet resolved into memo) is a WAITING-to-be-admitted agent frontier. On a
        # completed replay the script MUST re-emit it (it has neither started nor
        # resolved); if it didn't, the run diverged just as surely as a vanished
        # inflight call — fail closed rather than strand a frontier that will never
        # be admitted. A key that later gets a `call_started` (admitted) is tracked
        # by `inflight` instead, and one that resolved is in `memo` — both excluded.
        # A "started" key is one that already had a `call_started`: either it's
        # still open (`inflight`) or it has since resolved (`memo`). Every memo'd
        # key provably had a prior `call_started` (call_result requires it,
        # schema-enforced), so this is exactly the set of all `call_started` keys
        # — no separate scan of `events` needed.
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
        # unbounded loop. Counted from the monotonic call_started journal plus this
        # step's new agent frontier, checked BEFORE any spawn so an over-cap step
        # errors atomically — never leaving a partial fan-out of orphan children.
        # (A single parallel()'s width is bounded separately in the host.) Gated on
        # there BEING new spawns: a harvest-only re-suspend (no new agents) must never
        # error — else lowering the cap mid-flight would retroactively kill a run whose
        # children are already all inflight, orphaning them. H1 blocks NEW spawns only.
        new_agent_caps = [
            cap
            for cap in outcome.emitted
            if cap.capability_id == "agent"
            and cap.call_key not in memo
            and cap.call_key not in inflight
        ]
        prior_agent_calls = sum(
            1 for e in events if e.type == "call_started" and e.payload.get("capability") == "agent"
        )
        max_agent_calls = get_settings().workflow_max_agent_calls
        if new_agent_caps and prior_agent_calls + len(new_agent_caps) > max_agent_calls:
            await _complete_run(
                conn,
                run,
                output=(
                    f"workflow exceeded the {max_agent_calls}-agent call cap "
                    f"({prior_agent_calls} started, {len(new_agent_caps)} more requested)"
                ),
                is_error=True,
                error_kind="too_many_agents",
            )
            return

        # Per-run wave admission (#784): bound the number of concurrently in-flight
        # agent() children. Count this run's currently in-flight agents (harvested
        # `inflight` map), and admit at most `slots` NEW agent frontiers this wake;
        # journal the rest as `frontier_deferred` (no spawn, no call_started). Freed
        # slots admit deferred frontiers on the next wake — the existing child-
        # completion re-wake re-runs this step, no new machinery. Gate and tool
        # frontiers are UNTHROTTLED. H1 above already counted the full new-agent set,
        # so the wave gate neither masks nor false-trips the lifetime cap; a
        # re-emitted deferred frontier is still "new" (no call_started), so H1's
        # count is identical across wakes.
        inflight_agents = sum(
            1 for e in inflight.values() if e.payload.get("capability") == "agent"
        )
        slots = max(0, get_settings().workflow_max_inflight_children_per_run - inflight_agents)

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
                if slots > 0:
                    spawn = await _open_agent_capability(conn, pool, run, cap)
                    if spawn.rejected:
                        return  # the frontier was terminally rejected (bad call) — run errored
                    if spawn.needs_rewake:
                        needs_rewake = True  # C1'/C4: harvest the already-present marker next step
                    slots -= 1
                else:
                    # Over the per-run wave cap: defer this frontier. Journal a
                    # `frontier_deferred` marker (idempotent on (run_id, call_key,
                    # type)) so the divergence guard sees a waiting agent. Do NOT
                    # spawn and do NOT journal call_started. A later wake re-emits this
                    # same frontier (no call_started ⇒ still "new"); when a child
                    # resolves and frees a slot, it is admitted then. Deferral is a
                    # WAIT, never an error — it must never reach _open_agent_capability.
                    await wf_queries.append_run_event(
                        conn,
                        account_id=account_id,
                        run_id=run_id,
                        type="frontier_deferred",
                        call_key=cap.call_key,
                        payload={"capability": "agent"},
                    )
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
        await wf_queries.set_run_status(conn, run_id, "suspended", account_id=account_id)

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

    rejected: bool  # the call was terminally rejected (run errored) — stop the step
    needs_rewake: bool  # a re-attached child already has its marker — self-wake to harvest


async def _open_agent_capability(
    conn: asyncpg.Connection[Any],
    pool: asyncpg.Pool[Any],
    run: WfRun,
    cap: EmittedCapability,
) -> _SpawnResult:
    """Spawn (or, on replay/C1', re-attach) the ``agent()`` child for a frontier,
    then journal ``call_started{child_session_id, child_agent_version}``.

    Idempotent + crash-safe: the child id is deterministic, ``create_child_session``
    is ``ON CONFLICT`` (delivers the input atomically with the row on first spawn),
    ``call_started`` dedups on the memo. On replay the child id already exists →
    re-attach, journaling the *row's* pinned version (not a re-resolved one).

    Returns ``rejected=True`` iff the call was terminally rejected (the run was
    errored — the caller should stop the step). ``defer_wake`` fires **only on
    first spawn** (``created``): on a re-attach the child is already sweep-wakeable
    via its own first user message and may even be terminal, so appending a wake
    span to it would crash. ``needs_rewake=True`` asks the caller for a self-wake
    when a re-attached child *already* has its completion marker (C1'/C4: it
    finished before this wake journaled ``call_started``, so the pre-replay harvest
    — which only sees inflight ``call_started`` events — missed it).
    """
    account_id = run.account_id
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
        # schema. Three author-facing failures: a non-object schema (a bare boolean is
        # valid JSON Schema, but `false` rejects every value and `true` disables
        # enforcement), a structurally-invalid schema, and one with an unresolvable
        # reference (passes check_schema but raises at validation time).
        schema_error: str | None = None
        if not isinstance(output_schema, dict):
            schema_error = (
                f"output_schema must be a JSON object schema, got {type(output_schema).__name__}"
            )
        else:
            try:
                jsonschema.Draft202012Validator.check_schema(output_schema)
            except jsonschema.SchemaError as exc:
                schema_error = f"output_schema is not a valid JSON Schema: {exc.message}"
            else:
                if (bad_ref := _unresolvable_ref(output_schema)) is not None:
                    schema_error = (
                        f"output_schema has an unresolvable reference {bad_ref!r} "
                        "(references must resolve within the schema; remote refs are unsupported)"
                    )
        if schema_error is not None:
            await _complete_run(
                conn,
                run,
                output=f"agent() {schema_error}",
                is_error=True,
                error_kind="bad_agent_call",
            )
            return _SpawnResult(rejected=True, needs_rewake=False)
    agent_id = spec.get("agent_id")
    if not isinstance(agent_id, str):
        await _complete_run(
            conn,
            run,
            output=f"agent() requires a string agent_id, got {agent_id!r}",
            is_error=True,
            error_kind="bad_agent_call",
        )
        return _SpawnResult(rejected=True, needs_rewake=False)

    child_id = child_session_id(run.id, cap.call_key)
    try:
        # The full agent (head row) — both the pinned version AND its declared surface,
        # which the run clamps. One query, same NotFoundError contract as before.
        agent = await db_queries.get_agent(conn, agent_id, account_id=account_id)
    except NotFoundError:
        await _complete_run(
            conn,
            run,
            output=f"agent {agent_id!r} not found",
            is_error=True,
            error_kind="agent_not_found",
        )
        return _SpawnResult(rejected=True, needs_rewake=False)
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
