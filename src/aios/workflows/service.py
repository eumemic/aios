"""Service-layer entry points for the workflows runtime (Block 1, internal).

No HTTP/CLI surface yet — these are the functions tests and (later) the API layer
call to create and resume runs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db.queries import get_environment, get_session_bare, get_session_vault_ids
from aios.db.queries import workflows as wf_queries
from aios.errors import AiosError, ConflictError, ForbiddenError, NotFoundError, RateLimitedError
from aios.models.attenuation import Surface, surface_of
from aios.models.workflows import WfRun
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services.wake import defer_run_wake
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH

# The single shared trusted-invocation depth budget (#1124): how many trusted
# hops a chain of invocations (run→run sub-launches, run→session ``agent()``
# children, and — once #1127/#1128 land their call sites — session→session and
# api→session) may take before refusal. The DOWN-counter that replaces the
# run-only ``parent_run_id`` ancestor walk (the deleted ``run_ancestor_depth``
# CTE): every trusted edge carries ``parent.depth - 1`` and an edgeless root
# (operator/HTTP ``POST /runs``, foreground session) seeds at the full budget.
# The decrement IS the cycle bound — cycles (incl. session→session A↔B) bottom
# out at the budget BY CONSTRUCTION, no wait-for-graph. The wake-side
# ``WAKE_SESSION_MAX_DEPTH`` (#1083) is a separate carrier, left untouched here.
INVOKE_MAX_DEPTH = 10


class WorkflowRunDepthExceededError(AiosError):
    """A trusted invocation would nest past the shared depth budget (#1124).

    Raised before the over-budget child run/edge is written — the model-visible
    ``409`` refusal at every trusted-invocation hop (the ``error_type`` is kept
    stable for clients across the run-only → edge-carried generalization).
    """

    error_type = "workflow_run_depth_exceeded"
    status_code = 409


async def create_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    workflow_id: str,
    environment_id: str,
    input: Any = None,
    vault_ids: list[str] | None = None,
    launcher_session_id: str | None = None,
    parent_run_id: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    caller: dict[str, Any] | None = None,
    request_output_schema: dict[str, Any] | None = None,
    expected_version: int | None = None,
    budget_usd: float | None = None,
    default_child_model: str | None = None,
) -> WfRun:
    """Create a run that snapshots the workflow's current script, then wake it.

    The run carries its own immutable ``script`` (+ ``script_sha``), so every
    wake execs exactly that source regardless of later edits to the workflow.
    ``environment_id`` binds the run (and the sessions its ``agent()`` children
    spawn into) to an environment — chosen at run-creation time, like a session's.
    It is validated as account-owned (``[security]``: a bare FK would accept another
    tenant's env id and leak its image/env-vars/networking into this run).

    ``vault_ids`` binds credentials to the run (resolved at tool-call time, like a
    session's). **Launch-time attenuation:** when ``launcher_session_id`` is set (an
    agent launching the run), the requested vaults must be a subset of the launcher's
    own — authority never exceeds the invoker, so a breach raises
    :class:`ForbiddenError`. With no launcher (the HTTP/operator path) the requested
    vaults bind as-is, account-scoped. Insert + bind are one transaction, so a breach
    or a bad vault leaves no run row; the wake fires only after commit.

    ``parent_run_id`` records run lineage (an agent inside a run launching a sub-run).
    It also drives the **DOWN-counting trusted depth budget** (#1124): the new run
    carries ``parent.depth - 1`` and the launch is **refused before any row is written**
    when the parent run has no budget left — :class:`WorkflowRunDepthExceededError`. The
    operator/HTTP path passes no parent, so it is an **edgeless root** seeded at the full
    budget (``INVOKE_MAX_DEPTH``) — a chain launched off it still bottoms out at the
    budget by construction.

    ``expected_version`` is the trigger ``workflow`` action's drift-assertion pin:
    when set, the workflow's CURRENT version must equal it or the launch raises
    :class:`ConflictError` — checked here, at the same consistency point as the
    script snapshot (a caller-side pre-check would race a concurrent
    ``update_workflow`` between check and snapshot). ``None`` (every existing
    caller) floats to the current version.

    **Horizontal fan-out caps:** outstanding (non-terminal) runs are bounded per
    launcher session (``workflow_runs_per_launcher_max``, agent path only) and per
    account (``workflow_runs_per_account_max``, every launch) — a breach raises
    :class:`RateLimitedError`. COUNT+INSERT are serialized by a per-account advisory
    lock, so the caps are contractual against concurrent launches. (A concurrently
    *completing* run flips terminal without the lock, so a count can only be
    stale-high — a conservative early refusal, never a cap breach.)
    """
    requested = list(vault_ids or [])
    # #794 top edge: an agent-launched run cannot exceed the launcher's own surface.
    # #835: the launcher's effective surface is read INSIDE the run transaction (below),
    # threading `conn` into load_for_session — the same consistency point as the vault
    # check and the snapshot write, so a concurrent agent edit can't land a stale-broad
    # snapshot. The operator/HTTP path (no launcher) is the lattice top — the run
    # snapshots the workflow verbatim. Threading `conn` (vs a second pool.acquire())
    # keeps the whole path single-connection, so it is safe on a size-1 pool.
    launcher_surface: Surface | None = None
    run_default_child_model = default_child_model or get_settings().workflow_default_child_model
    async with pool.acquire() as conn, conn.transaction():
        # Idempotent re-attach (#1129): a deterministic ``run_id`` is the
        # ``invoke_workflow`` sub-run spawn's replay key. A crash between this
        # insert and the parent's ``call_started`` journal write re-drives the
        # spawn; recomputing the same id and finding the existing row must
        # re-attach (return it) rather than re-validate/re-insert (which would
        # both double-count fan-out and ``insert_wf_run``'s ``ON CONFLICT DO
        # NOTHING`` would return None). The operator/HTTP/agent ``create_run``
        # paths pass no ``run_id`` and always mint a fresh one.
        if run_id is not None:
            existing = await wf_queries.get_run_for_step(conn, run_id)
            if existing is not None and existing.account_id == account_id:
                return existing
        await get_environment(conn, environment_id, account_id=account_id)  # 404s foreign/absent
        workflow = await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)
        if workflow.archived_at is not None:
            raise ConflictError(f"workflow {workflow_id} is archived", detail={"id": workflow_id})
        if expected_version is not None and workflow.version != expected_version:
            raise ConflictError(
                f"workflow version drift: pinned {expected_version}, current {workflow.version}",
                detail={
                    "pinned": expected_version,
                    "current": workflow.version,
                    "id": workflow_id,
                },
            )
        script_sha = hashlib.sha256(workflow.script.encode("utf-8")).hexdigest()
        if launcher_session_id is not None:
            launcher_session = await get_session_bare(
                conn, launcher_session_id, account_id=account_id
            )
            launcher_agent = await agents_service.load_for_session(
                pool, launcher_session, account_id=account_id, conn=conn
            )
            launcher_surface = surface_of(launcher_agent)
            run_default_child_model = launcher_agent.model
        # Clamp the snapshot to the launcher's surface (sub-runs compose for free: a
        # child launcher's load_for_session already returns its frozen clamp).
        effective = (
            attenuation_service.clamp(surface_of(workflow), launcher_surface)
            if launcher_surface is not None
            else surface_of(workflow)
        )
        # The DOWN-counting trusted depth budget (#1124). An edgeless root
        # (operator/HTTP ``POST /runs``, a trigger fire with no completing-run
        # parent) seeds at the full budget; a nested launch reads the parent
        # run's remaining budget off its row and refuses BEFORE writing the
        # child when none is left, then carries ``parent.depth - 1``.
        if parent_run_id is None:
            child_depth = INVOKE_MAX_DEPTH
        else:
            # ``parent_run_id`` is trusted same-account. Two callers set it: the
            # ``create_run`` builtin, threading the launcher session's own
            # ``parent_run_id`` (set by the run-spawn machinery to a same-account
            # run), and the trigger fire path (#819), threading either the
            # completing run's id (same-account by the completion matcher's
            # account-equality conjunct) or the owner session's own
            # ``parent_run_id`` — the same provenance as the builtin. The
            # account-scoped read relies on that — a foreign/missing parent
            # raises NotFoundError. If a future path ever lets ``parent_run_id``
            # be caller-supplied, this same-account read is the gate (like
            # ``environment_id`` above).
            parent_depth = await wf_queries.get_run_depth(
                conn, parent_run_id, account_id=account_id
            )
            # Refuse-before-write: a parent with one (or zero) hop left cannot open
            # another trusted edge — the child would be born at depth 0 with no way
            # to bottom the chain out at the budget. The decrement IS the cycle
            # bound; this is the only refusal, no wait-for-graph.
            if parent_depth <= 1:
                raise WorkflowRunDepthExceededError(
                    f"trusted invocation would exceed depth budget {INVOKE_MAX_DEPTH}",
                    detail={"max_depth": INVOKE_MAX_DEPTH, "parent_run_id": parent_run_id},
                )
            child_depth = parent_depth - 1
        if launcher_session_id is not None:
            held = set(
                await get_session_vault_ids(conn, launcher_session_id, account_id=account_id)
            )
            ungranted = [v for v in requested if v not in held]
            if ungranted:
                raise ForbiddenError(
                    "run requested vaults the launching agent does not hold",
                    detail={"ungranted_vault_ids": ungranted},
                )
        # Fan-out caps, last (after all other validation, so a doomed launch never
        # takes the lock). The advisory lock serializes COUNT+INSERT account-wide.
        settings = get_settings()
        await wf_queries.acquire_account_wf_runs_lock(conn, account_id)
        if launcher_session_id is not None:
            launcher_cap = settings.workflow_runs_per_launcher_max
            outstanding = await wf_queries.count_active_runs(
                conn, account_id=account_id, launcher_session_id=launcher_session_id
            )
            if outstanding >= launcher_cap:
                raise RateLimitedError(
                    f"launcher at outstanding-run cap ({outstanding}/{launcher_cap}); "
                    "wait for runs you launched to finish (await_run) or cancel one "
                    "you no longer need (cancel_run) to free a slot",
                    detail={"outstanding": outstanding, "max": launcher_cap},
                )
        account_cap = settings.workflow_runs_per_account_max
        outstanding = await wf_queries.count_active_runs(conn, account_id=account_id)
        if outstanding >= account_cap:
            raise RateLimitedError(
                f"account at outstanding-run cap ({outstanding}/{account_cap}); "
                "wait for outstanding runs to complete, or have stuck runs cancelled, "
                "to free a slot",
                detail={"outstanding": outstanding, "max": account_cap},
            )
        run = await wf_queries.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=workflow_id,
            environment_id=environment_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            launcher_session_id=launcher_session_id,
            request_id=request_id,
            caller=caller,
            request_output_schema=request_output_schema,
            script=workflow.script,
            script_sha=script_sha,
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            input=input,
            # Snapshot the launch-clamped surface (like script), so a later
            # update_workflow never shifts this run's tool-authority.
            tools=effective.tools,
            mcp_servers=effective.mcp_servers,
            http_servers=effective.http_servers,
            budget_usd=budget_usd,
            default_child_model=run_default_child_model,
            depth=child_depth,
        )
        if requested:
            await wf_queries.set_run_vaults(conn, run.id, requested, account_id=account_id)
        # The run→run request edge (#1126/#1129) needs no session-scoped
        # ``request_opened`` event here: a run has no session ``events`` log, and
        # the *ask* half is already carried by the launching run's
        # ``call_started{capability:'invoke_workflow'}`` journal frame, whose
        # symmetric *answer* — the sub-run's ``request_response`` (#1126), keyed
        # on the same ``request_id``/``caller`` recorded on this row — is resolved
        # by ``derive_run_response``. (The session-creating launch sites
        # ``create_child_session`` / ``_open_agent_capability`` still emit the
        # session ``request_opened`` for their session-scoped edges, #1123.)
    await defer_run_wake(run.id)
    return run


async def resume_gate(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    call_key: str,
    result: Any,
) -> None:
    """Deliver an external resume for a suspended gate, keyed by ``call_key``.

    The internal/trusted variant: it keys off ``call_key`` directly and does NOT
    account-scope, so it must never be the HTTP path. The HTTP face is
    :func:`aios.services.workflows.resume_gate_by_nonce`, which scopes the run and
    resolves the public ``gate_nonce`` to a ``call_key`` before calling through here.
    Both record a durable ``wf_run_signals`` row and defer a wake; the run's next
    step harvests the signal into the journal — keeping ``wf_run_events`` single-writer.
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_run_for_step(conn, run_id)
        if run is None:
            raise NotFoundError(f"workflow run {run_id} not found", detail={"id": run_id})
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)
