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
from aios.errors import (
    AiosError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from aios.models.agents import (
    HttpServerRef,
    McpServerSpec,
    ToolSpec,
    resolve_http_server_refs,
)
from aios.models.attenuation import Surface, surface_diff, surface_of
from aios.models.workflows import WfRun
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services.wake import defer_run_wake
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from aios.workflows.script_validation import validate_workflow_script

# The single shared trusted invoke-depth budget (#1124): how many trusted
# hops a chain of invoke-edges (run→run sub-launches, run→session ``agent()``
# children, and — since #1127/#1128 landed their call sites — session→session and
# api→session) may take before refusal. The DOWN-counter that replaces the
# run-only ``parent_run_id`` ancestor walk (the deleted ``run_ancestor_depth``
# CTE): every trusted edge carries ``parent.depth - 1`` and an edgeless root
# (operator/HTTP ``POST /runs``, foreground session) seeds at the full budget.
# The decrement IS the cycle bound — cycles (incl. session→session A↔B) bottom
# out at the budget BY CONSTRUCTION, no wait-for-graph. The wake-side
# ``WAKE_SESSION_MAX_DEPTH`` (#1083) is a separate carrier, left untouched here.
INVOKE_MAX_DEPTH = 10


class WorkflowRunDepthExceededError(AiosError):
    """A trusted invoke-edge would nest past the shared depth budget (#1124).

    Raised before the over-budget child run/edge is written — the model-visible
    ``409`` refusal at every trusted invoke-edge hop (the ``error_type`` is kept
    stable for clients across the run-only → edge-carried generalization).
    """

    error_type = "workflow_run_depth_exceeded"
    status_code = 409


class InlineScript:
    """The inline-script body of an anonymous run launch (T5, #1466).

    The alternative to a ``workflow_id`` in :func:`create_run`: a one-shot run is
    launched directly from this ``{script, schemas, surface}`` body, with NO
    ``workflows`` row created. The run snapshots ``script`` exactly as it snapshots a
    registered workflow's — the run already pins script-at-launch — so the inline
    script lives only in the run's pinned snapshot.

    ``input_schema`` / ``output_schema`` are the declared schemas (the run carries
    ``request_output_schema`` as the launch obligation; ``input_schema`` is accepted
    for surface parity with ``create_workflow`` and validation, not persisted on the
    run today). ``tools`` / ``mcp_servers`` / ``http_servers`` are the declared
    surface — clamped to the launcher with the same create-time clamp
    ``create_workflow`` uses (``ForbiddenError`` on exceed), then snapshotted onto the
    run like a registered launch's surface.
    """

    __slots__ = (
        "http_servers",
        "input_schema",
        "mcp_servers",
        "output_schema",
        "script",
        "tools",
    )

    def __init__(
        self,
        *,
        script: str,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        tools: list[ToolSpec] | None = None,
        mcp_servers: list[McpServerSpec] | None = None,
        http_servers: list[HttpServerRef] | None = None,
    ) -> None:
        self.script = script
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.tools = tools or []
        self.mcp_servers = mcp_servers or []
        self.http_servers: list[HttpServerRef] = list(http_servers or [])


async def _enforce_inline_surface(
    *,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerRef],
    launcher_agent: Any | None,
) -> Surface:
    """Clamp an inline run's declared surface to the launcher, the same create-time
    clamp ``services.workflows._enforce_surface_attenuation`` applies at authoring.

    Returns the effective (clamped) surface to snapshot onto the run; raises
    :class:`ForbiddenError` when the declared surface exceeds the launcher's — an
    agent cannot grant an inline run a tool/server it does not itself hold. http
    servers are admitted by identity (name + base_url) and their routes inherited
    launcher-frozen into storage; a names-only entry (#953) is resolved against the
    launching agent first.

    ``launcher_agent is None`` is the operator/HTTP path (no acting agent): the
    declared surface binds verbatim (the lattice top), but names-only http sugar is
    rejected (nothing to resolve a bare name against) — mirroring the operator path
    of ``create_workflow``.
    """
    if launcher_agent is None:
        bare = [s for s in http_servers if isinstance(s, str)]
        if bare:
            raise ForbiddenError(
                "names-only http_servers require an acting agent to resolve against; "
                "the operator path must declare full HttpServerSpec objects "
                f"(got bare names {bare!r})",
            )
        return Surface(
            tools,
            mcp_servers,
            [s for s in http_servers if not isinstance(s, str)],
        )
    # Resolve names-only entries against the launching agent (#953). An unknown name —
    # a grant the agent does not hold — fails closed as ForbiddenError, the same
    # authority breach as declaring a server the agent lacks by full identity.
    try:
        resolved_http = resolve_http_server_refs(http_servers, launcher_agent.http_servers)
    except ValueError as exc:
        raise ForbiddenError(
            "inline run surface exceeds the launching agent's permissions",
            detail={"exceeds": {"http_servers": [r for r in http_servers if isinstance(r, str)]}},
        ) from exc
    declared = Surface(tools, mcp_servers, resolved_http)
    expected = attenuation_service.normalize(declared)
    effective = attenuation_service.clamp(declared, surface_of(launcher_agent))
    surviving_http = {(s.name, s.base_url) for s in effective.http_servers}
    exceeds = (
        effective.tools != expected.tools
        or effective.mcp_servers != expected.mcp_servers
        or any((s.name, s.base_url) not in surviving_http for s in expected.http_servers)
    )
    if exceeds:
        raise ForbiddenError(
            "inline run surface exceeds the launching agent's permissions",
            detail={"exceeds": surface_diff(expected, effective)},
        )
    return effective


async def create_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    workflow_id: str | None = None,
    inline: InlineScript | None = None,
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
    version: int | None = None,
    budget_usd: float | None = None,
    default_child_model: str | None = None,
) -> WfRun:
    """Create a run that snapshots a script, then wake it.

    Two **mutually exclusive** source arms (exactly one required):

    * ``workflow_id`` — the registered path: snapshot a pre-registered workflow's
      script + declared surface (the existing behaviour, all the version/drift logic
      below applies).
    * ``inline`` (:class:`InlineScript`) — the **inline-script arm** (T5, #1466): a
      one-shot run launched from an inline ``{script, schemas, surface}`` body with
      **NO ``workflows`` row created**. The run snapshots the inline script (same as a
      registered launch — a run already pins script-at-launch), and the inline script's
      declared surface is clamped to the launcher with the same create-time clamp
      ``create_workflow`` uses: a surface that exceeds the launcher raises
      :class:`ForbiddenError` (vs the registered path's silent clamp, which can only
      narrow a definition the author already passed that same gate). The inline run's
      ``workflow_id`` / ``source_version`` are NULL; it execs identically to the
      equivalent register-then-run, minus the persisted definition. ``version`` /
      ``expected_version`` are meaningless on this arm (there is no definition history)
      and are rejected if passed.

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

    ``version`` (#1321) is the re-run SELECTOR — orthogonal to ``expected_version``.
    When set, the run snapshots that **specific historical** ``workflow_versions``
    row's script + declared surface (default ``None`` resolves to the workflow's
    current version). Both rows are loaded — never substituted: the **live**
    ``get_workflow`` still drives the archived gate and the ``expected_version``
    drift check (a version row has no ``archived_at``, so substituting it would
    silently drop the gate), while the chosen ``get_workflow_version`` drives the
    snapshotted script + surface. **Any version of an archived workflow is refused.**
    The chosen version's declared surface is clamped against the *current* launcher
    (the clamp is the real bound — a re-run can reproduce an old surface but never
    exceed the launcher's present authority). The run's ``source_version`` is bound
    to the snapshotted version via a strict composite FK; the run still execs its
    own inline ``script`` copy (reading through the FK is deferred Phase 3).

    **Horizontal fan-out caps:** outstanding (non-terminal) runs are bounded per
    launcher session (``workflow_runs_per_launcher_max``, agent path only) and per
    account (``workflow_runs_per_account_max``, every launch) — a breach raises
    :class:`RateLimitedError`. COUNT+INSERT are serialized by a per-account advisory
    lock, so the caps are contractual against concurrent launches. (A concurrently
    *completing* run flips terminal without the lock, so a count can only be
    stale-high — a conservative early refusal, never a cap breach.)
    """
    # Exactly one source arm: ``workflow_id`` (registered) XOR ``inline`` (T5, #1466).
    if (workflow_id is None) == (inline is None):
        raise ValidationError(
            "create_run requires exactly one of workflow_id or inline (got "
            f"{'both' if inline is not None else 'neither'})",
        )
    if inline is not None and (version is not None or expected_version is not None):
        # version/expected_version select/assert a definition version; an inline run
        # has no definition history, so they are meaningless on this arm.
        raise ValidationError(
            "version / expected_version are not valid for an inline run "
            "(an inline run has no workflow version history)",
        )
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
        # Read the launcher's surface ONCE up front (#835: inside the txn, threading
        # ``conn``, the same consistency point as the snapshot write). Both arms need
        # it: the registered arm to silently clamp the snapshotted surface, the inline
        # arm to *enforce* attenuation (ForbiddenError on exceed).
        launcher_agent = None
        if launcher_session_id is not None:
            launcher_session = await get_session_bare(
                conn, launcher_session_id, account_id=account_id
            )
            launcher_agent = await agents_service.load_for_session(
                pool, launcher_session, account_id=account_id, conn=conn
            )
            launcher_surface = surface_of(launcher_agent)
            run_default_child_model = launcher_agent.model
        source_version: int | None
        if inline is not None:
            # ── inline-script arm (T5, #1466) ──────────────────────────────────
            # No ``workflows`` row, no version history: snapshot the inline script
            # directly. ``workflow_id`` / ``source_version`` stay NULL on the run.
            # The script is validated (compile + ``async def main(input)`` + every
            # literal ``tool("…")`` covered) exactly as ``create_workflow`` validates
            # a registered script, so a malformed inline body fails as a clean
            # ValidationError here rather than as a failed run later.
            validate_workflow_script(inline.script, inline.tools)
            source_script = inline.script
            source_version = None
            # Clamp the inline surface to the launcher with the create-time gate:
            # exceeding the launcher raises ForbiddenError (vs the registered path's
            # silent clamp — there the author already passed this same gate at
            # create_workflow, so a re-clamp can only narrow, never breach).
            effective = await _enforce_inline_surface(
                tools=inline.tools,
                mcp_servers=inline.mcp_servers,
                http_servers=inline.http_servers,
                launcher_agent=launcher_agent,
            )
        else:
            assert workflow_id is not None
            # Load the LIVE workflow row: it drives the archived gate and the
            # expected_version drift check — both must consult the live head, not a
            # snapshot. (#1321 M2: the version row is loaded ADDITIONALLY below, never
            # substituted — a version row has no ``archived_at``, so substituting it
            # would silently drop the archived gate.)
            workflow = await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)
            if workflow.archived_at is not None:
                # Refuse launching ANY version of an archived workflow — the live gate
                # applies even to a historical re-run.
                raise ConflictError(
                    f"workflow {workflow_id} is archived", detail={"id": workflow_id}
                )
            if expected_version is not None and workflow.version != expected_version:
                raise ConflictError(
                    f"workflow version drift: pinned {expected_version}, "
                    f"current {workflow.version}",
                    detail={
                        "pinned": expected_version,
                        "current": workflow.version,
                        "id": workflow_id,
                    },
                )
            # Resolve the SOURCE definition the run snapshots. Default (version=None)
            # is the current head — use the live row directly (it IS the current
            # version, and its surface row is identical). A pinned ``version`` loads
            # the immutable ``workflow_versions`` snapshot for the script + declared
            # surface; a missing version 404s. ``source`` carries both the snapshotted
            # script and the chosen surface; ``source_version`` is bound onto the run.
            if version is None:
                source_script = workflow.script
                source_surface = surface_of(workflow)
                source_version = workflow.version
            else:
                source = await wf_queries.get_workflow_version(
                    conn, workflow_id, version, account_id=account_id
                )
                source_script = source.script
                source_surface = surface_of(source)
                source_version = source.version
            # Clamp the snapshot to the launcher's surface (sub-runs compose for free: a
            # child launcher's load_for_session already returns its frozen clamp).
            effective = (
                attenuation_service.clamp(source_surface, launcher_surface)
                if launcher_surface is not None
                else source_surface
            )
        script_sha = hashlib.sha256(source_script.encode("utf-8")).hexdigest()
        # The DOWN-counting trusted depth budget (#1124). An edgeless root
        # (operator/HTTP ``POST /runs``, a trigger fire with no completing-run
        # parent) seeds at the full budget; a nested launch reads the parent
        # run's remaining budget off its row and refuses BEFORE writing the
        # child when none is left, then carries ``parent.depth - 1``.
        if parent_run_id is None:
            child_depth = INVOKE_MAX_DEPTH
        else:
            # ``parent_run_id`` is trusted same-account. Two callers set it: the
            # ``call_workflow`` builtin, threading the launcher session's own
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
                    f"trusted invoke-edge would exceed depth budget {INVOKE_MAX_DEPTH}",
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
                    "wait for runs you launched to finish or stop one you no longer need "
                    "(stop_task, passing its tool_call_id) to free a slot",
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
            script=source_script,
            script_sha=script_sha,
            source_version=source_version,
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
        # ``call_started{capability:'invoke_workflow'}`` journal frame. The
        # symmetric *answer* is the sub-run's own terminal record (``run_completed``
        # + ``status``) — a run is singly-inbound, so its terminal state IS the
        # answer — read back by ``derive_run_response`` (§3.6; no separate
        # ``request_response`` event). (The session-creating launch sites
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
