"""Service layer for the workflows HTTP/CLI surface (Block 3).

The router imports this as ``from aios.services import workflows as service``,
matching every other resource router. It is the account-scoped, pool-taking face
over the query layer (``db.queries.workflows``) and the runtime entry points.

``create_run`` / ``resume_gate`` (the runtime entry points) live in
``aios.workflows.service`` — re-exported here so callers have one import, and kept
there because the integration tests patch ``aios.workflows.service.defer_run_wake``.
The HTTP path adds account-scoping (the runtime variants are internal/trusted) and
the **by-nonce** gate resume.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError, ValidationError
from aios.ids import REQUEST, make_id
from aios.models.agents import (
    HttpServerRef,
    HttpServerSpec,
    McpServerSpec,
    ToolSpec,
    resolve_http_server_refs,
)
from aios.models.attenuation import Surface, surface_diff, surface_of
from aios.models.workflows import (
    TERMINAL_RUN_STATUSES,
    WfRun,
    WfRunEvent,
    WfRunUsage,
    Workflow,
    WorkflowVersion,
)
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services import sessions as sessions_service
from aios.services.wake import defer_run_wake
from aios.workflows.script_validation import (
    declared_tool_names,
    extract_required_agent_ids,
    validate_workflow_script,
)
from aios.workflows.service import InlineScript, create_run, resume_gate

__all__ = [
    "InlineScript",
    "archive_run",
    "archive_workflow",
    "cancel_run",
    "create_run",
    "create_workflow",
    "get_run",
    "get_workflow",
    "get_workflow_version",
    "launch_awaited_run",
    "list_run_events",
    "list_runs",
    "list_workflow_versions",
    "list_workflows",
    "resume_gate",
    "resume_gate_by_nonce",
    "unarchive_workflow",
    "update_workflow",
]


# ─── run launch ──────────────────────────────────────────────────────────────


async def launch_awaited_run(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    workflow_id: str | None = None,
    inline: InlineScript | None = None,
    environment_id: str,
    input: Any = None,
    caller: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
    launcher_session_id: str | None = None,
    parent_run_id: str | None = None,
    vault_ids: list[str] | None = None,
    budget_usd: float | None = None,
    version: int | None = None,
) -> tuple[WfRun, str]:
    """Launch a run as an **awaited** servicer — the one place the run-as-Ask contract lives.

    Both Ask-shaped callers (the model ``call_workflow`` builtin and the API
    ``POST /v1/tasks`` workflow arm) go through here, so the contract — mint a fresh
    ``request_id`` and stamp ``caller.awaited=True`` so the run carries a response obligation —
    is correct-by-construction at one site rather than re-typed (and forgettable) at each.
    Returns ``(run, request_id)``; the caller awaits the run via the unified awaiter. The
    Tell-shaped trigger fire is deliberately NOT routed here — it launches fire-and-forget
    (no ``request_id``/``caller``) and calls :func:`create_run` directly.

    Pass EITHER ``workflow_id`` (registered) OR ``inline`` (the T5 inline-script arm,
    #1466) — :func:`create_run` enforces exactly-one.
    """
    request_id = make_id(REQUEST)
    run = await create_run(
        pool,
        account_id=account_id,
        workflow_id=workflow_id,
        inline=inline,
        environment_id=environment_id,
        input=input,
        vault_ids=vault_ids,
        launcher_session_id=launcher_session_id,
        parent_run_id=parent_run_id,
        request_id=request_id,
        caller={**caller, "awaited": True},
        request_output_schema=output_schema,
        budget_usd=budget_usd,
        version=version,
    )
    return run, request_id


# ─── workflow definitions ────────────────────────────────────────────────────


async def _enforce_surface_attenuation(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    actor_session_id: str,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerRef],
) -> Surface:
    """Raise ``ForbiddenError`` unless the declared surface is admissible against the acting
    agent's; return the effective (clamped) surface for storage.

    Serves both create (the creator's declared surface) and update (the editor's merged
    final surface). The predicate is per-dimension:

    * tools / mcp_servers — byte-equality of ``clamp(declared, actor)`` against
      ``canonicalize(declared)``: equal iff the meet did not narrow anything, which is
      exactly "``declared`` ≤ ``actor``" on membership *and* per-tool permission/transport
      and MCP server identity (joint name+url).
    * http_servers — identity survival: every declared ``(name, base_url)`` must appear in
      the clamp's http servers. The agent's routes/fields are inherited launcher-frozen
      into storage (the returned ``effective``), so a workflow need only name the server,
      not reproduce its routes byte-perfect. A bare-name entry (names-only sugar, #953)
      is resolved against the acting agent first — ``["davenant"]`` becomes the agent's
      ``davenant`` server (empty routes), then admitted by identity like any other.

    http servers remain parent-wins-frozen at run-time (run-launch clamps to
    launcher-verbatim) — only the AUTHORING gate relaxes to identity, which grants no new
    run-time authority. The HTTP/operator path passes no actor and skips this entirely.
    """
    session = await sessions_service.get_session_basic(
        pool, actor_session_id, account_id=account_id
    )
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    # Resolve names-only entries against the acting agent (#953). An unknown name —
    # a grant the agent does not hold — fails closed as a ForbiddenError, the same
    # authority breach as declaring a server the agent lacks by full identity.
    try:
        resolved_http = resolve_http_server_refs(http_servers, agent.http_servers)
    except ValueError as exc:
        raise ForbiddenError(
            "workflow surface exceeds the acting agent's permissions",
            detail={"exceeds": {"http_servers": [r for r in http_servers if isinstance(r, str)]}},
        ) from exc
    declared = Surface(tools, mcp_servers, resolved_http)
    expected = attenuation_service.normalize(declared)
    effective = attenuation_service.clamp(declared, surface_of(agent))
    surviving_http = {(s.name, s.base_url) for s in effective.http_servers}
    exceeds = (
        effective.tools != expected.tools
        or effective.mcp_servers != expected.mcp_servers
        or any((s.name, s.base_url) not in surviving_http for s in expected.http_servers)
    )
    if exceeds:
        raise ForbiddenError(
            "workflow surface exceeds the acting agent's permissions",
            detail={"exceeds": surface_diff(expected, effective)},
        )
    return effective


def _reject_operator_path_names_only(
    http_servers: list[HttpServerRef] | None,
) -> list[HttpServerSpec] | None:
    """The HTTP/operator path has no acting agent to resolve a bare name against, so
    names-only sugar (#953) is meaningless there — a bare string would otherwise leak
    into storage as a non-``HttpServerSpec``. Reject it with a clear error; otherwise
    return the input narrowed to ``list[HttpServerSpec]`` (no bare names remain).

    Operators declaring an http_server must give a full ``HttpServerSpec`` (verbatim,
    unattenuated). Names-only is an agent-authoring convenience, keyed on the actor.
    """
    if http_servers is None:
        return None
    bare = [s for s in http_servers if isinstance(s, str)]
    if bare:
        raise ForbiddenError(
            "names-only http_servers require an acting agent to resolve against; "
            f"the operator path must declare full HttpServerSpec objects (got bare names {bare!r})",
        )
    return [s for s in http_servers if isinstance(s, HttpServerSpec)]


async def _validate_script_surface(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    script: str,
    tools: list[ToolSpec],
) -> None:
    """Create-time validation of a workflow ``script`` (#1285).

    Two halves of one check, run at author time so a structural or surface-drift
    failure surfaces as a :class:`ValidationError` (a 4xx, turned into a clean
    model-visible result by the tool dispatch layer) instead of a failed *run*:

    * **Structural + script-local surface** — compile, top-level
      ``async def main(input)``, and every string-literal ``tool("X")`` covered by
      the declared ``tools``. Pure/AST (no pool); see
      :func:`aios.workflows.script_validation.validate_workflow_script`.
    * **Named-agent surface union** — for every string-literal
      ``agent(agent_id="A")``, the run's declared surface must cover that child
      agent's declared **tool** surface, or the #794 ``agent ∩ run`` clamp would
      silently strip the child's tools at launch (the omitted-DELETE / omitted-PATCH
      production incidents an application workflow recorded). Resolution needs the
      pool, so it lives here, not in the pure validator.

      **Chosen depth (documented in the PR):** resolve the child agent and require
      the declared **tools** to be a superset of the child agent's declared tools. A
      literal ``agent_id`` that does **not** resolve to a live same-account agent is
      **skipped** (no false rejection) — such a call fails loud at run time as
      ``agent_not_found``, which is a *different* failure class than surface drift and
      out of scope for this create-time check. MCP / http-server union for child
      agents is left to a follow-up; tool union covers the recorded incidents.
    """
    # Structural + script-local tool surface (raises ValidationError on failure).
    validate_workflow_script(script, tools)

    required_agent_ids = extract_required_agent_ids(script)
    if not required_agent_ids:
        return
    declared = declared_tool_names(tools)
    for agent_id in sorted(required_agent_ids):
        try:
            child = await agents_service.get_agent(pool, agent_id, account_id=account_id)
        except NotFoundError:
            # Un-resolvable (absent / archived / cross-account) — not a surface-drift
            # failure; it fails loud at run time as ``agent_not_found``. Skip.
            continue
        child_tool_names = declared_tool_names([t for t in child.tools if t.enabled])
        missing = sorted(child_tool_names - declared)
        if missing:
            joined = ", ".join(repr(m) for m in missing)
            raise ValidationError(
                f"workflow script invokes agent(agent_id={agent_id!r}) whose declared "
                f"tool(s) {joined} are not in the workflow's declared tool surface; the "
                "child surface is agent ∩ run (#794) — under-declaring silently strips "
                "those tools from the child at launch. Add them to the workflow's `tools`",
                detail={"agent_id": agent_id, "missing_tools": missing},
            )


async def create_workflow(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    name: str,
    script: str,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    output_model: str | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerRef] | None = None,
    creator_session_id: str | None = None,
) -> Workflow:
    """Create a workflow definition.

    **Create-time attenuation:** when ``creator_session_id`` is set (an agent authoring
    the workflow), the declared surface (``tools``/``mcp_servers``/``http_servers``) must
    be a subset of the creating agent's own — an agent cannot grant a workflow a tool or
    server it does not itself have; a breach raises :class:`ForbiddenError`. http servers
    are admitted by identity (name + base_url) and their routes inherited launcher-frozen
    into storage; a names-only entry (``["davenant"]``, #953) is resolved against the
    creating agent first. With no creator (the HTTP/operator path) any surface may be
    declared verbatim, account-scoped — but names-only sugar is unavailable there (no
    acting agent to resolve against).
    """
    # Create-time validation (#1285): compile + `async def main(input)` + the declared
    # surface must cover every literal tool("…")/agent(agent_id="…") the script uses.
    # Runs on EVERY path (agent author + HTTP/operator), against the declared `tools`.
    await _validate_script_surface(pool, account_id=account_id, script=script, tools=tools or [])
    effective: Surface | None = None
    operator_http: list[HttpServerSpec] | None = None
    if creator_session_id is not None:
        effective = await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=creator_session_id,
            tools=tools or [],
            mcp_servers=mcp_servers or [],
            http_servers=http_servers or [],
        )
    else:
        operator_http = _reject_operator_path_names_only(http_servers)
    # Agent-authored: store the agent's launcher-frozen http routes (inherited by identity).
    # Operator path: store the declared http servers verbatim (bare names already rejected).
    http_to_store = effective.http_servers if effective is not None else operator_http
    async with pool.acquire() as conn:
        return await wf_queries.insert_workflow(
            conn,
            account_id=account_id,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            output_model=output_model,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_to_store,
        )


async def update_workflow(
    pool: asyncpg.Pool[Any],
    workflow_id: str,
    *,
    account_id: str,
    expected_version: int,
    name: str | None = None,
    script: str | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    output_model: str | None = None,
    description: str | None = None,
    tools: list[ToolSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerRef] | None = None,
    actor_session_id: str | None = None,
) -> Workflow:
    """Update a workflow in place (optimistic concurrency on ``expected_version``).

    **Update-time attenuation:** when ``actor_session_id`` is set (an agent editing the
    workflow), the **merged final surface** must be a subset of the acting agent's own —
    checked even when the update doesn't touch the surface fields, because an agent that
    rewrites the *script* of a workflow with a broader surface would wield that surface
    through it. An agent may only update a workflow whose resulting surface it could have
    declared itself. With no actor (the HTTP/operator path) anything may be updated.

    In-flight runs are unaffected: a run snapshots ``script`` + the declared surface at
    launch; only runs created after the update see the new definition.

    http servers are admitted by identity and their routes inherited launcher-frozen into
    storage (mirroring create); a names-only entry (#953) is resolved against the acting
    agent. The operator path stores declared http verbatim and rejects names-only sugar
    (no acting agent to resolve against).
    """
    # Create-time validation (#1285) applies on UPDATE too. Runs whenever the update
    # touches EITHER the script or the tool surface (``script is not None or tools is
    # not None``): a tools-only update that narrows away a tool the STORED (unchanged)
    # script still calls is itself surface drift, so re-validating only on a script
    # change would leave that hole open. Validated against the EFFECTIVE merged surface
    # — the new ``script`` if given else the stored body, against the new ``tools`` if
    # given else the stored tools. Fetch the current definition once here; the actor
    # branch below re-reads under the version pin for its attenuation check.
    if script is not None or tools is not None:
        current_for_validation = await get_workflow(pool, workflow_id, account_id=account_id)
        await _validate_script_surface(
            pool,
            account_id=account_id,
            script=script if script is not None else current_for_validation.script,
            tools=tools if tools is not None else current_for_validation.tools,
        )
    effective: Surface | None = None
    operator_http: list[HttpServerSpec] | None = None
    if actor_session_id is None:
        operator_http = _reject_operator_path_names_only(http_servers)
    if actor_session_id is not None:
        current = await get_workflow(pool, workflow_id, account_id=account_id)
        if current.version != expected_version:
            # Pin the attenuation read to the caller's token. Without this, an actor
            # could send a FUTURE token: attenuation passes against today's surface,
            # a concurrent update broadens it and bumps to exactly that token, and the
            # optimistic UPDATE then matches — landing the actor's script on a surface
            # that was never checked. With the pin, the UPDATE's ``WHERE version``
            # guarantees nothing changed between this read and the write.
            raise ConflictError(
                f"version mismatch: expected {expected_version}, current is {current.version}",
                detail={
                    "expected": expected_version,
                    "current": current.version,
                    "id": workflow_id,
                },
            )
        # ``current.http_servers`` is the already-resolved stored spec; ``list(...)``
        # widens it to ``list[HttpServerRef]`` for the (str | spec)-typed parameter.
        merged_http: list[HttpServerRef] = (
            http_servers if http_servers is not None else list(current.http_servers)
        )
        effective = await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=actor_session_id,
            tools=tools if tools is not None else current.tools,
            mcp_servers=mcp_servers if mcp_servers is not None else current.mcp_servers,
            http_servers=merged_http,
        )
    # Agent-actor touching http: store the inherited launcher-frozen routes. Operator path,
    # or an edit that didn't touch http (``http_servers is None`` → query preserves current):
    # pass the original argument through unchanged (operator bare names already rejected).
    http_to_store = (
        effective.http_servers
        if (effective is not None and http_servers is not None)
        else operator_http
    )
    async with pool.acquire() as conn:
        return await wf_queries.update_workflow(
            conn,
            workflow_id,
            account_id=account_id,
            expected_version=expected_version,
            name=name,
            script=script,
            input_schema=input_schema,
            output_schema=output_schema,
            output_model=output_model,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_to_store,
        )


async def get_workflow(pool: asyncpg.Pool[Any], workflow_id: str, *, account_id: str) -> Workflow:
    async with pool.acquire() as conn:
        return await wf_queries.get_workflow(conn, workflow_id, account_id=account_id)


async def archive_workflow(
    pool: asyncpg.Pool[Any], workflow_id: str, *, account_id: str
) -> Workflow:
    async with pool.acquire() as conn:
        return await wf_queries.archive_workflow(conn, workflow_id, account_id=account_id)


async def unarchive_workflow(
    pool: asyncpg.Pool[Any], workflow_id: str, *, account_id: str
) -> Workflow:
    async with pool.acquire() as conn:
        return await wf_queries.unarchive_workflow(conn, workflow_id, account_id=account_id)


async def list_workflows(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Workflow]:
    async with pool.acquire() as conn:
        return await wf_queries.list_workflows(
            conn, account_id=account_id, limit=limit, after=after, name=name
        )


async def get_workflow_version(
    pool: asyncpg.Pool[Any], workflow_id: str, version: int, *, account_id: str
) -> WorkflowVersion:
    async with pool.acquire() as conn:
        return await wf_queries.get_workflow_version(
            conn, workflow_id, version, account_id=account_id
        )


async def list_workflow_versions(
    pool: asyncpg.Pool[Any],
    workflow_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: int | None = None,
) -> list[WorkflowVersion]:
    async with pool.acquire() as conn:
        return await wf_queries.list_workflow_versions(
            conn, workflow_id, limit=limit, after=after, account_id=account_id
        )


# ─── runs ────────────────────────────────────────────────────────────────────


def _wall_clock_ms(run: WfRun) -> int | None:
    """Wall-clock span (ms) for a TERMINAL run; ``None`` while it's still live (#1324).

    A terminal run's ``updated_at`` is its completion instant, so ``updated_at -
    created_at`` is its true wall-clock. A non-terminal run's ``updated_at`` is a
    moving "last touched" stamp — surfacing a span off it would be a misleading
    partial, so we report explicit ``None`` (cannot-determine) instead.
    """
    if run.status not in TERMINAL_RUN_STATUSES:
        return None
    return max(0, round((run.updated_at - run.created_at).total_seconds() * 1000))


def _run_usage(run: WfRun, children: wf_queries.RunChildrenUsage) -> WfRunUsage:
    """Project a run + its summed child usage into the read-path :class:`WfRunUsage`.

    cost/tokens are the real summed children; ``iteration_count`` is explicit
    ``None`` (no per-run iteration counter exists on any substrate yet — see the
    model docstring); ``wall_clock_ms`` is terminal-only (:func:`_wall_clock_ms`).
    """
    return WfRunUsage(
        cost_microusd=children.cost_microusd,
        input_tokens=children.input_tokens,
        output_tokens=children.output_tokens,
        cache_read_input_tokens=children.cache_read_input_tokens,
        cache_creation_input_tokens=children.cache_creation_input_tokens,
        iteration_count=None,
        wall_clock_ms=_wall_clock_ms(run),
    )


async def get_run(pool: asyncpg.Pool[Any], run_id: str, *, account_id: str) -> WfRun:
    async with pool.acquire() as conn:
        run = await wf_queries.get_wf_run(conn, run_id, account_id=account_id)
        children = await wf_queries.run_children_usage(conn, run.id, account_id=account_id)
        return run.model_copy(update={"usage": _run_usage(run, children)})


async def archive_run(pool: asyncpg.Pool[Any], run_id: str, *, account_id: str) -> WfRun:
    """Archive a terminal run (the run-side analog of ``archive_workflow``).

    Refuses non-terminal runs (``ConflictError``), sets ``archived_at`` on terminal
    ones, and drops them from the default ``list_runs`` while keeping them fetchable
    by id (and keeping their journal). Returns the archived run with its usage
    roll-up populated, matching ``get_run``'s public read shape.
    """
    async with pool.acquire() as conn:
        run = await wf_queries.archive_run(conn, run_id, account_id=account_id)
        children = await wf_queries.run_children_usage(conn, run.id, account_id=account_id)
        return run.model_copy(update={"usage": _run_usage(run, children)})


async def list_runs(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    workflow_id: str | None = None,
    status: str | None = None,
    parent_run_id: str | None = None,
    launcher_session_id: str | None = None,
) -> list[WfRun]:
    async with pool.acquire() as conn:
        runs = await wf_queries.list_wf_runs(
            conn,
            account_id=account_id,
            limit=limit,
            after=after,
            workflow_id=workflow_id,
            status=status,
            parent_run_id=parent_run_id,
            launcher_session_id=launcher_session_id,
        )
        # Enrich the whole page in ONE batched aggregate (no N+1) so list_runs
        # carries the same per-run usage substrate as get_run (#1324).
        usage_by_run = await wf_queries.runs_children_usage(
            conn, [r.id for r in runs], account_id=account_id
        )
        return [r.model_copy(update={"usage": _run_usage(r, usage_by_run[r.id])}) for r in runs]


async def list_run_events(
    pool: asyncpg.Pool[Any],
    run_id: str,
    *,
    account_id: str,
    after_seq: int = 0,
    limit: int = 200,
) -> list[WfRunEvent]:
    async with pool.acquire() as conn:
        return await wf_queries.list_run_events_scoped(
            conn, run_id, account_id=account_id, after_seq=after_seq, limit=limit
        )


async def resume_gate_by_nonce(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    account_id: str,
    gate_nonce: str,
    result: Any,
    resumer_session_id: str | None = None,
) -> WfRun:
    """Deliver an external resume to the gate identified by its capability ``nonce``.

    The HTTP-facing gate resume: account-scope the run first (the internal
    :func:`aios.workflows.service.resume_gate` keys off ``call_key`` and does NOT
    scope, so it must never be the HTTP path), then resolve ``gate_nonce`` →
    ``call_key`` by scanning the run's **open** gate ``call_started`` events for the
    one whose payload carries that nonce. Only an OPEN gate (no ``call_result`` yet)
    matches — a nonce for an already-resolved gate (or any gate on a terminal run)
    raises ``NotFoundError`` rather than writing an orphaned signal nothing harvests.
    ``resumer_session_id`` applies the agent builtin's launcher attenuation: when
    present, only the session that launched the run may resume its gates. The HTTP
    operator path leaves it unset and remains account-scoped. ``insert_run_signal``
    is idempotent, so a concurrent double-resume of a still-open gate is a no-op.
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_wf_run(conn, run_id, account_id=account_id)  # 404s cross-tenant
        if resumer_session_id is not None and run.launcher_session_id != resumer_session_id:
            raise ForbiddenError(
                "run was not launched by this session; only the launcher can resume its gates",
                detail={"run_id": run_id},
            )
        call_key = await wf_queries.find_open_gate_call_key(conn, run_id, gate_nonce=gate_nonce)
        if call_key is None:
            raise NotFoundError(
                "no open gate matches that nonce on this run", detail={"run_id": run_id}
            )
        await wf_queries.insert_run_signal(
            conn, run_id=run_id, call_key=call_key, kind="gate_resume", result=result
        )
    await defer_run_wake(run_id)
    return run


async def cancel_run(
    pool: asyncpg.Pool[Any],
    *,
    run_id: str,
    account_id: str,
    reason: Any = None,
    canceller_session_id: str | None = None,
) -> WfRun:
    """Request cancellation of a run, returning it (still in its pre-cancel status).

    Signal-driven, mirroring :func:`resume_gate_by_nonce`: record a ``cancel``
    side-marker and wake the run; the next ``run_workflow_step`` harvests it under
    the lock and finalizes ``cancelled`` (so the journal keeps a single writer and
    any live ``/stream`` closes on the terminal ``run_completed``). The flip lands
    on the wake, so the returned run still shows its current status — as gate-resume
    returns a still-``suspended`` run.

    **Cancel-time attenuation:** when ``canceller_session_id`` is set (an agent
    cancelling via the builtin), the run must have been launched by that very session
    — you may cancel what you launched, nothing else; a breach raises
    :class:`ForbiddenError`. With no canceller (the HTTP/operator path) any
    account-scoped run may be cancelled.

    Idempotent: an already-terminal run is returned unchanged, with no signal or
    wake. A cross-tenant / missing run 404s (via :func:`get_wf_run`).
    """
    async with pool.acquire() as conn:
        run = await wf_queries.get_wf_run(conn, run_id, account_id=account_id)  # 404s cross-tenant
        if canceller_session_id is not None and run.launcher_session_id != canceller_session_id:
            raise ForbiddenError(
                "run was not launched by this session; only the operator can cancel it",
                detail={"run_id": run_id},
            )
        if run.status in TERMINAL_RUN_STATUSES:
            return run  # already terminal — nothing to cancel
        await wf_queries.insert_run_signal(
            conn,
            run_id=run_id,
            call_key=wf_queries.CANCEL_SIGNAL_CALL_KEY,
            kind="cancel",
            result=reason,
        )
    await defer_run_wake(run_id)
    return run
