"""Business logic for agents.

Agents are versioned: every update creates a new immutable version.
The ``agents`` table holds the latest config; ``agent_versions`` stores
the full history.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import ForbiddenError
from aios.models.agents import (
    Agent,
    AgentBinding,
    AgentCreate,
    AgentVersion,
    GenericChildBinding,
    HttpServerSpec,
    McpServerSpec,
    PermissionPolicy,
    StepSurface,
    ToolSpec,
    resolve_mcp_permission,
)
from aios.models.attenuation import Surface, surface_diff, surface_of
from aios.models.skills import AgentSkillRef
from aios.services import skills as skills_service
from aios.services.model_binding_authz import enforce_workflow_binding_privilege
from aios.workflows.generic_child import GENERIC_CHILD_SYSTEM


async def _enforce_surface_attenuation(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    actor_session_id: str,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerSpec],
) -> None:
    """Raise ``ForbiddenError`` unless the declared agent surface is admissible against
    the acting (creator/editor) session's agent surface.

    The create-time clamp the issue (T1) calls for: ``create_agent`` authored from
    inside a session may only declare a surface ⊆ the creating agent's own — an agent
    cannot grant a child agent a tool, MCP server, or HTTP server it does not itself
    hold. ``update_agent`` runs the same check on the merged final surface against the
    editing agent. This is the model-tool plane's analog of the ``create_workflow``
    surface clamp (``services.workflows._enforce_surface_attenuation``); the HTTP/operator
    path passes no actor and skips this entirely (declares verbatim, account-scoped).

    The predicate is the lattice fixpoint: ``clamp(declared, actor) == normalize(declared)``
    iff the meet narrowed nothing, i.e. ``declared ≤ actor`` on tool membership +
    per-tool permission/transport, MCP server identity, and HTTP server identity
    (name + base_url). A breach raises ``ForbiddenError`` with ``detail=surface_diff``.

    Unlike the workflow author edge, agent ``http_servers`` are full ``HttpServerSpec``
    objects (no names-only sugar), and an agent stores its declared surface verbatim, so
    this is a pure predicate — it raises on a breach and returns ``None``; the caller
    stores the originally-declared surface unchanged.
    """
    # Imported lazily to break the import cycle: ``services.sessions`` imports this
    # module at top level, and ``services.attenuation`` pulls in ``tools.registry``
    # (hence the whole tools package). ``services.agents`` loads very early in the
    # sandbox/session chains, so a top-level import of either here would partially
    # initialize the tools package mid-cycle. Deferring keeps this module's import
    # graph at the bottom (it only needs them at call time).
    from aios.services import attenuation as attenuation_service
    from aios.services import sessions as sessions_service

    session = await sessions_service.get_session_basic(
        pool, actor_session_id, account_id=account_id
    )
    agent = await load_for_session(pool, session, account_id=account_id)
    declared = Surface(tools, mcp_servers, http_servers)
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
            "agent surface exceeds the acting agent's permissions",
            detail={"exceeds": surface_diff(expected, effective)},
        )


async def create_agent(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    name: str,
    model: str,
    system: str,
    tools: list[ToolSpec],
    skills: list[AgentSkillRef] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    description: str | None,
    metadata: dict[str, Any],
    litellm_extra: dict[str, Any] | None = None,
    window_min: int,
    window_max: int,
    creator_session_id: str | None = None,
) -> Agent:
    """Create a new agent at version 1.

    **Create-time attenuation (T1):** when ``creator_session_id`` is set (an agent
    authoring another agent from inside a session, via the ``create_agent`` model
    tool), the declared surface (``tools``/``mcp_servers``/``http_servers``) must be a
    subset of the creating agent's own — an agent cannot grant a child agent a tool,
    MCP server, or HTTP server it does not itself hold; a breach raises
    :class:`ForbiddenError`. With no creator (the HTTP/operator path) any surface may be
    declared verbatim, account-scoped. This is the authoring bound; the #794/#823
    spawn-edge reclamp (``workflows/step.py``) independently re-clamps the agent's
    surface ⊆ the run at spawn time and is unchanged by this path.
    """
    if creator_session_id is not None:
        # #1636: the model-binding privilege. A ``creator_session_id`` IS a
        # self-authoring (non-operator) principal — it may not bind a ``workflow:``
        # model. The operator/HTTP path (no creator) passes ``is_operator=True``.
        enforce_workflow_binding_privilege(model, is_operator=False)
        await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=creator_session_id,
            tools=tools,
            mcp_servers=mcp_servers or [],
            http_servers=http_servers or [],
        )
    skill_refs = skills or []
    resolved = await skills_service.resolve_skill_refs(pool, skill_refs, account_id=account_id)
    snapshot_json = skills_service.serialize_skills_for_snapshot(skill_refs, resolved)
    async with pool.acquire() as conn:
        return await queries.insert_agent(
            conn,
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills_json=snapshot_json,
            mcp_servers=mcp_servers or [],
            http_servers=http_servers or [],
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra or {},
            window_min=window_min,
            window_max=window_max,
            account_id=account_id,
        )


async def get_agent(pool: asyncpg.Pool[Any], agent_id: str, *, account_id: str) -> Agent:
    async with pool.acquire() as conn:
        return await queries.get_agent(conn, agent_id, account_id=account_id)


async def list_agents(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> list[Agent]:
    async with pool.acquire() as conn:
        return await queries.list_agents(
            conn, limit=limit, after=after, name=name, account_id=account_id
        )


async def archive_agent(pool: asyncpg.Pool[Any], agent_id: str, *, account_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_agent(conn, agent_id, account_id=account_id)


async def update_agent(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    *,
    account_id: str,
    expected_version: int,
    name: str | None = None,
    model: str | None = None,
    system: str | None = None,
    tools: list[ToolSpec] | None = None,
    skills: list[AgentSkillRef] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    litellm_extra: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
    editor_session_id: str | None = None,
) -> Agent:
    """Update an agent in place, creating a new immutable version (optimistic
    concurrency on ``expected_version``).

    **Update-time attenuation (T1):** when ``editor_session_id`` is set (an agent
    editing another agent via the ``update_agent`` model tool), the **merged final
    surface** must be a subset of the editing agent's own — an agent may only update an
    agent to a surface it could have declared itself; a breach raises
    :class:`ForbiddenError`. Omitted surface fields are merged from the current stored
    version before the check. With no editor (the HTTP/operator path) anything may be
    updated.
    """
    if editor_session_id is not None:
        # #1636: the model-binding privilege, keyed on the editor being a
        # self-authoring (non-operator) principal. ``model is None`` (the field is
        # being preserved) introduces no new binding and is a no-op; a non-None
        # ``workflow:`` model is rejected. The operator/HTTP path skips this.
        enforce_workflow_binding_privilege(model, is_operator=False)
        current = await get_agent(pool, agent_id, account_id=account_id)
        await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=editor_session_id,
            tools=tools if tools is not None else current.tools,
            mcp_servers=mcp_servers if mcp_servers is not None else current.mcp_servers,
            http_servers=http_servers if http_servers is not None else current.http_servers,
        )
    skills_json_str: str | None = None
    if skills is not None:
        resolved = await skills_service.resolve_skill_refs(pool, skills, account_id=account_id)
        skills_json_str = skills_service.serialize_skills_for_snapshot(skills, resolved)
    async with pool.acquire() as conn:
        return await queries.update_agent(
            conn,
            agent_id,
            expected_version=expected_version,
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills_json=skills_json_str,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra,
            window_min=window_min,
            window_max=window_max,
            account_id=account_id,
        )


async def get_agent_version(
    pool: asyncpg.Pool[Any], agent_id: str, version: int, *, account_id: str
) -> AgentVersion:
    async with pool.acquire() as conn:
        return await queries.get_agent_version(conn, agent_id, version, account_id=account_id)


async def validate_pinned_agent_version(
    conn: asyncpg.Connection[Any],
    *,
    agent_id: str | None,
    agent_version: int | None,
    account_id: str,
) -> None:
    """Reject a pinned ``agent_version`` that has no matching ``agent_versions`` row.

    ``agent_version=None`` means "latest" and needs no check. A non-null pin is
    otherwise write-trusted but read-fatal: the bare ``agent_id`` FK doesn't
    constrain the integer, so a bad pin (e.g. a version past the agent's current
    one) is accepted at write time, then the first step's ``load_for_session``
    calls ``get_agent_version``, raises ``NotFoundError``, and the session burns
    its retry budget into the terminal ``errored`` state — unrecoverable, since
    ``agent_versions`` is append-only and the number only ever increments past
    the bad pin. Validating here turns that silent mid-run brick into a clean
    404 at the write the operator initiated. Shared by the session and template
    create/update writers (on a caller ``conn`` so it joins their txn and rolls
    back with them) so every pin path enforces it.
    """
    if agent_version is None:
        return
    # A non-null version always pairs with a non-null agent_id: sessions enforce
    # it via sessions_agent_version_pair_ck (migration 0095); session_templates
    # via the NOT NULL agent_id column (migration 0027).
    assert agent_id is not None
    await queries.get_agent_version(conn, agent_id, agent_version, account_id=account_id)


def _surface_from_agent(agent: Agent | AgentVersion, binding: AgentBinding) -> StepSurface:
    """Project a wire read-model (``Agent``/``AgentVersion``) onto the nominal
    :class:`StepSurface`, carrying exactly the nine harness-consumed fields.

    A missing field here is a compile-loud ``StepSurface`` construction error,
    not a silent structural-overlap drift — the projection's drift surface is
    mypy, not runtime.
    """
    return StepSurface(
        tools=agent.tools,
        mcp_servers=agent.mcp_servers,
        http_servers=agent.http_servers,
        model=agent.model,
        system=agent.system,
        skills=agent.skills,
        litellm_extra=agent.litellm_extra,
        window_min=agent.window_min,
        window_max=agent.window_max,
        binding=binding,
    )


async def _load_for_session_conn(
    conn: asyncpg.Connection[Any], session: Any, *, account_id: str
) -> StepSurface:
    """Resolve a session's effective surface on a caller-supplied ``conn``.

    The body of :func:`load_for_session` — see that function's docstring for the
    branch semantics (frozen-child overlay / pinned version / latest). Kept on a
    supplied ``conn`` so a caller already inside a transaction (``create_run``'s
    launcher read) resolves at the same consistency point without a second acquire.
    """
    if session.parent_run_id is not None:
        frozen = await queries.get_session_frozen_surface(conn, session.id, account_id=account_id)
        if frozen is None:
            raise RuntimeError(
                f"workflow child {session.id} has no frozen surface snapshot "
                "(parent_run_id set but surface_frozen is false)"
            )
        if session.agent_id is None:
            # Generic workflow child: no agent at all. Build the surface from
            # ``AgentCreate``'s field defaults (window_min/window_max) so the
            # old duplicated 50k/150k literals can't silently diverge, and give
            # it an explicit ``generic_child`` binding keyed on its own session
            # — no ``agent_id=""``/``version=0`` sentinel.
            defaults = AgentCreate.model_fields
            return StepSurface(
                model=session.model,
                system=GENERIC_CHILD_SYSTEM.format(model=session.model),
                tools=frozen.tools,
                skills=[],
                mcp_servers=frozen.mcp_servers,
                http_servers=frozen.http_servers,
                litellm_extra={},
                window_min=defaults["window_min"].default,
                window_max=defaults["window_max"].default,
                binding=GenericChildBinding(session_id=session.id),
            )
        version = await queries.get_agent_version(
            conn, session.agent_id, session.agent_version, account_id=account_id
        )
        # #823: read the model identity (litellm_extra, api_base foremost) frozen +
        # clamped at spawn, NOT the live agent version's — replay-sound against a later
        # update_agent, and the spawn-edge clamp's persisted result. The surface
        # (tools/mcp/http) is overlaid from #794's snapshot; both are frozen-once.
        frozen_litellm_extra = await queries.get_session_frozen_litellm_extra(
            conn, session.id, account_id=account_id
        )
        updates: dict[str, Any] = {
            "tools": frozen.tools,
            "mcp_servers": frozen.mcp_servers,
            "http_servers": frozen.http_servers,
            "litellm_extra": frozen_litellm_extra,
        }
        if session.model is not None:
            updates["model"] = session.model
        # An **agented** workflow child keeps an ``agent`` binding on
        # ``(agent_id, version)`` so sibling runs still share the #1391 raw
        # discovery cache (trap 1: two arms, not three).
        overlaid = version.model_copy(update=updates)
        return _surface_from_agent(
            overlaid, AgentBinding(agent_id=overlaid.agent_id, version=overlaid.version)
        )
    if session.agent_version is not None:
        pinned = await queries.get_agent_version(
            conn, session.agent_id, session.agent_version, account_id=account_id
        )
        return _surface_from_agent(
            pinned, AgentBinding(agent_id=pinned.agent_id, version=pinned.version)
        )
    latest = await queries.get_agent(conn, session.agent_id, account_id=account_id)
    return _surface_from_agent(latest, AgentBinding(agent_id=latest.id, version=latest.version))


def tool_cache_binding_id(surface: StepSurface) -> str:
    """Stable identity for the #1391 MCP tool-list cache key.

    The cached tool set is static for a given *binding identity* — the
    precondition that lets sibling sessions of the same agent/version share
    one discovery. A total match on the discriminated ``binding.kind``: an
    ``agent`` binding is agent-level so siblings (latest/pinned/agented child)
    share; a ``generic_child`` has only its own attenuated per-run surface, so
    it keys on its own session (no cross-session sharing — surfaces differ).
    """
    binding = surface.binding
    if binding.kind == "agent":
        return f"{binding.agent_id}:{binding.version}"
    return f"child:{binding.session_id}"


async def load_for_session(
    pool: asyncpg.Pool[Any],
    session: Any,
    *,
    account_id: str,
    conn: asyncpg.Connection[Any] | None = None,
) -> StepSurface:
    """Load the :class:`StepSurface` the harness sees for ``session`` at step time.

    A **workflow child** (``parent_run_id`` set) reads the surface **frozen at spawn**
    (#794: the ``attenuate(agent, run)`` clamp) over its pinned ``AgentVersion`` —
    model/system/skills/litellm_extra preserved, only tools/mcp/http overridden. This is
    the single chokepoint every reader (loop, broker, http_request, prelude,
    compute_awaiting, the context preview) inherits the clamp through, with no per-site
    edit. The meet is **never recomputed** here — the frozen result is read verbatim, so
    replay is stable against later agent-version or operator-default changes. A
    ``parent_run_id`` child whose row is somehow not ``surface_frozen`` **fails closed**
    (never silently falls back to the full agent surface).

    Otherwise: ``session.agent_version is None`` means "latest" — fetches the current
    ``Agent``; an integer pins to a specific ``AgentVersion``.

    Pass ``conn`` to resolve on a caller-supplied connection (e.g. inside an open
    transaction); with no ``conn`` the read self-acquires from ``pool`` as before.
    """
    if conn is not None:
        return await _load_for_session_conn(conn, session, account_id=account_id)
    async with pool.acquire() as acquired:
        return await _load_for_session_conn(acquired, session, account_id=account_id)


def effective_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> PermissionPolicy:
    """Resolved MCP permission with operator-default fallback applied.

    Wraps :func:`aios.models.agents.resolve_mcp_permission` (which
    returns ``None`` when no ``mcp_toolset`` entry matches the server)
    and substitutes ``AIOS_DEFAULT_MCP_PERMISSION_POLICY`` (or
    ``always_ask`` if no operator default is set) so callers see the
    effective policy the dispatcher actually applies — never ``None``.
    """
    perm = resolve_mcp_permission(name, agent_tools)
    if perm is not None:
        return perm
    from aios.config import get_settings

    return get_settings().default_mcp_permission_policy or "always_ask"


async def list_agent_versions(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: int | None = None,
) -> list[AgentVersion]:
    async with pool.acquire() as conn:
        return await queries.list_agent_versions(
            conn, agent_id, limit=limit, after=after, account_id=account_id
        )
