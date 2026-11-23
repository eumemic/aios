"""Business logic for agents.

Agents are versioned: every update creates a new immutable version.
The ``agents`` table holds the latest config; ``agent_versions`` stores
the full history.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import litellm

from aios.db import queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.logging import get_logger
from aios.models.agents import (
    Agent,
    AgentBinding,
    AgentCreate,
    AgentVersion,
    GenericChildBinding,
    HttpServerSpec,
    McpPermissionValue,
    McpServerSpec,
    OutputStyle,
    PreemptPolicy,
    SshServerSpec,
    StepSurface,
    ToolSpec,
    resolve_mcp_permission,
)
from aios.models.attenuation import (
    Surface,
    endpoints_of,
    http_server_dominated,
    surface_diff,
    surface_of,
)
from aios.models.skills import AgentSkillRef
from aios.services import skills as skills_service
from aios.services.litellm_params import OPENAI_REQUEST_PARAMS
from aios.services.model_binding_authz import enforce_workflow_binding_privilege
from aios.workflows.generic_child import GENERIC_CHILD_SYSTEM

log = get_logger("aios.services.agents")


def _skill_authority(skills: list[AgentSkillRef]) -> set[str]:
    return {skill.skill_id for skill in skills}


_CREDENTIAL_KEYS = {"headers", "extra_headers", "credentials", "token", "secret", "password"}
_CREDENTIAL_SUFFIXES = ("_key", "_token", "_secret", "_password", "_credentials", "_key_id")


def _is_credential_key(key: str) -> bool:
    """Whether a ``litellm_extra`` key carries (or injects) a credential."""
    lowered = key.lower()
    return lowered in _CREDENTIAL_KEYS or lowered.endswith(_CREDENTIAL_SUFFIXES)


# ``litellm_extra`` keys that only tune the request body and can never change where
# LiteLLM sends it (or with which credential). Everything NOT listed here is treated as
# destination-shaping — deny by default — because LiteLLM reads many routing keys
# (``custom_llm_provider``, ``aws_bedrock_runtime_endpoint``, ``vertex_project``,
# ``api_version``, ``azure_ad_token_provider``, ...) and a projection that named only
# the ones we know of would let an editor keep a holder's credential while adding an
# unknown routing key to redirect it.
_NON_ROUTING_KEYS = frozenset(OPENAI_REQUEST_PARAMS) - {
    "api_version",
    "extra_headers",
    "model",
    "messages",
} | {
    "allowed_openai_params",
    "additional_drop_params",
    "drop_params",
    "max_retries",
    "num_retries",
    "timeout",
    "stream_timeout",
}


# Compared as an alias-insensitive set via ``endpoints_of``.
_ENDPOINT_ALIAS_KEYS = frozenset({"api_base", "base_url"})


def _routing_items(extra: dict[str, Any]) -> frozenset[tuple[str, str]]:
    """Every destination-shaping, non-credential key (and value) in ``extra``."""
    return frozenset(
        (key, repr(value))
        for key, value in extra.items()
        if key not in _NON_ROUTING_KEYS
        and key not in _ENDPOINT_ALIAS_KEYS
        and not _is_credential_key(key)
    )


def _effective_provider(model: str, extra: dict[str, Any]) -> str:
    """The provider LiteLLM resolves for ``model`` given any ``custom_llm_provider``."""
    override = extra.get("custom_llm_provider")
    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model, custom_llm_provider=override if isinstance(override, str) else None
        )
    except Exception:
        # Unresolvable: fall back to the raw inputs so identity still compares exactly.
        return f"unresolved:{model!r}:{override!r}"
    return str(provider)


def _destination(
    model: str, litellm_extra: dict[str, Any] | None
) -> tuple[str | None, str, frozenset[str], frozenset[tuple[str, str]]]:
    """Where LiteLLM actually sends a call: provider, endpoints, and routing keys.

    With no explicit ``api_base``/``base_url`` the model string itself selects the
    host, so the model joins the identity in that case only.
    """
    extra = litellm_extra or {}
    endpoints = endpoints_of(extra)
    return (
        None if endpoints else model,
        _effective_provider(model, extra),
        endpoints,
        _routing_items(extra),
    )


def _endpoint_credential_projection(
    model: str, litellm_extra: dict[str, Any] | None
) -> tuple[tuple[str | None, str, frozenset[str], frozenset[tuple[str, str]]], dict[str, Any]]:
    """The authority-bearing slice of ``litellm_extra``: destination plus all credentials.

    The destination is what LiteLLM resolves, not just ``api_base``: the effective
    provider (model string overridden by ``custom_llm_provider``), every endpoint key
    (``api_base`` and its ``base_url`` alias — LiteLLM's ``completion`` lets
    ``base_url`` override ``api_base``), and every other non-allowlisted key, since
    provider-specific routing keys such as ``aws_bedrock_runtime_endpoint`` redirect
    calls too.
    """
    extra = litellm_extra or {}
    credentials = {key: value for key, value in extra.items() if _is_credential_key(key)}
    return (_destination(model, extra), credentials)


def _enforce_authority_delta(
    *,
    model: str,
    litellm_extra: dict[str, Any],
    skills: list[AgentSkillRef],
    prior: Agent,
    editor: Agent | StepSurface,
) -> None:
    """Authorize non-Surface authority against prior-or-editor, never patch syntax."""
    exceeds: dict[str, list[str]] = {}
    if model not in {prior.model, editor.model}:
        exceeds["model"] = ["model identity"]

    # The endpoint and every credential are one atomic authority: credentials are only
    # safe at the endpoint their holder paired them with. Checking each piece against
    # "prior OR editor" independently would let an editor route the target's secret to
    # the editor's endpoint, so the whole projection must match one holder.
    projection = _endpoint_credential_projection(model, litellm_extra)
    if not projection[1]:
        # No credential to misroute: the routing (endpoints + destination-shaping keys)
        # alone must be one a holder uses. The model is checked separately above.
        routing = projection[0][2:]
        if routing not in {
            _destination(prior.model, prior.litellm_extra)[2:],
            _destination(editor.model, editor.litellm_extra)[2:],
        }:
            exceeds["litellm_extra"] = ["inference endpoint"]
    elif projection not in (
        _endpoint_credential_projection(prior.model, prior.litellm_extra),
        _endpoint_credential_projection(editor.model, editor.litellm_extra),
    ):
        exceeds["litellm_extra"] = ["inference endpoint", "credential"]

    admitted_skills = _skill_authority(prior.skills) | _skill_authority(editor.skills)
    new_skills = _skill_authority(skills) - admitted_skills
    if new_skills:
        exceeds["skills"] = sorted(new_skills)

    if exceeds:
        raise ForbiddenError(
            "agent authority exceeds the acting agent's permissions",
            detail={"exceeds": exceeds},
        )


def _field_paths(old: dict[str, Any], new: dict[str, Any], prefix: str = "") -> list[str]:
    """Return changed paths only; values are deliberately excluded from audit output."""
    paths: list[str] = []
    for key in sorted(old.keys() | new.keys()):
        path = f"{prefix}.{key}" if prefix else key
        before = old.get(key)
        after = new.get(key)
        if isinstance(before, dict) and isinstance(after, dict):
            paths.extend(_field_paths(before, after, path))
        elif before != after:
            paths.append(path)
    return paths


AUTHORITY_FIELDS = (
    "tools",
    "mcp_servers",
    "http_servers",
    "ssh_servers",
    "model",
    "litellm_extra",
    "skills",
)
_AUDIT_EXCLUDE = {"id", "version", "created_at", "updated_at", "archived_at", "created_by"}


def _log_agent_update(current: Agent, updated: Agent, *, editor_session_id: str | None) -> None:
    """Emit the ``agent_updated`` audit event (#1961).

    Records the actor, old/new version, and the changed field *paths* — never values,
    so no system prompt, metadata, header, or ``litellm_extra`` credential reaches the
    log. ``preserved_authority_fields`` lists the authority-bearing fields carried
    forward unchanged from the prior version (preserved, not re-granted by the actor);
    ``changed_authority_fields`` lists those the actor's own authority was checked on.
    """
    changed_paths = _field_paths(
        current.model_dump(mode="json", exclude=_AUDIT_EXCLUDE),
        updated.model_dump(mode="json", exclude=_AUDIT_EXCLUDE),
    )
    changed_roots = {path.split(".", 1)[0] for path in changed_paths}
    log.info(
        "agent_updated",
        actor="operator" if editor_session_id is None else "session",
        actor_session_id=editor_session_id,
        agent_id=updated.id,
        old_version=current.version,
        new_version=updated.version,
        changed_paths=changed_paths,
        changed_authority_fields=[f for f in AUTHORITY_FIELDS if f in changed_roots],
        preserved_authority_fields=[f for f in AUTHORITY_FIELDS if f not in changed_roots],
    )


async def _enforce_surface_attenuation(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    actor_session_id: str,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerSpec],
    ssh_servers: list[SshServerSpec],
    prior_surface: Surface | None = None,
) -> StepSurface:
    """Raise ``ForbiddenError`` unless the declared agent surface is admissible against
    the acting (creator/editor) session's agent surface.

    At create time, an agent authored from inside a session may only declare a surface
    ⊆ the creating agent's own. At update time, ``prior_surface`` admits capabilities
    already held by the target as well: each resulting capability must be dominated by
    either the editor or the target's prior surface. Preservation and shrinking therefore
    pass without requiring the editor to hold the target's connectors, while additions
    remain bounded by the editor. The HTTP/operator path passes no actor and skips this.

    Without ``prior_surface``, the predicate is the lattice fixpoint:
    ``clamp(declared, actor) == normalize(declared)`` iff the meet narrowed nothing,
    i.e. ``declared ≤ actor`` on tool membership +
    per-tool permission/transport, MCP server identity, and HTTP server identity
    (name + base_url). A breach raises ``ForbiddenError`` with ``detail=surface_diff``.

    Unlike the workflow author edge, agent ``http_servers`` are full ``HttpServerSpec``
    objects (no names-only sugar), and an agent stores its declared surface verbatim, so
    this is a predicate — it raises on a breach and otherwise returns the editor's
    :class:`StepSurface` (reused by the non-Surface authority-delta check); the caller
    stores the originally-declared surface unchanged. On update, http servers are
    additionally checked route-for-route (:func:`http_server_dominated`), because the
    identity-only ``surface_diff`` would otherwise admit an added route or widened verb
    on a server the editor or target already holds.
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
    declared = Surface(tools, mcp_servers, http_servers, ssh_servers)
    expected = attenuation_service.normalize(declared)
    effective = attenuation_service.clamp(declared, surface_of(agent))
    if prior_surface is not None:
        preserved = attenuation_service.clamp(declared, prior_surface)
        effective = Surface(
            effective.tools + preserved.tools,
            effective.mcp_servers + preserved.mcp_servers,
            effective.http_servers + preserved.http_servers,
            effective.ssh_servers + preserved.ssh_servers,
        )
    diff = surface_diff(expected, effective)
    if prior_surface is not None:
        # #1961: ``surface_diff`` keys http servers on identity only, but an agent
        # stores its declared servers verbatim — so on update, every declared http
        # server must also be route-for-route within the editor's or the target's
        # prior grant (no added route, widened verb, or relaxed gate).
        holders = {srv.base_url: srv for srv in agent.http_servers}
        prior_holders = {srv.base_url: srv for srv in prior_surface.http_servers}
        widened = [
            srv.name
            for srv in declared.http_servers
            if not any(
                holder is not None and http_server_dominated(srv, holder)
                for holder in (holders.get(srv.base_url), prior_holders.get(srv.base_url))
            )
        ]
        for name in widened:
            if name not in diff.get("http_servers", []):
                diff.setdefault("http_servers", []).append(name)
    if diff:
        raise ForbiddenError(
            "agent surface exceeds the acting agent's permissions",
            detail={"exceeds": diff},
        )
    return agent


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
    ssh_servers: list[SshServerSpec] | None = None,
    description: str | None,
    metadata: dict[str, Any],
    litellm_extra: dict[str, Any] | None = None,
    window_min: int,
    window_max: int,
    preempt_policy: PreemptPolicy = "wait",
    output_style: OutputStyle = "default",
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
        # model. The operator/HTTP path (no creator) skips this.
        enforce_workflow_binding_privilege(model)
        await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=creator_session_id,
            tools=tools,
            mcp_servers=mcp_servers or [],
            http_servers=http_servers or [],
            ssh_servers=ssh_servers or [],
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
            ssh_servers=ssh_servers or [],
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra or {},
            window_min=window_min,
            window_max=window_max,
            preempt_policy=preempt_policy,
            output_style=output_style,
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


async def resolve_role(pool: asyncpg.Pool[Any], role: str, *, account_id: str) -> Agent:
    """Resolve ``role`` to the single live agent of ``account_id`` holding it (#1940).

    A role is held by the live agent whose ``name`` or ``metadata.role`` equals it.
    Resolution is read at call time, so callers address the role and never cache
    the agent id — an archive + re-create (re-spawn) moves the role to the new id.

    Fails loud, never empty: no holder is a ``NotFoundError`` ("no live binding
    for role …"); more than one is a ``ConflictError`` listing the candidate ids.
    """
    async with pool.acquire() as conn:
        candidates = await queries.list_agents_for_role(conn, role, account_id=account_id)
    if not candidates:
        raise NotFoundError(f"no live binding for role {role!r}", detail={"role": role})
    if len(candidates) > 1:
        ids = [a.id for a in candidates]
        raise ConflictError(
            f"role {role!r} is ambiguous: held by {len(ids)} live agents {ids} "
            "(match on name or metadata.role); give it to exactly one agent",
            detail={"role": role, "agent_ids": ids},
        )
    return candidates[0]


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
    ssh_servers: list[SshServerSpec] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    litellm_extra: dict[str, Any] | None = None,
    window_min: int | None = None,
    window_max: int | None = None,
    preempt_policy: PreemptPolicy | None = None,
    output_style: OutputStyle | None = None,
    editor_session_id: str | None = None,
) -> Agent:
    """Update an agent in place, creating a new immutable version (optimistic
    concurrency on ``expected_version``).

    **Update-time attenuation:** when ``editor_session_id`` is set, the merged final
    surface may contain only capabilities held by the editor or already held by the
    target's current version. Thus preserving or shrinking existing surface is allowed,
    while additions remain bounded by the editor; a breach raises :class:`ForbiddenError`.
    With no editor (the HTTP/operator path) anything may be updated.
    """
    if editor_session_id is not None:
        # #1636: reject privileged bindings before any database access.
        enforce_workflow_binding_privilege(model)
    current = await get_agent(pool, agent_id, account_id=account_id)
    if editor_session_id is not None:
        # #1961: the authority delta is authorized against ``current`` as the prior
        # version, so it must BE the version the writer will replace. A token that
        # does not match the row we authorized against is a lost-update race; the
        # writer's ``WHERE version = expected`` then pins the base to ``current``.
        if current.version != expected_version:
            raise ConflictError(
                f"agent {agent_id} version mismatch: expected {expected_version}, "
                f"current is {current.version}",
                detail={"id": agent_id, "current_version": current.version},
            )
        editor = await _enforce_surface_attenuation(
            pool,
            account_id=account_id,
            actor_session_id=editor_session_id,
            tools=tools if tools is not None else current.tools,
            mcp_servers=mcp_servers if mcp_servers is not None else current.mcp_servers,
            http_servers=http_servers if http_servers is not None else current.http_servers,
            ssh_servers=ssh_servers if ssh_servers is not None else current.ssh_servers,
            prior_surface=surface_of(current),
        )
        _enforce_authority_delta(
            model=model if model is not None else current.model,
            litellm_extra=(litellm_extra if litellm_extra is not None else current.litellm_extra),
            skills=skills if skills is not None else current.skills,
            prior=current,
            editor=editor,
        )
    skills_json_str: str | None = None
    if skills is not None:
        resolved = await skills_service.resolve_skill_refs(pool, skills, account_id=account_id)
        skills_json_str = skills_service.serialize_skills_for_snapshot(skills, resolved)
    async with pool.acquire() as conn:
        updated = await queries.update_agent(
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
            ssh_servers=ssh_servers,
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra,
            window_min=window_min,
            window_max=window_max,
            preempt_policy=preempt_policy,
            output_style=output_style,
            account_id=account_id,
        )
    _log_agent_update(current, updated, editor_session_id=editor_session_id)
    return updated


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
    :class:`StepSurface`, carrying exactly the harness-consumed fields.

    A missing field here is a compile-loud ``StepSurface`` construction error,
    not a silent structural-overlap drift — the projection's drift surface is
    mypy, not runtime.

    ``description`` (the auto-review checker's scope input) exists only on the
    live ``Agent`` read-model — ``agent_versions`` never snapshotted it — so a
    version-pinned surface carries ``None`` and the checker simply omits the
    scope line. Floating sessions (the jarbot shape) always resolve the live
    ``Agent`` and get the current profile.
    """
    return StepSurface(
        tools=agent.tools,
        mcp_servers=agent.mcp_servers,
        http_servers=agent.http_servers,
        ssh_servers=agent.ssh_servers,
        model=agent.model,
        system=agent.system,
        skills=agent.skills,
        litellm_extra=agent.litellm_extra,
        window_min=agent.window_min,
        window_max=agent.window_max,
        preempt_policy=agent.preempt_policy,
        output_style=agent.output_style,
        description=agent.description if isinstance(agent, Agent) else None,
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
    # Every spawn edge that persists a frozen surface must consume it here.  In
    # particular, call_agent children are ordinary sessions (no parent_run_id),
    # so keying this branch only on workflow lineage would silently re-expand
    # their authority to the live agent surface on the first wake.
    if session.surface_frozen or session.parent_run_id is not None:
        frozen = await queries.get_session_frozen_surface(conn, session.id, account_id=account_id)
        if frozen is None:
            raise RuntimeError(
                f"spawned session {session.id} has no frozen surface snapshot "
                "(spawn lineage set but surface_frozen is false)"
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
                ssh_servers=frozen.ssh_servers,
                litellm_extra={},
                window_min=defaults["window_min"].default,
                window_max=defaults["window_max"].default,
                preempt_policy=defaults["preempt_policy"].default,
                output_style=defaults["output_style"].default,
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
            "ssh_servers": frozen.ssh_servers,
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


def effective_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> McpPermissionValue:
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
