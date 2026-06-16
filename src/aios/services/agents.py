"""Business logic for agents.

Agents are versioned: every update creates a new immutable version.
The ``agents`` table holds the latest config; ``agent_versions`` stores
the full history.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db import queries
from aios.models.agents import (
    Agent,
    AgentVersion,
    HttpServerSpec,
    McpServerSpec,
    PermissionPolicy,
    ToolSpec,
    resolve_mcp_permission,
)
from aios.models.skills import AgentSkillRef
from aios.services import skills as skills_service
from aios.workflows.generic_child import GENERIC_CHILD_SYSTEM


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
) -> Agent:
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
) -> Agent:
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


async def _load_for_session_conn(
    conn: asyncpg.Connection[Any], session: Any, *, account_id: str
) -> Agent | AgentVersion:
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
            return AgentVersion(
                agent_id="",
                version=0,
                model=session.model,
                system=GENERIC_CHILD_SYSTEM.format(model=session.model),
                tools=frozen.tools,
                skills=[],
                mcp_servers=frozen.mcp_servers,
                http_servers=frozen.http_servers,
                litellm_extra={},
                window_min=50_000,
                window_max=150_000,
                created_at=session.created_at,
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
        return version.model_copy(update=updates)
    if session.agent_version is not None:
        return await queries.get_agent_version(
            conn, session.agent_id, session.agent_version, account_id=account_id
        )
    return await queries.get_agent(conn, session.agent_id, account_id=account_id)


async def load_for_session(
    pool: asyncpg.Pool[Any],
    session: Any,
    *,
    account_id: str,
    conn: asyncpg.Connection[Any] | None = None,
) -> Agent | AgentVersion:
    """Load the Agent / AgentVersion the harness sees for ``session`` at step time.

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
