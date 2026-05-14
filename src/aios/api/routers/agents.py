"""Agent CRUD endpoints with versioning and version history."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.agents import Agent, AgentCreate, AgentUpdate, AgentVersion
from aios.models.common import ListResponse
from aios.services import agents as service

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post("", operation_id="create_agent", status_code=status.HTTP_201_CREATED)
async def create(body: AgentCreate, pool: PoolDep, _auth: AuthDep) -> Agent:
    """Create a new agent at version 1.

    Subsequent updates produce new immutable versions in the history; see
    ``update_agent`` and ``list_agent_versions``.
    """
    account_id, _, _ = _auth
    return await service.create_agent(
        pool,
        name=body.name,
        model=body.model,
        system=body.system,
        tools=body.tools,
        skills=body.skills,
        mcp_servers=body.mcp_servers,
        description=body.description,
        metadata=body.metadata,
        litellm_extra=body.litellm_extra,
        window_min=body.window_min,
        window_max=body.window_max,
        account_id=account_id,
    )


@router.get("", operation_id="list_agents")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> ListResponse[Agent]:
    """List agents (latest version of each), newest first, excluding archived.

    Cursor pagination: pass ``after`` from a previous response's
    ``next_after`` to get the next page. Optional ``name`` filter matches
    exactly.
    """
    account_id, _, _ = _auth
    items = await service.list_agents(
        pool, limit=limit, after=after, name=name, account_id=account_id
    )
    return ListResponse[Agent](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{agent_id}", operation_id="get_agent")
async def get(agent_id: str, pool: PoolDep, _auth: AuthDep) -> Agent:
    """Fetch one agent by id, returning the latest version's config."""
    account_id, _, _ = _auth
    return await service.get_agent(pool, agent_id, account_id=account_id)


@router.put("/{agent_id}", operation_id="update_agent")
async def update(agent_id: str, body: AgentUpdate, pool: PoolDep, _auth: AuthDep) -> Agent:
    """Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).
    """
    account_id, _, _ = _auth
    return await service.update_agent(
        pool,
        agent_id,
        expected_version=body.version,
        name=body.name,
        model=body.model,
        system=body.system,
        tools=body.tools,
        skills=body.skills,
        mcp_servers=body.mcp_servers,
        description=body.description,
        metadata=body.metadata,
        litellm_extra=body.litellm_extra,
        window_min=body.window_min,
        window_max=body.window_max,
        account_id=account_id,
    )


@router.delete("/{agent_id}", operation_id="archive_agent", status_code=status.HTTP_204_NO_CONTENT)
async def archive(agent_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    """Archive an agent: sets ``archived_at`` and hides it from default lists.

    The row and all version history persist; sessions referencing the agent
    continue to function. There is no API surface to un-archive currently.
    """
    account_id, _, _ = _auth
    await service.archive_agent(pool, agent_id, account_id=account_id)


@router.get("/{agent_id}/versions", operation_id="list_agent_versions")
async def list_versions(
    agent_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: int | None = None,
) -> ListResponse[AgentVersion]:
    """List historical versions of an agent, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete snapshot of the agent's config at the time it was created.
    """
    account_id, _, _ = _auth
    items = await service.list_agent_versions(
        pool, agent_id, limit=limit, after=after, account_id=account_id
    )
    return ListResponse[AgentVersion](
        data=items,
        has_more=len(items) == limit,
        next_after=str(items[-1].version) if items else None,
    )


@router.get("/{agent_id}/versions/{version}", operation_id="get_agent_version")
async def get_version(agent_id: str, version: int, pool: PoolDep, _auth: AuthDep) -> AgentVersion:
    """Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.
    """
    account_id, _, _ = _auth
    return await service.get_agent_version(pool, agent_id, version, account_id=account_id)
