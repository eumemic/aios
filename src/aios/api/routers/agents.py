"""Agent CRUD endpoints with versioning and version history."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.agents import Agent, AgentCreate, AgentUpdate, AgentVersion
from aios.models.common import ListResponse
from aios.services import agents as service

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: AgentCreate, pool: PoolDep, _auth: AuthDep) -> Agent:
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
        window_min=body.window_min,
        window_max=body.window_max,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
    name: str | None = None,
) -> ListResponse[Agent]:
    items = await service.list_agents(pool, limit=limit, after=after, name=name)
    return ListResponse[Agent](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{agent_id}")
async def get(agent_id: str, pool: PoolDep, _auth: AuthDep) -> Agent:
    return await service.get_agent(pool, agent_id)


@router.put("/{agent_id}")
async def update(agent_id: str, body: AgentUpdate, pool: PoolDep, _auth: AuthDep) -> Agent:
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
        window_min=body.window_min,
        window_max=body.window_max,
    )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive(agent_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_agent(pool, agent_id)


@router.get("/{agent_id}/versions")
async def list_versions(
    agent_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: int | None = None,
) -> ListResponse[AgentVersion]:
    items = await service.list_agent_versions(pool, agent_id, limit=limit, after=after)
    return ListResponse[AgentVersion](
        data=items,
        has_more=len(items) == limit,
        next_after=str(items[-1].version) if items else None,
    )


@router.get("/{agent_id}/versions/{version}")
async def get_version(agent_id: str, version: int, pool: PoolDep, _auth: AuthDep) -> AgentVersion:
    return await service.get_agent_version(pool, agent_id, version)
