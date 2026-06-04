"""Agent CRUD endpoints with versioning and version history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.agents import Agent, AgentCreate, AgentUpdate, AgentVersion
from aios.models.common import ListResponse
from aios.models.pagination import page_cursor
from aios.services import agents as service

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post("", operation_id="create_agent", status_code=status.HTTP_201_CREATED)
async def create(body: AgentCreate, pool: PoolDep, account_id: AccountIdDep) -> Agent:
    """Create a new agent at version 1.

    Subsequent updates produce new immutable versions in the history; see
    ``update_agent`` and ``list_agent_versions``.
    """
    return await service.create_agent(
        pool,
        name=body.name,
        model=body.model,
        system=body.system,
        tools=body.tools,
        skills=body.skills,
        mcp_servers=body.mcp_servers,
        http_servers=body.http_servers,
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
    account_id: AccountIdDep,
    cursor: str | None = None,
    name: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
) -> ListResponse[Agent]:
    """List agents (latest version of each), newest first, excluding archived.

    First page: optional ``name`` filter (exact match) + ``?limit=``.
    Subsequent pages: ``?cursor=<next_cursor>`` (carries the filter; no other
    params accepted alongside it).
    """
    st = page_cursor(cursor, {"name": name, "limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = st.limit if st is not None else (limit if limit is not None else 50)
    if st is not None:
        name = st.filters.get("name")
    items = await service.list_agents(
        pool, limit=page_limit + 1, after=after, name=name, account_id=account_id
    )
    return ListResponse[Agent].paginate(
        items, page_limit, cursor=lambda x: x.id, filters={"name": name}
    )


@router.get("/{agent_id}", operation_id="get_agent")
async def get(agent_id: str, pool: PoolDep, account_id: AccountIdDep) -> Agent:
    """Fetch one agent by id, returning the latest version's config."""
    return await service.get_agent(pool, agent_id, account_id=account_id)


@router.put("/{agent_id}", operation_id="update_agent")
async def update(
    agent_id: str, body: AgentUpdate, pool: PoolDep, account_id: AccountIdDep
) -> Agent:
    """Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).
    """
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
        http_servers=body.http_servers,
        description=body.description,
        metadata=body.metadata,
        litellm_extra=body.litellm_extra,
        window_min=body.window_min,
        window_max=body.window_max,
        account_id=account_id,
    )


@router.delete("/{agent_id}", operation_id="archive_agent", status_code=status.HTTP_204_NO_CONTENT)
async def archive(agent_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive an agent: sets ``archived_at`` and hides it from default lists.

    The row and all version history persist; sessions referencing the agent
    continue to function. There is no API surface to un-archive currently.
    """
    await service.archive_agent(pool, agent_id, account_id=account_id)


@router.get("/{agent_id}/versions", operation_id="list_agent_versions")
async def list_versions(
    agent_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
) -> ListResponse[AgentVersion]:
    """List historical versions of an agent, newest first.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Each
    version is a complete snapshot of the agent's config at creation time.
    """
    st = page_cursor(cursor, {"limit": limit})
    after = int(st.cursor) if st is not None else None
    page_limit = st.limit if st is not None else (limit if limit is not None else 50)
    items = await service.list_agent_versions(
        pool, agent_id, limit=page_limit + 1, after=after, account_id=account_id
    )
    return ListResponse[AgentVersion].paginate(items, page_limit, cursor=lambda x: x.version)


@router.get("/{agent_id}/versions/{version}", operation_id="get_agent_version")
async def get_version(
    agent_id: str, version: int, pool: PoolDep, account_id: AccountIdDep
) -> AgentVersion:
    """Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.
    """
    return await service.get_agent_version(pool, agent_id, version, account_id=account_id)
