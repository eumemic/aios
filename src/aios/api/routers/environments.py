"""Environment CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.environments import Environment, EnvironmentCreate, EnvironmentUpdate
from aios.services import environments as service

router = APIRouter(prefix="/v1/environments", tags=["environments"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: EnvironmentCreate, pool: PoolDep, _auth: AuthDep) -> Environment:
    return await service.create_environment(pool, name=body.name, config=body.config)


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Environment]:
    items = await service.list_environments(pool, limit=limit, after=after)
    return ListResponse[Environment](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{env_id}")
async def get(env_id: str, pool: PoolDep, _auth: AuthDep) -> Environment:
    return await service.get_environment(pool, env_id)


@router.put("/{env_id}")
async def update(
    env_id: str, body: EnvironmentUpdate, pool: PoolDep, _auth: AuthDep
) -> Environment:
    return await service.update_environment(pool, env_id, name=body.name, config=body.config)


@router.delete("/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive(env_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_environment(pool, env_id)
