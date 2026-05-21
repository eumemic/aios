"""Environment CRUD endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.environments import Environment, EnvironmentCreate, EnvironmentUpdate
from aios.services import environments as service

router = APIRouter(prefix="/v1/environments", tags=["environments"])


@router.post("", operation_id="create_environment", status_code=status.HTTP_201_CREATED)
async def create(body: EnvironmentCreate, pool: PoolDep, account_id: AccountIdDep) -> Environment:
    """Create a new environment — a reusable sandbox configuration template.

    The ``config`` field describes the container the sessions running this
    environment will spawn (pre-installed packages, network access rules,
    base image). Sessions reference environments by id; multiple sessions
    can share one environment.
    """
    return await service.create_environment(
        pool, name=body.name, config=body.config, account_id=account_id
    )


@router.get("", operation_id="list_environments")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: str | None = None,
) -> ListResponse[Environment]:
    """List environments, newest first, excluding archived.

    Cursor pagination via ``after``.
    """
    items = await service.list_environments(
        pool, limit=limit + 1, after=after, account_id=account_id
    )
    return ListResponse[Environment].paginate(items, limit, cursor=lambda x: x.id)


@router.get("/{env_id}", operation_id="get_environment")
async def get(env_id: str, pool: PoolDep, account_id: AccountIdDep) -> Environment:
    """Fetch one environment by id."""
    return await service.get_environment(pool, env_id, account_id=account_id)


@router.put("/{env_id}", operation_id="update_environment")
async def update(
    env_id: str, body: EnvironmentUpdate, pool: PoolDep, account_id: AccountIdDep
) -> Environment:
    """Update an environment's ``name`` and/or ``config``.

    Sessions resolve the environment config fresh each time their sandbox
    is provisioned (lazily, at the next session step that needs the
    container), so updates take effect for existing sessions on their next
    provision rather than at update time.
    """
    return await service.update_environment(
        pool, env_id, name=body.name, config=body.config, account_id=account_id
    )


@router.delete(
    "/{env_id}", operation_id="archive_environment", status_code=status.HTTP_204_NO_CONTENT
)
async def archive(env_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive an environment: hides from default lists, leaves the row in place.

    Sessions referencing the archived environment continue to function — the
    sandbox provisioner reads the config by JOIN and does not filter by
    archive state. There is no API surface to un-archive currently.
    """
    await service.archive_environment(pool, env_id, account_id=account_id)
