"""Session-template endpoints — frozen recipes for per_chat session spawn.

A session template carries the agent + environment + bound vaults +
attached memory stores that a per_chat connection should use when
spawning a new session for an unseen chat partner.  ``DELETE``
soft-archives; existing per_chat connections keep working with
already-spawned sessions, but new chat sessions on those connections
will fail at the inbound handler until the connection is reconfigured.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.db.queries import _UNSET
from aios.models.common import ListResponse
from aios.models.session_templates import (
    SessionTemplate,
    SessionTemplateCreate,
    SessionTemplateUpdate,
)
from aios.services import session_templates as service

router = APIRouter(prefix="/v1/session-templates", tags=["session-templates"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: SessionTemplateCreate, pool: PoolDep, _auth: AuthDep) -> SessionTemplate:
    return await service.create_session_template(
        pool,
        name=body.name,
        agent_id=body.agent_id,
        environment_id=body.environment_id,
        agent_version=body.agent_version,
        vault_ids=body.vault_ids,
        memory_store_ids=body.memory_store_ids,
        metadata=body.metadata,
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[SessionTemplate]:
    items = await service.list_session_templates(pool, limit=limit, after=after)
    return ListResponse[SessionTemplate](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{template_id}")
async def get(template_id: str, pool: PoolDep, _auth: AuthDep) -> SessionTemplate:
    return await service.get_session_template(pool, template_id)


@router.put("/{template_id}")
async def update(
    template_id: str, body: SessionTemplateUpdate, pool: PoolDep, _auth: AuthDep
) -> SessionTemplate:
    return await service.update_session_template(
        pool,
        template_id,
        name=body.name,
        agent_id=body.agent_id,
        agent_version=(body.agent_version if "agent_version" in body.model_fields_set else _UNSET),
        environment_id=body.environment_id,
        vault_ids=body.vault_ids,
        memory_store_ids=body.memory_store_ids,
        metadata=body.metadata,
    )


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(template_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_session_template(pool, template_id)
