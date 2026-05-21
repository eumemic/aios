"""Session-template endpoints — frozen recipes for per_chat session spawn.

A session template carries the agent + environment + bound vaults +
attached memory stores that a per_chat connection should use when
spawning a new session for an unseen chat partner.  ``DELETE``
soft-archives; existing per_chat connections keep working with
already-spawned sessions, but new chat sessions on those connections
will fail at the inbound handler until the connection is reconfigured.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.session_templates import (
    SessionTemplate,
    SessionTemplateCreate,
    SessionTemplateUpdate,
)
from aios.services import session_templates as service

router = APIRouter(prefix="/v1/session-templates", tags=["session-templates"])


@router.post("", operation_id="create_session_template", status_code=status.HTTP_201_CREATED)
async def create(
    body: SessionTemplateCreate, pool: PoolDep, account_id: AccountIdDep
) -> SessionTemplate:
    """Create a session template — a frozen recipe for per_chat session spawn.

    Captures the agent + environment + vaults + memory stores that a
    per_chat connection will use when spawning a session for a new chat
    partner. Pin ``agent_version`` to a specific version for deterministic
    spawning, or leave unset to track the agent's latest.
    """
    return await service.create_session_template(
        pool,
        name=body.name,
        agent_id=body.agent_id,
        environment_id=body.environment_id,
        agent_version=body.agent_version,
        vault_ids=body.vault_ids,
        memory_store_ids=body.memory_store_ids,
        metadata=body.metadata,
        account_id=account_id,
    )


@router.get("", operation_id="list_session_templates")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: str | None = None,
) -> ListResponse[SessionTemplate]:
    """List session templates, newest first, excluding archived.

    Cursor pagination via ``after``.
    """
    items = await service.list_session_templates(
        pool, limit=limit + 1, after=after, account_id=account_id
    )
    return ListResponse[SessionTemplate].paginate(items, limit, cursor=lambda x: x.id)


@router.get("/{template_id}", operation_id="get_session_template")
async def get(template_id: str, pool: PoolDep, account_id: AccountIdDep) -> SessionTemplate:
    """Fetch one session template by id."""
    return await service.get_session_template(pool, template_id, account_id=account_id)


@router.put("/{template_id}", operation_id="update_session_template")
async def update(
    template_id: str, body: SessionTemplateUpdate, pool: PoolDep, account_id: AccountIdDep
) -> SessionTemplate:
    """Update a session template's recipe fields. Omitted fields are preserved.

    The ``agent_version`` field uses sentinel-based partial-update semantics:
    omit it to preserve the current pin (or current "track latest" state),
    pass null to switch to "track latest," pass a number to pin to that
    specific version. Already-spawned sessions are unaffected.
    """
    return await service.update_session_template(
        pool,
        template_id,
        name=body.name,
        agent_id=body.agent_id,
        agent_version=(body.agent_version if "agent_version" in body.model_fields_set else ...),
        environment_id=body.environment_id,
        vault_ids=body.vault_ids,
        memory_store_ids=body.memory_store_ids,
        metadata=body.metadata,
        account_id=account_id,
    )


@router.delete(
    "/{template_id}",
    operation_id="archive_session_template",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete(template_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive a session template (soft-delete via DELETE verb).

    Already-spawned sessions are unaffected and continue normally. Per-chat
    connections that reference this template by id keep their existing
    sessions but will fail to spawn new chat sessions at the inbound
    handler until the connection is reconfigured to point at a different
    template. There is no API surface to un-archive currently.
    """
    await service.archive_session_template(pool, template_id, account_id=account_id)
