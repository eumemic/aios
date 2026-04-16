"""Channel binding endpoints — explicit address → session mappings.

No PUT: bindings are immutable.  To re-route an address, archive the
existing binding and create a new one.  ``DELETE`` soft-archives.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.channel_bindings import ChannelBinding, ChannelBindingCreate
from aios.models.common import ListResponse
from aios.services import channels as service

router = APIRouter(prefix="/v1/channel-bindings", tags=["channel-bindings"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: ChannelBindingCreate, pool: PoolDep, _auth: AuthDep) -> ChannelBinding:
    return await service.create_binding(pool, address=body.address, session_id=body.session_id)


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    session_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[ChannelBinding]:
    items = await service.list_bindings(pool, session_id=session_id, limit=limit, after=after)
    return ListResponse[ChannelBinding](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{binding_id}")
async def get(binding_id: str, pool: PoolDep, _auth: AuthDep) -> ChannelBinding:
    return await service.get_binding(pool, binding_id)


@router.delete("/{binding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(binding_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_binding(pool, binding_id)
