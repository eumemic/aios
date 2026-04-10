"""Credential CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep, VaultDep
from aios.models.common import ListResponse
from aios.models.credentials import Credential, CredentialCreate
from aios.services import credentials as service

router = APIRouter(prefix="/v1/credentials", tags=["credentials"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    body: CredentialCreate, pool: PoolDep, vault: VaultDep, _auth: AuthDep
) -> Credential:
    return await service.create_credential(
        pool,
        vault,
        name=body.name,
        provider=body.provider,
        plaintext_value=body.value.get_secret_value(),
    )


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Credential]:
    items = await service.list_credentials(pool, limit=limit, after=after)
    return ListResponse[Credential](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{cred_id}")
async def get(cred_id: str, pool: PoolDep, _auth: AuthDep) -> Credential:
    return await service.get_credential(pool, cred_id)


@router.delete("/{cred_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive(cred_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.archive_credential(pool, cred_id)
