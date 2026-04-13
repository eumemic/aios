"""Vault and vault credential endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, CryptoBoxDep, PoolDep
from aios.models.common import ListResponse
from aios.models.vaults import (
    Vault,
    VaultCreate,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
    VaultUpdate,
)
from aios.services import vaults as service

router = APIRouter(prefix="/v1/vaults", tags=["vaults"])


# ── Vault endpoints ─────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: VaultCreate, pool: PoolDep, _auth: AuthDep) -> Vault:
    return await service.create_vault(pool, display_name=body.display_name, metadata=body.metadata)


@router.get("")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[Vault]:
    items = await service.list_vaults(pool, limit=limit, after=after)
    return ListResponse[Vault](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{vault_id}")
async def get(vault_id: str, pool: PoolDep, _auth: AuthDep) -> Vault:
    return await service.get_vault(pool, vault_id)


@router.put("/{vault_id}")
async def update(vault_id: str, body: VaultUpdate, pool: PoolDep, _auth: AuthDep) -> Vault:
    return await service.update_vault(
        pool, vault_id, display_name=body.display_name, metadata=body.metadata
    )


@router.post("/{vault_id}/archive")
async def archive(vault_id: str, pool: PoolDep, _auth: AuthDep) -> Vault:
    return await service.archive_vault(pool, vault_id)


@router.delete("/{vault_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(vault_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    await service.delete_vault(pool, vault_id)


# ── Vault credential endpoints ──────────────────────────────────────────────


@router.post("/{vault_id}/credentials", status_code=status.HTTP_201_CREATED)
async def create_credential(
    vault_id: str,
    body: VaultCredentialCreate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> VaultCredential:
    return await service.create_vault_credential(pool, crypto_box, vault_id=vault_id, body=body)


@router.get("/{vault_id}/credentials")
async def list_credentials(
    vault_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: int = 50,
    after: str | None = None,
) -> ListResponse[VaultCredential]:
    items = await service.list_vault_credentials(pool, vault_id, limit=limit, after=after)
    return ListResponse[VaultCredential](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{vault_id}/credentials/{credential_id}")
async def get_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> VaultCredential:
    return await service.get_vault_credential(pool, vault_id, credential_id)


@router.put("/{vault_id}/credentials/{credential_id}")
async def update_credential(
    vault_id: str,
    credential_id: str,
    body: VaultCredentialUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> VaultCredential:
    return await service.update_vault_credential(
        pool, crypto_box, vault_id=vault_id, credential_id=credential_id, body=body
    )


@router.post("/{vault_id}/credentials/{credential_id}/archive")
async def archive_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> VaultCredential:
    return await service.archive_vault_credential(pool, vault_id, credential_id)


@router.delete("/{vault_id}/credentials/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> None:
    await service.delete_vault_credential(pool, vault_id, credential_id)
