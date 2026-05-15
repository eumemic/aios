"""Vault and vault credential endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

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


@router.post("", operation_id="create_vault", status_code=status.HTTP_201_CREATED)
async def create(body: VaultCreate, pool: PoolDep, _auth: AuthDep) -> Vault:
    """Create a new vault — a named collection for MCP server credentials.

    Credentials are added separately via ``create_vault_credential``.
    """
    account_id, _, _ = _auth
    return await service.create_vault(
        pool, display_name=body.display_name, metadata=body.metadata, account_id=account_id
    )


@router.get("", operation_id="list_vaults")
async def list_(
    pool: PoolDep,
    _auth: AuthDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: str | None = None,
) -> ListResponse[Vault]:
    """List vaults, newest first, excluding archived.

    Cursor pagination: pass ``after`` from a previous response's
    ``next_after`` to get the next page.
    """
    account_id, _, _ = _auth
    items = await service.list_vaults(pool, limit=limit, after=after, account_id=account_id)
    return ListResponse[Vault](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{vault_id}", operation_id="get_vault")
async def get(vault_id: str, pool: PoolDep, _auth: AuthDep) -> Vault:
    """Fetch one vault by id."""
    account_id, _, _ = _auth
    return await service.get_vault(pool, vault_id, account_id=account_id)


@router.put("/{vault_id}", operation_id="update_vault")
async def update(vault_id: str, body: VaultUpdate, pool: PoolDep, _auth: AuthDep) -> Vault:
    """Update a vault's ``display_name`` and/or ``metadata``.

    Omitted fields are preserved.
    """
    account_id, _, _ = _auth
    return await service.update_vault(
        pool,
        vault_id,
        display_name=body.display_name,
        metadata=body.metadata,
        account_id=account_id,
    )


@router.post(
    "/{vault_id}/archive",
    operation_id="archive_vault",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive(vault_id: str, pool: PoolDep, _auth: AuthDep) -> Vault:
    """Archive a vault and **purge the encrypted secret material** of its credentials.

    Sets ``archived_at`` and hides the vault from default lists. In the same
    transaction, every active credential is archived and its encrypted blob
    is zeroed — the credential rows persist for audit but the stored secrets
    are unrecoverable. Defense in depth: a future DB dump cannot leak
    secrets that were already retired.

    Use ``delete_vault`` instead if you want to remove the rows entirely.
    """
    account_id, _, _ = _auth
    return await service.archive_vault(pool, vault_id, account_id=account_id)


@router.delete("/{vault_id}", operation_id="delete_vault", status_code=status.HTTP_204_NO_CONTENT)
async def delete(vault_id: str, pool: PoolDep, _auth: AuthDep) -> None:
    """Hard-delete a vault and all its credentials (``ON DELETE CASCADE``).

    Returns 204. Unlike ``archive_vault``, this removes the rows entirely
    and leaves no audit trail. Prefer archive unless you specifically need
    the rows gone.
    """
    account_id, _, _ = _auth
    await service.delete_vault(pool, vault_id, account_id=account_id)


# ── Vault credential endpoints ──────────────────────────────────────────────


@router.post(
    "/{vault_id}/credentials",
    operation_id="create_vault_credential",
    status_code=status.HTTP_201_CREATED,
)
async def create_credential(
    vault_id: str,
    body: VaultCredentialCreate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> VaultCredential:
    """Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``mcp_oauth`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``static_bearer`` requires ``token``. Caps at 20 active credentials per
    vault. The ``mcp_server_url`` is immutable after creation — to retarget
    a credential, archive the existing one and create a new credential at
    the new URL.
    """
    account_id, _, _ = _auth
    return await service.create_vault_credential(
        pool, crypto_box, vault_id=vault_id, body=body, account_id=account_id
    )


@router.get("/{vault_id}/credentials", operation_id="list_vault_credentials")
async def list_credentials(
    vault_id: str,
    pool: PoolDep,
    _auth: AuthDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    after: str | None = None,
) -> ListResponse[VaultCredential]:
    """List credentials in a vault, newest first, excluding archived.

    Cursor pagination via ``after``. Secret material is never returned —
    only metadata (display name, mcp_server_url, auth_type, timestamps).
    """
    account_id, _, _ = _auth
    items = await service.list_vault_credentials(
        pool, vault_id, limit=limit, after=after, account_id=account_id
    )
    return ListResponse[VaultCredential](
        data=items,
        has_more=len(items) == limit,
        next_after=items[-1].id if items else None,
    )


@router.get("/{vault_id}/credentials/{credential_id}", operation_id="get_vault_credential")
async def get_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> VaultCredential:
    """Fetch one credential's metadata. Secrets are never returned.

    Internal MCP clients resolve the secret directly through the service
    layer (with OAuth refresh as needed); the HTTP API never exposes it.
    """
    account_id, _, _ = _auth
    return await service.get_vault_credential(pool, vault_id, credential_id, account_id=account_id)


@router.put("/{vault_id}/credentials/{credential_id}", operation_id="update_vault_credential")
async def update_credential(
    vault_id: str,
    credential_id: str,
    body: VaultCredentialUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    _auth: AuthDep,
) -> VaultCredential:
    """Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``mcp_server_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.
    """
    account_id, _, _ = _auth
    return await service.update_vault_credential(
        pool,
        crypto_box,
        vault_id=vault_id,
        credential_id=credential_id,
        body=body,
        account_id=account_id,
    )


@router.post(
    "/{vault_id}/credentials/{credential_id}/archive",
    operation_id="archive_vault_credential",
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> VaultCredential:
    """Archive a credential and **zero its encrypted secret payload**.

    Sets ``archived_at`` and hides the credential from default lists. The
    encrypted blob is scrubbed at archive time so a future DB dump cannot
    leak the secret. Use ``delete_vault_credential`` for full removal.
    """
    account_id, _, _ = _auth
    return await service.archive_vault_credential(
        pool, vault_id, credential_id, account_id=account_id
    )


@router.delete(
    "/{vault_id}/credentials/{credential_id}",
    operation_id="delete_vault_credential",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_credential(
    vault_id: str, credential_id: str, pool: PoolDep, _auth: AuthDep
) -> None:
    """Hard-delete a credential row. Returns 204.

    Unlike ``archive_vault_credential``, removes the row entirely and
    leaves no audit trail. Prefer archive unless you specifically need the
    row gone.
    """
    account_id, _, _ = _auth
    await service.delete_vault_credential(pool, vault_id, credential_id, account_id=account_id)
