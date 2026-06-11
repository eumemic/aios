"""Vault and vault credential endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aios.api.deps import AccountIdDep, CryptoBoxDep, PoolDep
from aios.models.common import ListResponse
from aios.models.pagination import page_cursor
from aios.models.vaults import (
    OAuthCompleteRequest,
    OAuthStartRequest,
    OAuthStartResponse,
    Vault,
    VaultCreate,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
    VaultUpdate,
)
from aios.services import vault_oauth as oauth_service
from aios.services import vaults as service

router = APIRouter(prefix="/v1/vaults", tags=["vaults"])


# ── Vault endpoints ─────────────────────────────────────────────────────────


@router.post("", operation_id="create_vault", status_code=status.HTTP_201_CREATED)
async def create(body: VaultCreate, pool: PoolDep, account_id: AccountIdDep) -> Vault:
    """Create a new vault — a named collection for MCP server credentials.

    Credentials are added separately via ``create_vault_credential``.
    """
    return await service.create_vault(
        pool, display_name=body.display_name, metadata=body.metadata, account_id=account_id
    )


@router.get("", operation_id="list_vaults")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
) -> ListResponse[Vault]:
    """List vaults, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``.
    """
    st = page_cursor(cursor, {"limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = st.limit if st is not None else (limit if limit is not None else 50)
    items = await service.list_vaults(
        pool, limit=page_limit + 1, after=after, account_id=account_id
    )
    return ListResponse[Vault].paginate(items, page_limit, cursor=lambda x: x.id)


@router.get("/{vault_id}", operation_id="get_vault")
async def get(vault_id: str, pool: PoolDep, account_id: AccountIdDep) -> Vault:
    """Fetch one vault by id."""
    return await service.get_vault(pool, vault_id, account_id=account_id)


@router.put("/{vault_id}", operation_id="update_vault")
async def update(
    vault_id: str, body: VaultUpdate, pool: PoolDep, account_id: AccountIdDep
) -> Vault:
    """Update a vault's ``display_name`` and/or ``metadata``.

    Omitted fields are preserved.
    """
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
async def archive(vault_id: str, pool: PoolDep, account_id: AccountIdDep) -> Vault:
    """Archive a vault and **purge the encrypted secret material** of its credentials.

    Sets ``archived_at`` and hides the vault from default lists. In the same
    transaction, every active credential is archived and its encrypted blob
    is zeroed — the credential rows persist for audit but the stored secrets
    are unrecoverable. Defense in depth: a future DB dump cannot leak
    secrets that were already retired.

    Use ``delete_vault`` instead if you want to remove the rows entirely.
    """
    return await service.archive_vault(pool, vault_id, account_id=account_id)


@router.delete("/{vault_id}", operation_id="delete_vault", status_code=status.HTTP_204_NO_CONTENT)
async def delete(vault_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Hard-delete a vault and all its credentials (``ON DELETE CASCADE``).

    Returns 204. Unlike ``archive_vault``, this removes the rows entirely
    and leaves no audit trail. Prefer archive unless you specifically need
    the rows gone.
    """
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
    account_id: AccountIdDep,
) -> VaultCredential:
    """Add a credential to a vault. Secrets are encrypted at rest via the CryptoBox.

    Validates required fields per ``auth_type``: ``oauth2_refresh`` requires
    ``access_token`` (plus the refresh fields needed for rotation);
    ``bearer_header`` requires ``token``; ``basic`` requires ``username``
    and ``password``; ``custom_header`` requires ``header_name`` and
    ``header_value``. ``environment_variable`` is the sandbox-materialized
    kind: it requires ``secret_name`` (a POSIX env var name) + a non-empty
    ``allowed_hosts`` egress scope and ``secret_value``, and carries no
    ``target_url``. Caps at 20 active credentials per vault. ``target_url``,
    ``secret_name``, and ``auth_type`` are immutable after creation — archive
    and recreate to change them.
    """
    return await service.create_vault_credential(
        pool, crypto_box, vault_id=vault_id, body=body, account_id=account_id
    )


@router.post(
    "/{vault_id}/credentials/oauth/start",
    operation_id="start_vault_credential_oauth",
)
async def start_credential_oauth(
    vault_id: str,
    body: OAuthStartRequest,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> OAuthStartResponse:
    """Begin an interactive OAuth "Connect" flow for an MCP server.

    Discovers the target's OAuth metadata, registers a client (RFC 7591
    Dynamic Client Registration) or uses a caller-supplied ``client_id`` /
    ``client_secret`` for servers without DCR, generates PKCE + a CSRF
    ``state``, and returns the provider ``authorization_url`` to redirect the
    user to. The token fields are obtained from the provider — the caller does
    not supply them. Complete the flow with the returned ``state`` + the
    authorization ``code`` via ``complete_vault_credential_oauth``.
    """
    return await oauth_service.start_oauth_flow(
        pool, crypto_box, vault_id=vault_id, body=body, account_id=account_id
    )


@router.post(
    "/{vault_id}/credentials/oauth/complete",
    operation_id="complete_vault_credential_oauth",
    status_code=status.HTTP_201_CREATED,
)
async def complete_credential_oauth(
    vault_id: str,
    body: OAuthCompleteRequest,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> VaultCredential:
    """Finish an interactive OAuth flow: exchange the code and store the credential.

    Validates the ``state`` against the in-progress flow, exchanges the
    authorization ``code`` for tokens, and stores them as an ``oauth2_refresh``
    credential (creating a new one, or rotating an existing credential for the
    same ``target_url``). Secrets are encrypted at rest and never returned.
    """
    return await oauth_service.complete_oauth_flow(
        pool, crypto_box, vault_id=vault_id, body=body, account_id=account_id
    )


@router.get("/{vault_id}/credentials", operation_id="list_vault_credentials")
async def list_credentials(
    vault_id: str,
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
) -> ListResponse[VaultCredential]:
    """List credentials in a vault, newest first, excluding archived.

    First page: ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>``. Secret
    material is never returned — only metadata (display name, target_url,
    auth_type, timestamps).
    """
    st = page_cursor(cursor, {"limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = st.limit if st is not None else (limit if limit is not None else 50)
    items = await service.list_vault_credentials(
        pool, vault_id, limit=page_limit + 1, after=after, account_id=account_id
    )
    return ListResponse[VaultCredential].paginate(items, page_limit, cursor=lambda x: x.id)


@router.get("/{vault_id}/credentials/{credential_id}", operation_id="get_vault_credential")
async def get_credential(
    vault_id: str, credential_id: str, pool: PoolDep, account_id: AccountIdDep
) -> VaultCredential:
    """Fetch one credential's metadata. Secrets are never returned.

    Internal MCP clients resolve the secret directly through the service
    layer (with OAuth refresh as needed); the HTTP API never exposes it.
    """
    return await service.get_vault_credential(pool, vault_id, credential_id, account_id=account_id)


@router.put("/{vault_id}/credentials/{credential_id}", operation_id="update_vault_credential")
async def update_credential(
    vault_id: str,
    credential_id: str,
    body: VaultCredentialUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> VaultCredential:
    """Update a credential's metadata and/or rotate its auth secrets.

    Omitted secret fields are preserved (decrypt-merge-encrypt cycle on the
    encrypted payload). ``target_url`` and ``auth_type`` are immutable
    and not accepted in the body. To rotate an OAuth refresh token, send
    only the new ``refresh_token`` (and optional ``access_token`` /
    ``expires_at``); other auth fields stay intact.
    """
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
    vault_id: str, credential_id: str, pool: PoolDep, account_id: AccountIdDep
) -> VaultCredential:
    """Archive a credential and **zero its encrypted secret payload**.

    Sets ``archived_at`` and hides the credential from default lists. The
    encrypted blob is scrubbed at archive time so a future DB dump cannot
    leak the secret. Use ``delete_vault_credential`` for full removal.
    """
    return await service.archive_vault_credential(
        pool, vault_id, credential_id, account_id=account_id
    )


@router.delete(
    "/{vault_id}/credentials/{credential_id}",
    operation_id="delete_vault_credential",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_credential(
    vault_id: str, credential_id: str, pool: PoolDep, account_id: AccountIdDep
) -> None:
    """Hard-delete a credential row. Returns 204.

    Unlike ``archive_vault_credential``, removes the row entirely and
    leaves no audit trail. Prefer archive unless you specifically need the
    row gone.
    """
    await service.delete_vault_credential(pool, vault_id, credential_id, account_id=account_id)
