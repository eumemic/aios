"""Business logic for vaults and vault credentials.

Vaults are named collections of encrypted credentials for MCP server
authentication. The CryptoBox handles encryption/decryption; the service
layer validates auth-type-specific fields and enforces limits.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.errors import NotFoundError, ValidationError
from aios.models.vaults import (
    TokenEndpointAuth,
    TokenEndpointAuthNone,
    Vault,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)

MAX_CREDENTIALS_PER_VAULT = 20

# Fields that go into the encrypted payload for each auth type. The
# ``token_endpoint_auth`` field is a discriminated Pydantic union and gets
# special-cased in the (de)serializers below; ``client_secret`` lives inside
# its variants.
_OAUTH_FIELDS = (
    "access_token",
    "expires_at",
    "client_id",
    "refresh_token",
    "token_endpoint",
    "token_endpoint_auth",
    "scope",
    "resource",
)
_BEARER_FIELDS = ("token",)


def _serialize_token_endpoint_auth(v: TokenEndpointAuth) -> dict[str, str]:
    """Flatten the typed union into a JSON-serializable dict with the secret unwrapped."""
    if isinstance(v, TokenEndpointAuthNone):
        return {"method": "none"}
    return {"method": v.method, "client_secret": v.client_secret.get_secret_value()}


# ── vault CRUD ──────────────────────────────────────────────────────────────


async def create_vault(
    pool: asyncpg.Pool[Any],
    *,
    display_name: str,
    metadata: dict[str, Any],
) -> Vault:
    async with pool.acquire() as conn:
        return await queries.insert_vault(conn, display_name=display_name, metadata=metadata)


async def get_vault(pool: asyncpg.Pool[Any], vault_id: str) -> Vault:
    async with pool.acquire() as conn:
        return await queries.get_vault(conn, vault_id)


async def list_vaults(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Vault]:
    async with pool.acquire() as conn:
        return await queries.list_vaults(conn, limit=limit, after=after)


async def update_vault(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    *,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Vault:
    async with pool.acquire() as conn:
        return await queries.update_vault(
            conn, vault_id, display_name=display_name, metadata=metadata
        )


async def archive_vault(pool: asyncpg.Pool[Any], vault_id: str) -> Vault:
    async with pool.acquire() as conn:
        return await queries.archive_vault(conn, vault_id)


async def delete_vault(pool: asyncpg.Pool[Any], vault_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.delete_vault(conn, vault_id)


# ── vault credential CRUD ──────────────────────────────────────────────────


def _extract_auth_payload(body: VaultCredentialCreate) -> dict[str, Any]:
    """Build the auth payload dict from the request body.

    Validates required fields per auth_type. Returns the dict to be
    JSON-serialized and encrypted.
    """
    if body.auth_type == "static_bearer":
        if body.token is None:
            raise ValidationError(
                "static_bearer credentials require 'token'",
                detail={"auth_type": body.auth_type},
            )
        payload: dict[str, Any] = {}
        for field in _BEARER_FIELDS:
            val = getattr(body, field)
            if val is not None:
                payload[field] = val.get_secret_value() if hasattr(val, "get_secret_value") else val
        return payload
    else:  # mcp_oauth
        if body.access_token is None:
            raise ValidationError(
                "mcp_oauth credentials require 'access_token'",
                detail={"auth_type": body.auth_type},
            )
        payload = {}
        for field in _OAUTH_FIELDS:
            val = getattr(body, field)
            if val is None:
                continue
            if field == "token_endpoint_auth":
                payload[field] = _serialize_token_endpoint_auth(val)
            elif hasattr(val, "get_secret_value"):
                payload[field] = val.get_secret_value()
            elif hasattr(val, "isoformat"):
                payload[field] = val.isoformat()
            else:
                payload[field] = val
        return payload


def _merge_auth_payload(
    existing: dict[str, Any],
    body: VaultCredentialUpdate,
    auth_type: str,
) -> dict[str, Any]:
    """Merge update fields into the existing decrypted payload.

    Only fields present in model_fields_set replace existing values.
    """
    fields = _OAUTH_FIELDS if auth_type == "mcp_oauth" else _BEARER_FIELDS
    merged = dict(existing)
    for field in fields:
        if field not in body.model_fields_set:
            continue
        val = getattr(body, field)
        if val is None:
            merged.pop(field, None)
        elif field == "token_endpoint_auth":
            merged[field] = _serialize_token_endpoint_auth(val)
        elif hasattr(val, "get_secret_value"):
            merged[field] = val.get_secret_value()
        elif hasattr(val, "isoformat"):
            merged[field] = val.isoformat()
        else:
            merged[field] = val
    return merged


async def create_vault_credential(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    vault_id: str,
    body: VaultCredentialCreate,
) -> VaultCredential:
    payload = _extract_auth_payload(body)
    blob = crypto_box.encrypt(json.dumps(payload))
    async with pool.acquire() as conn, conn.transaction():
        # Lock the parent vault row to serialize concurrent credential inserts
        # within this vault. Without it, two parallel inserts can both observe
        # ``count == MAX-1`` and overflow the cap.
        locked = await conn.fetchrow(
            "SELECT 1 FROM vaults WHERE id = $1 FOR UPDATE",
            vault_id,
        )
        if locked is None:
            raise NotFoundError(
                f"vault {vault_id} not found",
                detail={"vault_id": vault_id},
            )
        count = await queries.count_active_vault_credentials(conn, vault_id)
        if count >= MAX_CREDENTIALS_PER_VAULT:
            raise ValidationError(
                f"vault has reached the maximum of {MAX_CREDENTIALS_PER_VAULT} credentials",
                detail={"vault_id": vault_id, "limit": MAX_CREDENTIALS_PER_VAULT},
            )
        return await queries.insert_vault_credential(
            conn,
            vault_id=vault_id,
            display_name=body.display_name,
            mcp_server_url=body.mcp_server_url,
            auth_type=body.auth_type,
            blob=blob,
            metadata=body.metadata,
        )


async def get_vault_credential(
    pool: asyncpg.Pool[Any], vault_id: str, credential_id: str
) -> VaultCredential:
    async with pool.acquire() as conn:
        return await queries.get_vault_credential(conn, vault_id, credential_id)


async def list_vault_credentials(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    *,
    limit: int = 50,
    after: str | None = None,
) -> list[VaultCredential]:
    async with pool.acquire() as conn:
        return await queries.list_vault_credentials(conn, vault_id, limit=limit, after=after)


async def update_vault_credential(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    vault_id: str,
    credential_id: str,
    body: VaultCredentialUpdate,
) -> VaultCredential:
    async with pool.acquire() as conn:
        # Decrypt existing payload, merge provided fields, re-encrypt.
        cred, existing_blob = await queries.get_vault_credential_with_blob(
            conn, vault_id, credential_id
        )
        existing_payload = json.loads(crypto_box.decrypt(existing_blob))
        merged = _merge_auth_payload(existing_payload, body, cred.auth_type)
        new_blob = crypto_box.encrypt(json.dumps(merged))

        return await queries.update_vault_credential(
            conn,
            vault_id,
            credential_id,
            blob=new_blob,
            display_name=(body.display_name if "display_name" in body.model_fields_set else ...),
            metadata=body.metadata if "metadata" in body.model_fields_set else ...,
        )


async def archive_vault_credential(
    pool: asyncpg.Pool[Any], vault_id: str, credential_id: str
) -> VaultCredential:
    async with pool.acquire() as conn:
        return await queries.archive_vault_credential(conn, vault_id, credential_id)


async def delete_vault_credential(
    pool: asyncpg.Pool[Any], vault_id: str, credential_id: str
) -> None:
    async with pool.acquire() as conn:
        await queries.delete_vault_credential(conn, vault_id, credential_id)
