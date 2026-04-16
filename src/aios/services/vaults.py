"""Business logic for vaults and vault credentials.

Vaults are named collections of encrypted credentials for MCP server
authentication. The CryptoBox handles encryption/decryption; the service
layer validates auth-type-specific fields and enforces limits.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import httpx

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.errors import NotFoundError, OAuthRefreshError, ValidationError
from aios.logging import get_logger
from aios.models.vaults import (
    TokenEndpointAuth,
    TokenEndpointAuthNone,
    Vault,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)

MAX_CREDENTIALS_PER_VAULT = 20

# Refresh an OAuth token if it expires within this window — gives the
# in-flight MCP request enough headroom to complete on the new token.
REFRESH_SKEW_SECONDS = 30

# Timeout for the OAuth token-endpoint POST. Generous because OAuth providers
# vary widely; tighter values cause spurious refresh failures.
_REFRESH_HTTP_TIMEOUT_SECONDS = 30

log = get_logger("aios.services.vaults")

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


def _is_expiring(payload: dict[str, Any], skew: int = REFRESH_SKEW_SECONDS) -> bool:
    """Return True if the token's ``expires_at`` is within ``skew`` seconds of now.

    Missing ``expires_at`` is treated as "never expires" (False) so that
    legacy / non-expiring tokens behave like ``static_bearer``.
    """
    raw = payload.get("expires_at")
    if not raw:
        return False
    try:
        expires_at = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC) + timedelta(seconds=skew)


async def refresh_credential(
    crypto_box: CryptoBox,
    conn: asyncpg.Connection[Any],
    *,
    vault_id: str,
    mcp_server_url: str,
) -> None:
    """Refresh the OAuth access token for ``(vault_id, mcp_server_url)``.

    Concurrency-safe: opens a transaction, locks the credential row with
    ``SELECT … FOR UPDATE``, then re-checks ``expires_at`` against the skew
    window. If a parallel coroutine already refreshed during the wait on
    the lock, this call returns without POSTing.

    Raises :class:`OAuthRefreshError` on any HTTP failure, malformed
    response, or missing fields. The transaction rolls back on raise so the
    stale token stays in place for the next attempt.
    """
    async with conn.transaction():
        locked = await queries.lock_oauth_credential_for_refresh(conn, vault_id, mcp_server_url)
        if locked is None:
            raise OAuthRefreshError(
                f"no active credential for {mcp_server_url!r} in vault {vault_id}",
                detail={"vault_id": vault_id, "mcp_server_url": mcp_server_url},
            )
        credential_id, blob = locked
        try:
            payload: dict[str, Any] = json.loads(crypto_box.decrypt(blob))
        except Exception as exc:
            raise OAuthRefreshError(
                "failed to decrypt stored credential",
                detail={"credential_id": credential_id, "reason": str(exc)},
            ) from exc

        # Double-check after lock — another worker may have refreshed already.
        if not _is_expiring(payload):
            return

        token_endpoint = payload.get("token_endpoint")
        refresh_token = payload.get("refresh_token")
        client_id = payload.get("client_id")
        if not token_endpoint or not refresh_token or not client_id:
            raise OAuthRefreshError(
                "credential is missing required refresh fields",
                detail={
                    "credential_id": credential_id,
                    "missing": [
                        name
                        for name, val in (
                            ("token_endpoint", token_endpoint),
                            ("refresh_token", refresh_token),
                            ("client_id", client_id),
                        )
                        if not val
                    ],
                },
            )

        # Build the POST per the configured token_endpoint_auth method.
        endpoint_auth = payload.get("token_endpoint_auth") or {"method": "none"}
        method = endpoint_auth.get("method", "none")
        body: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
        post_kwargs: dict[str, Any] = {"data": body}
        if method == "client_secret_basic":
            post_kwargs["auth"] = httpx.BasicAuth(client_id, endpoint_auth.get("client_secret", ""))
        elif method == "client_secret_post":
            body["client_secret"] = endpoint_auth.get("client_secret", "")
        # method == "none" → nothing extra; client_id alone identifies the public client.

        scope = payload.get("scope")
        if scope:
            body["scope"] = scope

        try:
            async with httpx.AsyncClient(timeout=_REFRESH_HTTP_TIMEOUT_SECONDS) as client:
                resp = await client.post(token_endpoint, **post_kwargs)
                resp.raise_for_status()
                token_data = resp.json()
        except httpx.HTTPError as exc:
            log.warning(
                "vault.oauth_refresh_http_error",
                credential_id=credential_id,
                token_endpoint=token_endpoint,
                exc_info=True,
            )
            raise OAuthRefreshError(
                f"OAuth token endpoint request failed: {exc}",
                detail={
                    "credential_id": credential_id,
                    "token_endpoint": token_endpoint,
                },
            ) from exc

        new_access_token = token_data.get("access_token")
        if not new_access_token:
            raise OAuthRefreshError(
                "OAuth response missing 'access_token'",
                detail={
                    "credential_id": credential_id,
                    "token_endpoint": token_endpoint,
                },
            )

        payload["access_token"] = new_access_token
        # Rotate refresh_token if the provider returned a new one; otherwise keep.
        rotated = token_data.get("refresh_token")
        if rotated:
            payload["refresh_token"] = rotated
        # Update expires_at if the provider declared an expires_in.
        expires_in = token_data.get("expires_in")
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            new_expiry = datetime.now(UTC) + timedelta(seconds=int(expires_in))
            payload["expires_at"] = new_expiry.isoformat()

        new_blob = crypto_box.encrypt(json.dumps(payload))
        await conn.execute(
            "UPDATE vault_credentials "
            "SET ciphertext = $1, nonce = $2, updated_at = now() "
            "WHERE id = $3",
            new_blob.ciphertext,
            new_blob.nonce,
            credential_id,
        )


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
