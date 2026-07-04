"""Business logic for vaults and vault credentials.

Vaults are named collections of encrypted credentials for authenticated
outbound services (MCP servers, HTTP APIs). The CryptoBox handles
encryption/decryption; the service layer validates auth-type-specific
fields and enforces limits.

Sandbox eviction (#713): session-BINDING changes (which vaults a session
can reach) flow through :func:`aios.db.queries.set_session_vaults` from
:func:`aios.services.sessions.update_session`, which fires the post-commit
sandbox-eviction hook as defense-in-depth.

MCP-pool eviction (#1030): the in-place credential mutations here overwrite
the credential row under a pool key that is stable across token changes
(``(url, vault_id, headers_key)`` — by design, for OAuth-refresh reuse).
But the resolved bearer is baked into the live ``ClientSession`` at open
time, so an operator rotation would otherwise keep serving the OLD token
until the 900s idle reaper — which active polling defeats. So a credential
mutation (:func:`update_vault_credential`, :func:`archive_vault_credential`,
:func:`delete_vault_credential`) — and a vault-level archive/delete that
retires its credentials — fires a NOTIFY on ``MCP_EVICT_VAULT_CHANNEL`` with
the ``vault_id``. The worker (which owns the pool, in a separate process)
LISTENs and calls :meth:`McpSessionPool.evict_by_vault`. The automated
:func:`refresh_credential` path is deliberately NOT evicted: it is the OAuth
case the idle reaper already covers (#459). ``create_vault_credential`` adds
a brand-new credential (a new ``target_url`` with no pooled session yet), so
it needs no eviction either.

Since #873, ``environment_variable``
credentials DO feed the sandbox spec builder
(:func:`resolve_session_env_var_credentials` below): vault BINDING
changes now bump ``spec_version`` (migration 0082) so a live sandbox
recycles, while credential-LEVEL drift in a still-bound vault (rotation,
create/archive) now recycles the sandbox too, via the ``updated_at`` drift
key (#877).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, assert_never, cast

import asyncpg
import httpx

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.listen import MCP_EVICT_VAULT_CHANNEL
from aios.errors import NotFoundError, OAuthRefreshError, ValidationError
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.vaults import (
    AuthType,
    TokenEndpointAuth,
    TokenEndpointAuthNone,
    Vault,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
    parse_allowed_host_entry,
)

MAX_CREDENTIALS_PER_VAULT = 20

# Refresh an OAuth token if it expires within this window — gives the
# in-flight request enough headroom to complete on the new token.
REFRESH_SKEW_SECONDS = 30

# Timeout for the OAuth token-endpoint POST. Generous because OAuth providers
# vary widely; tighter values cause spurious refresh failures.
_REFRESH_HTTP_TIMEOUT_SECONDS = 30

log = get_logger("aios.services.vaults")


async def _notify_evict_vault(pool: asyncpg.Pool[Any], vault_id: str) -> None:
    """Signal the worker's MCP session pool to evict every pooled session keyed
    on ``vault_id`` after an operator credential mutation (#1030).

    The mutation runs in the API process; the pool lives in the worker. This
    fires a ``pg_notify`` on ``MCP_EVICT_VAULT_CHANNEL`` (function form, never
    literal NOTIFY — our ids carry uppercase) which the worker's
    ``_run_mcp_evict_listener`` drains and dispatches to
    :meth:`McpSessionPool.evict_by_vault`. Fire-and-forget, like the session
    interrupt signal: a notification lost during a listener reconnect falls
    back to the idle reaper, and the next mutation re-fires.
    """
    await pool.execute("SELECT pg_notify($1, $2)", MCP_EVICT_VAULT_CHANNEL, vault_id)


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
_BASIC_FIELDS = ("username", "password")
_CUSTOM_HEADER_FIELDS = ("header_name", "header_value")
_ENV_VAR_FIELDS = ("secret_value",)

# Fields required by each auth_type. Used by both create-time validation
# (against the request body) and post-merge validation (against the merged
# decrypted payload).
_AUTH_REQUIRED: dict[AuthType, tuple[str, ...]] = {
    "bearer_header": _BEARER_FIELDS,
    "basic": _BASIC_FIELDS,
    "custom_header": _CUSTOM_HEADER_FIELDS,
    "oauth2_refresh": ("access_token",),
    "environment_variable": _ENV_VAR_FIELDS,
}


def _fields_for(auth_type: AuthType) -> tuple[str, ...]:
    if auth_type == "oauth2_refresh":
        return _OAUTH_FIELDS
    if auth_type == "bearer_header":
        return _BEARER_FIELDS
    if auth_type == "basic":
        return _BASIC_FIELDS
    if auth_type == "custom_header":
        return _CUSTOM_HEADER_FIELDS
    if auth_type == "environment_variable":
        return _ENV_VAR_FIELDS
    assert_never(auth_type)


def _serialize_token_endpoint_auth(v: TokenEndpointAuth) -> dict[str, str]:
    """Flatten the typed union into a JSON-serializable dict with the secret unwrapped."""
    if isinstance(v, TokenEndpointAuthNone):
        return {"method": "none"}
    return {"method": v.method, "client_secret": v.client_secret.get_secret_value()}


def build_token_endpoint_post(
    body: dict[str, str],
    *,
    client_id: str,
    endpoint_auth: dict[str, str] | None,
) -> dict[str, Any]:
    """Apply the configured token-endpoint client-auth method to an OAuth POST.

    Mutates ``body`` in place (``client_id`` is always added — it identifies the
    public client; ``client_secret`` is added for the ``client_secret_post``
    method) and returns the kwargs for ``httpx.AsyncClient.post`` (``data``,
    plus ``auth`` for ``client_secret_basic``). Shared by the refresh path
    (:func:`refresh_credential`) and the authorization-code exchange
    (:mod:`aios.services.vault_oauth`) so the two never drift.
    """
    auth = endpoint_auth or {"method": "none"}
    method = auth.get("method", "none")
    body["client_id"] = client_id
    post_kwargs: dict[str, Any] = {"data": body}
    if method == "client_secret_basic":
        post_kwargs["auth"] = httpx.BasicAuth(client_id, auth.get("client_secret", ""))
    elif method == "client_secret_post":
        body["client_secret"] = auth.get("client_secret", "")
    # method == "none" → nothing extra; client_id alone identifies the public client.
    return post_kwargs


def is_expiring(payload: dict[str, Any], skew: int = REFRESH_SKEW_SECONDS) -> bool:
    """Return True if the token's ``expires_at`` is within ``skew`` seconds of now.

    Missing ``expires_at`` is treated as "never expires" (False) so that
    legacy / non-expiring tokens behave like ``bearer_header``.
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
    account_id: str,
    vault_id: str,
    target_url: str,
) -> None:
    """Refresh the OAuth access token for ``(vault_id, target_url)``.

    Concurrency-safe: opens a transaction, locks the credential row with
    ``SELECT … FOR UPDATE``, then re-checks ``expires_at`` against the skew
    window. If a parallel coroutine already refreshed during the wait on
    the lock, this call returns without POSTing.

    Raises :class:`OAuthRefreshError` on any HTTP failure, malformed
    response, or missing fields. The transaction rolls back on raise so the
    stale token stays in place for the next attempt.
    """
    async with conn.transaction():
        locked = await queries.lock_oauth_credential_for_refresh(
            conn, vault_id, target_url, account_id=account_id
        )
        if locked is None:
            raise OAuthRefreshError(
                f"no active credential for {target_url!r} in vault {vault_id}",
                detail={"vault_id": vault_id, "target_url": target_url},
            )
        credential_id, blob = locked
        subkey = crypto_box.derive_account_subkey(account_id)
        try:
            payload = subkey.decrypt_dict(blob)
        except Exception as exc:
            raise OAuthRefreshError(
                "failed to decrypt stored credential",
                detail={"credential_id": credential_id, "reason": str(exc)},
            ) from exc

        # Double-check after lock — another worker may have refreshed already.
        if not is_expiring(payload):
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

        # Build the refresh POST. Client-auth handling (none/basic/post) is
        # shared with the authorization-code exchange via
        # ``build_token_endpoint_post`` so the two paths can't drift.
        body: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        scope = payload.get("scope")
        if scope:
            body["scope"] = scope
        # RFC 8707: some providers bind the refresh to the same resource the
        # tokens were minted for; include it when the credential stored one.
        resource = payload.get("resource")
        if resource:
            body["resource"] = resource
        post_kwargs = build_token_endpoint_post(
            body, client_id=client_id, endpoint_auth=payload.get("token_endpoint_auth")
        )

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
        # Update expires_at if the provider declared an expires_in. Accept
        # numeric strings as well as int/float — RFC 6749 says SHOULD be a
        # number, but real providers (Slack, others) sometimes return strings.
        # Without this, the new access_token would be stored without an
        # ``expires_at`` and ``is_expiring`` would treat it as never-expiring.
        try:
            seconds = int(token_data.get("expires_in", 0))
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            payload["expires_at"] = (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
        else:
            # No expires_in in the response: drop any prior expires_at so the
            # fresh token reads as never-expiring. The prior value is within-skew
            # by construction (a refresh only fires when is_expiring was True), so
            # inheriting it would keep is_expiring True and re-trigger a refresh on
            # every subsequent call — churning a rotated refresh_token and tripping
            # rate limits. Mirrors _store_oauth_credential's write-even-when-None.
            payload.pop("expires_at", None)

        new_blob = subkey.encrypt_dict(payload)
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
    account_id: str,
    display_name: str,
    metadata: dict[str, Any],
) -> Vault:
    async with pool.acquire() as conn:
        return await queries.insert_vault(
            conn, display_name=display_name, metadata=metadata, account_id=account_id
        )


async def get_vault(pool: asyncpg.Pool[Any], vault_id: str, *, account_id: str) -> Vault:
    async with pool.acquire() as conn:
        return await queries.get_vault(conn, vault_id, account_id=account_id)


async def list_vaults(
    pool: asyncpg.Pool[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[Vault]:
    async with pool.acquire() as conn:
        return await queries.list_vaults(conn, limit=limit, after=after, account_id=account_id)


async def update_vault(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    *,
    account_id: str,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Vault:
    async with pool.acquire() as conn:
        return await queries.update_vault(
            conn, vault_id, display_name=display_name, metadata=metadata, account_id=account_id
        )


async def archive_vault(
    pool: asyncpg.Pool[Any], vault_id: str, *, account_id: str, idempotent: bool = False
) -> Vault:
    async with pool.acquire() as conn:
        archived = await queries.archive_vault(
            conn, vault_id, account_id=account_id, idempotent=idempotent
        )
    # Archiving a vault retires (zeroes) all its credentials' secrets, so the
    # pooled MCP sessions keyed on this vault must be evicted too (#1030).
    await _notify_evict_vault(pool, vault_id)
    return archived


async def delete_vault(pool: asyncpg.Pool[Any], vault_id: str, *, account_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.delete_vault(conn, vault_id, account_id=account_id)
    # Deleting a vault cascades to its credentials, so evict pooled sessions (#1030).
    await _notify_evict_vault(pool, vault_id)


# ── vault credential CRUD ──────────────────────────────────────────────────


def _extract_auth_payload(body: VaultCredentialCreate) -> dict[str, Any]:
    """Build the auth payload dict from the request body.

    Validates required fields per auth_type. Returns the dict to be
    JSON-serialized and encrypted.
    """
    missing = [f for f in _AUTH_REQUIRED[body.auth_type] if getattr(body, f) is None]
    if missing:
        raise ValidationError(
            f"{body.auth_type} credentials require {missing}",
            detail={"auth_type": body.auth_type, "missing": missing},
        )

    payload: dict[str, Any] = {}
    for field_name in _fields_for(body.auth_type):
        val = getattr(body, field_name)
        if val is None:
            continue
        if field_name == "token_endpoint_auth":
            payload[field_name] = _serialize_token_endpoint_auth(val)
        elif hasattr(val, "get_secret_value"):
            payload[field_name] = val.get_secret_value()
        elif hasattr(val, "isoformat"):
            payload[field_name] = val.isoformat()
        else:
            payload[field_name] = val
    return payload


def _merge_auth_payload(
    existing: dict[str, Any],
    body: VaultCredentialUpdate,
    auth_type: AuthType,
) -> dict[str, Any]:
    """Merge update fields into the existing decrypted payload.

    Only fields present in model_fields_set replace existing values.
    Fields explicitly set to ``None`` are unset from the merged payload
    (the documented unset-via-null behavior). After the merge, the
    payload is re-validated against ``auth_type``'s required-fields list:
    a PUT that unsets a required-by-auth-type field (e.g. ``token`` for
    ``bearer_header``) raises ``ValidationError`` rather than landing a
    silently-broken credential whose downstream auth-header rendering
    would emit an empty bearer.
    """
    fields = _fields_for(auth_type)
    merged = dict(existing)
    for field_name in fields:
        if field_name not in body.model_fields_set:
            continue
        val = getattr(body, field_name)
        if val is None:
            merged.pop(field_name, None)
        elif field_name == "token_endpoint_auth":
            merged[field_name] = _serialize_token_endpoint_auth(val)
        elif hasattr(val, "get_secret_value"):
            merged[field_name] = val.get_secret_value()
        elif hasattr(val, "isoformat"):
            merged[field_name] = val.isoformat()
        else:
            merged[field_name] = val
    _validate_required_in_payload(merged, auth_type)
    return merged


def _validate_required_in_payload(payload: dict[str, Any], auth_type: AuthType) -> None:
    """Raise ``ValidationError`` if any field required by ``auth_type`` is
    missing or empty in ``payload``. Mirrors the create-side checks in
    :func:`_extract_auth_payload`; the merge path delegates here so a
    PUT can't land an incomplete credential under the unset-via-null
    contract.
    """
    missing = [f for f in _AUTH_REQUIRED[auth_type] if not payload.get(f)]
    if missing:
        raise ValidationError(
            f"{auth_type} credentials require {missing}",
            detail={"auth_type": auth_type, "missing": missing},
        )


async def create_vault_credential(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    vault_id: str,
    body: VaultCredentialCreate,
) -> VaultCredential:
    payload = _extract_auth_payload(body)
    blob = crypto_box.derive_account_subkey(account_id).encrypt_dict(payload)
    async with pool.acquire() as conn, conn.transaction():
        # Lock the parent vault row to serialize concurrent credential inserts
        # within this vault. Without it, two parallel inserts can both observe
        # ``count == MAX-1`` and overflow the cap.
        #
        # The ``account_id = $2`` filter is load-bearing for tenant isolation,
        # not just the lock's correctness: ``insert_vault_credential`` trusts
        # the caller's ``account_id`` (writes it verbatim) and the
        # ``vault_credentials.vault_id`` FK enforces only existence, not
        # account-matching. Without scoping the lock, tenant A can target
        # tenant B's vault_id, land the lock, pass the (correctly scoped)
        # quota check at count 0, and write a credential row at
        # ``(account_id=A, vault_id=B-vault)``. Mirrors the sessions.py
        # ``SELECT id FROM sessions … FOR UPDATE`` pattern, where the
        # account_id is similarly load-bearing rather than redundant.
        locked = await conn.fetchrow(
            "SELECT 1 FROM vaults WHERE id = $1 AND account_id = $2 FOR UPDATE",
            vault_id,
            account_id,
        )
        if locked is None:
            raise NotFoundError(
                f"vault {vault_id} not found",
                detail={"vault_id": vault_id},
            )
        count = await queries.count_active_vault_credentials(conn, vault_id, account_id=account_id)
        if count >= MAX_CREDENTIALS_PER_VAULT:
            raise ValidationError(
                f"vault has reached the maximum of {MAX_CREDENTIALS_PER_VAULT} credentials",
                detail={"vault_id": vault_id, "limit": MAX_CREDENTIALS_PER_VAULT},
            )
        return await queries.insert_vault_credential(
            conn,
            vault_id=vault_id,
            display_name=body.display_name,
            target_url=body.target_url,
            secret_name=body.secret_name,
            allowed_hosts=body.allowed_hosts,
            auth_type=body.auth_type,
            blob=blob,
            metadata=body.metadata,
            account_id=account_id,
        )


async def get_vault_credential(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
) -> VaultCredential:
    async with pool.acquire() as conn:
        return await queries.get_vault_credential(
            conn, vault_id, credential_id, account_id=account_id
        )


async def list_vault_credentials(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
) -> list[VaultCredential]:
    async with pool.acquire() as conn:
        return await queries.list_vault_credentials(
            conn, vault_id, limit=limit, after=after, account_id=account_id
        )


async def update_vault_credential(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    vault_id: str,
    credential_id: str,
    body: VaultCredentialUpdate,
) -> VaultCredential:
    # Serialize the decrypt-merge-encrypt-update sequence against the row
    # so two concurrent PUTs each modifying a different field (e.g. token
    # via auth-payload, display_name via column) don't both read the same
    # pre-race blob and have the second commit clobber the first's edit.
    # PR #496 fixed the in-call merge (None-unset); this transaction +
    # FOR UPDATE closes the cross-call read-modify-write race that #496
    # left uncovered. Mirrors the pattern in ``refresh_credential`` above.
    async with pool.acquire() as conn, conn.transaction():
        cred, existing_blob = await queries.get_vault_credential_with_blob(
            conn, vault_id, credential_id, account_id=account_id, for_update=True
        )
        subkey = crypto_box.derive_account_subkey(account_id)
        existing_payload = subkey.decrypt_dict(existing_blob)
        merged = _merge_auth_payload(existing_payload, body, cred.auth_type)
        new_blob = subkey.encrypt_dict(merged)

        updated = await queries.update_vault_credential(
            conn,
            vault_id,
            credential_id,
            blob=new_blob,
            display_name=(body.display_name if "display_name" in body.model_fields_set else ...),
            metadata=body.metadata if "metadata" in body.model_fields_set else ...,
            account_id=account_id,
        )
    # NOTIFY after commit (invariant 6) so the worker evicts the pooled MCP
    # session that still carries the pre-rotation secret (#1030).
    await _notify_evict_vault(pool, vault_id)
    return updated


async def archive_vault_credential(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
    idempotent: bool = False,
) -> VaultCredential:
    async with pool.acquire() as conn:
        archived = await queries.archive_vault_credential(
            conn, vault_id, credential_id, account_id=account_id, idempotent=idempotent
        )
    # Evict the pooled MCP session: the credential's secret is now retired (#1030).
    await _notify_evict_vault(pool, vault_id)
    return archived


async def delete_vault_credential(
    pool: asyncpg.Pool[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
) -> None:
    async with pool.acquire() as conn:
        await queries.delete_vault_credential(conn, vault_id, credential_id, account_id=account_id)
    # Evict the pooled MCP session: the credential row is gone (#1030).
    await _notify_evict_vault(pool, vault_id)


# ─── environment_variable resolution (#873) ──────────────────────────────────

# The opaque per-(session, credential) stand-in for an env-var secret.
# Same shape as the probed CMA reference (ANTHROPIC_SECRET_PLACEHOLDER_<32
# hex>): a distinctive greppable prefix plus 128 bits derived via HKDF from
# the account's encrypted placeholder salt. It is rotation-stable (master-key
# re-encryption does not change the salt value), scoped per owner+credential,
# stores no per-owner rows, and remains unguessable because the salt is secret,
# encrypted at rest, never logged, and never leaves the worker.
SECRET_PLACEHOLDER_PREFIX = "AIOS_SECRET_PLACEHOLDER_"


@dataclass(frozen=True, slots=True)
class ResolvedEnvVarCredential:
    """A decrypted ``environment_variable`` credential plus its minted
    placeholder, as consumed by sandbox materialization.

    ``secret_value`` is excluded from ``repr`` so a logged plan never
    prints the plaintext (an equality-assertion diff in pytest still
    drills into it — don't assert equality across differing secrets).
    ``allowed_hosts`` holds the stored canonical entries; parse with
    :func:`aios.models.vaults.parse_allowed_host_entry`.
    ``updated_at`` feeds the recycle-on-rotation drift key (#877).
    """

    credential_id: str
    secret_name: str
    secret_value: str = field(repr=False)
    allowed_hosts: tuple[str, ...]
    updated_at: datetime
    placeholder: str


def mint_secret_placeholder(account_salt: bytes, owner_id: str, credential_id: str) -> str:
    """The placeholder for ``credential_id`` owned by ``owner_id`` — a pure
    function of the account salt and the two ids."""
    digest = CryptoBox(account_salt).derive_subkey_bytes(
        f"aios-secret-placeholder-{owner_id}-{credential_id}"
    )
    return SECRET_PLACEHOLDER_PREFIX + digest[:16].hex()


async def resolve_run_env_var_credentials(
    conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    run_id: str,
    *,
    account_id: str,
) -> list[ResolvedEnvVarCredential]:
    """Decrypt every active env-var credential bound to ``run_id``.

    Run twin of :func:`resolve_session_env_var_credentials`; placeholders are
    minted with the run id as owner, so they are deterministic across run
    sandbox recycles and distinct from any session placeholder.
    """
    rows = await queries.list_run_env_var_credentials(conn, run_id, account_id=account_id)
    subkey = crypto_box.derive_account_subkey(account_id)
    account_salt = await queries.get_or_create_account_placeholder_salt(
        conn, crypto_box, account_id
    )
    return [
        ResolvedEnvVarCredential(
            credential_id=row.credential_id,
            secret_name=row.secret_name,
            secret_value=cast(str, subkey.decrypt_dict(row.blob)["secret_value"]),
            allowed_hosts=row.allowed_hosts,
            updated_at=row.updated_at,
            placeholder=mint_secret_placeholder(account_salt, run_id, row.credential_id),
        )
        for row in rows
    ]


async def resolve_session_env_var_credentials(
    conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    session_id: str,
    *,
    account_id: str,
) -> list[ResolvedEnvVarCredential]:
    """Decrypt every active env-var credential bound to ``session_id``.

    First-vault-wins on duplicate ``secret_name`` comes from the query
    (rank-ordered ``DISTINCT ON``). Decrypt failures propagate — one
    corrupt blob aborts the provision loudly rather than materializing a
    partial credential set.
    """
    rows = await queries.list_session_env_var_credentials(conn, session_id, account_id=account_id)
    subkey = crypto_box.derive_account_subkey(account_id)
    account_salt = await queries.get_or_create_account_placeholder_salt(
        conn, crypto_box, account_id
    )
    return [
        ResolvedEnvVarCredential(
            credential_id=row.credential_id,
            secret_name=row.secret_name,
            # The write path stores SecretStr.get_secret_value() — always
            # a str; cast records that contract without a masking str().
            secret_value=cast(str, subkey.decrypt_dict(row.blob)["secret_value"]),
            allowed_hosts=row.allowed_hosts,
            updated_at=row.updated_at,
            placeholder=mint_secret_placeholder(account_salt, session_id, row.credential_id),
        )
        for row in rows
    ]


def env_var_credential_containment_error(
    env_config: EnvironmentConfig | None,
    credential_allowed_hosts: Iterable[Sequence[str]],
) -> str | None:
    """Verdict for the #879 env-var-credential security gate.

    Returns a human-readable error string if the session's
    ``environment_variable`` credentials are NOT safely contained by the
    environment, or ``None`` if the configuration is acceptable (including
    the no-credentials common path).

    ``credential_allowed_hosts`` is one ``allowed_hosts`` tuple per bound
    env-var credential (the stored canonical entries). The caller supplies
    these already-fetched — this function does no DB I/O so it stays pure
    and unit-testable.

    Permit-with-warning posture (#1153), matching the CMA spec the #871
    epic ports: env-var credentials are accepted under **any** networking
    mode. The secret-swap chokepoint (the credential-host → proxy DNAT)
    now installs under Unrestricted too (DNAT-only, no lockdown), so the
    swap fires regardless of the environment networking; the operator
    accepts the loss of exfil-containment as an informed, CMA-documented
    choice (the warning is logged at provision in ``spec.py``).

    The one residual check applies **only under Limited**:

    - **Limited** → every credential host must appear in the env's
      ``allowed_hosts`` host set (egress-escalation prevention: the DNAT
      sits in nat-OUTPUT *ahead of* the filter DROP, so a credential host
      outside the allowlist would manufacture a forbidden egress channel).
      Host parts are compared via ``parse_allowed_host_entry`` on BOTH
      sides — the single grammar authority — so a credential path-prefix
      only tightens below an already-allowed host.
    - **Unrestricted / None** → ``None`` (accepted). There is no filter
      DROP under Unrestricted, so every host is reachable anyway and the
      subset check guards no real escalation hole — its rationale
      evaporates. (``env_config=None`` and ``env_config.networking=None``
      are both treated as unrestricted at runtime, so this is one runtime
      case.)

    The two sides parse asymmetrically. ``LimitedNetworking`` validates
    its ``allowed_hosts`` against ``HOSTNAME_RE``, which accepts IP
    literals; the credential grammar (``parse_allowed_host_entry`` →
    ``_validate_credential_host``) is stricter and rejects them. So an env
    entry can be a bare IP that the credential parse refuses. The ENV-side
    extraction therefore skips entries ``parse_allowed_host_entry`` rejects
    (an IP env host can never cover a non-IP credential host anyway — fail
    closed, the credential is then rejected with the actionable message).
    The CREDENTIAL side does NOT catch ``ValueError``: credential hosts are
    canonicalized at create, so a malformed stored credential entry is a
    real integrity problem and should surface (at provision it becomes a
    refusal to provision, the safe direction).
    """
    cred_lists = list(credential_allowed_hosts)
    # Iterate the OUTER list, not flattened hosts: a credential with an
    # EMPTY allowed_hosts tuple still counts as "has a credential". Only a
    # session with zero env-var credentials skips the gate entirely.
    if not cred_lists:
        return None

    networking = env_config.networking if env_config else None
    if not isinstance(networking, LimitedNetworking):
        # Permit-with-warning (#1153): under Unrestricted (or no env / no
        # networking config) the secret swap still fires via the DNAT-only
        # chokepoint, and with no filter DROP there is no egress-escalation
        # hole for the host-subset check to guard. So the credential is
        # accepted; the operator-facing exfil-containment warning is logged at
        # provision (spec.py), not raised here.
        return None

    # DNS hostnames are case-insensitive but ``HOSTNAME_RE`` (and thus
    # ``parse_allowed_host_entry``) preserves the stored casing, so case-fold
    # both sides — mirrors ``_validate_credential_host`` in ``models/vaults.py``,
    # which documents that downstream comparisons (this is one) case-fold.
    env_hosts: set[str] = set()
    for entry in networking.allowed_hosts:
        try:
            env_hosts.add(parse_allowed_host_entry(entry)[0].lower())
        except ValueError:
            # LimitedNetworking validates allowed_hosts with HOSTNAME_RE, which
            # accepts IP literals that parse_allowed_host_entry (the stricter
            # credential grammar) rejects. Such an env entry can never cover a
            # credential host — credential hosts are validated non-IP hostnames
            # — so drop it from the comparable set. Fail closed: an otherwise
            # uncovered credential host is then rejected with the actionable
            # message instead of crashing the advisory 422 / provision gate.
            continue
    for cred_hosts in cred_lists:
        for entry in cred_hosts:
            # Keep the stored spelling for the message; compare case-folded.
            cred_host = parse_allowed_host_entry(entry)[0]
            if cred_host.lower() not in env_hosts:
                return (
                    "environment_variable credential requires a Limited environment whose "
                    f"allowed_hosts cover the credential's hosts (credential host {cred_host!r} "
                    "not covered)"
                )
    return None
