"""Interactive OAuth "Connect" flow for vault credentials.

This is the server-side half of the console's one-click "Connect": instead of
pasting tokens, the user authorizes an MCP server interactively. The flow spans
two calls:

* :func:`start_oauth_flow` — discover the target's OAuth metadata (RFC 9728
  protected-resource → RFC 8414 authorization-server), register a client
  (RFC 7591 Dynamic Client Registration) or use a caller-supplied client,
  generate PKCE (RFC 7636) + a CSRF ``state``, persist the encrypted flow, and
  return the provider's authorization URL to redirect the user to.
* :func:`complete_oauth_flow` — (see below) exchange the returned code for
  tokens and store them as an ``oauth2_refresh`` credential.

It reuses the MCP SDK's standalone request-builders/response-handlers
(``mcp.client.auth.utils``) and data models (``mcp.shared.auth``) rather than
its inline ``OAuthClientProvider``, which is coupled to a live transport. The
token-endpoint client-auth (none/basic/post) is shared with the refresh path
via :func:`aios.services.vaults.build_token_endpoint_post`.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlparse

import asyncpg
import httpx
from mcp.client.auth import OAuthTokenError
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_registration_request,
    create_oauth_metadata_request,
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    handle_registration_response,
    handle_token_response_scopes,
)
from mcp.shared.auth import (
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import AnyUrl, SecretStr

from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.errors import ConflictError, OAuthFlowError, ValidationError
from aios.logging import get_logger
from aios.models.vaults import (
    OAuthCompleteRequest,
    OAuthProviderApp,
    OAuthStartRequest,
    OAuthStartResponse,
    TokenEndpointAuth,
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
    VaultCredential,
    VaultCredentialCreate,
    VaultCredentialUpdate,
)
from aios.services.vaults import (
    build_token_endpoint_post,
    create_vault_credential,
    update_vault_credential,
)
from aios.tools.url_safety import is_safe_url

log = get_logger("aios.services.vault_oauth")

# In-progress flows are short-lived: the user signs in at the provider then
# returns. Ten minutes is generous headroom for a manual sign-in.
OAUTH_FLOW_TTL_SECONDS = 600

# Generous per-call HTTP timeout — OAuth providers vary widely.
_OAUTH_HTTP_TIMEOUT_SECONDS = 30


# ── SSRF guard ───────────────────────────────────────────────────────────────


async def _guard_url(url: str, *, allow_insecure: frozenset[str], label: str) -> None:
    """Reject a server-side-fetched URL that could reach an internal host.

    Operator-allowlisted hosts (dev only — ``Settings.oauth_allow_insecure_hosts``)
    bypass the check so a self-hosted MCP fleet on plain http can be connected
    in dev. Otherwise requires https and delegates to the shared
    :func:`aios.tools.url_safety.is_safe_url`, which resolves DNS and blocks
    private/CGNAT/loopback/link-local/cloud-metadata addresses — the same guard
    ``web_fetch``/``http_request`` use, so the policy can't drift. ``is_safe_url``
    is blocking (``getaddrinfo``), so it runs in a worker thread.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in allow_insecure or parsed.netloc.lower() in allow_insecure:
        return
    if parsed.scheme != "https":
        raise OAuthFlowError(f"{label} must use https", detail={label: url})
    if not await asyncio.to_thread(is_safe_url, url):
        raise OAuthFlowError(f"{label} host is not allowed", detail={label: url})


def _assert_token_endpoint_bound(app: OAuthProviderApp, metadata: OAuthMetadata) -> None:
    """Refuse to use the operator client unless the discovered token endpoint
    belongs to ``app``.

    The operator's confidential ``client_secret`` is POSTed to the discovered
    ``token_endpoint`` at completion. ``_match_provider_app`` selects the app on
    the issuer / authorization-endpoint / target host, all of which come from
    the (untrusted) target's discovery metadata. Without this check an attacker
    could serve metadata advertising ``accounts.google.com`` as the issuer (to
    select the operator's Google app) while pointing ``token_endpoint`` at a host
    they control, exfiltrating the secret. Bind it: the token host must be
    ``match`` (or a sub-host) or one of ``token_endpoint_hosts`` (e.g. Google
    issues at ``oauth2.googleapis.com``, a different apex than the authorize
    host).
    """
    allowed = {_norm_host(app.match)}
    allowed.update(_norm_host(h) for h in app.token_endpoint_hosts if h.strip())
    token_host = _host(str(metadata.token_endpoint))
    if not token_host or not _host_covered_by(token_host, allowed):
        raise OAuthFlowError(
            "discovered token endpoint is not trusted for this operator app",
            detail={"token_endpoint": str(metadata.token_endpoint), "match": app.match},
        )


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _host(url: str | None) -> str:
    return (urlparse(url).hostname or "").lower() if url else ""


def _norm_host(value: str) -> str:
    """Normalize a host / ``match`` pattern for comparison (lowercase, no leading dot)."""
    return value.strip().lower().lstrip(".")


def _host_covered_by(host: str, patterns: set[str]) -> bool:
    """True if ``host`` equals or is a dotted sub-host of any pattern.

    Single source of the provider-app host-match rule, shared by the app
    *selection* path (:func:`_match_provider_app`) and the secret-binding path
    (:func:`_assert_token_endpoint_bound`) so a future tightening can't make the
    two security checks drift. ``patterns`` must be pre-normalized via
    :func:`_norm_host`.
    """
    return any(host == p or host.endswith("." + p) for p in patterns)


def _match_provider_app(
    apps: Sequence[OAuthProviderApp], metadata: OAuthMetadata, target_url: str
) -> OAuthProviderApp | None:
    """The operator-configured app whose ``match`` host covers this server.

    Matches against the discovered OAuth issuer, authorization endpoint, or the
    target URL host — so e.g. ``accounts.google.com`` covers every Google MCP
    server. Exact host or dotted-suffix match.
    """
    hosts = {
        _host(str(metadata.issuer)),
        _host(str(metadata.authorization_endpoint)),
        _host(target_url),
    }
    hosts.discard("")
    for app in apps:
        m = _norm_host(app.match)
        if not m:
            continue
        if any(_host_covered_by(h, {m}) for h in hosts):
            return app
    return None


# ── metadata discovery ───────────────────────────────────────────────────────


async def _discover_protected_resource(
    client: httpx.AsyncClient, server_url: str
) -> ProtectedResourceMetadata | None:
    """Try the RFC 9728 protected-resource well-known URLs in priority order."""
    for url in build_protected_resource_metadata_discovery_urls(None, server_url):
        try:
            resp = await client.send(create_oauth_metadata_request(url))
        except httpx.HTTPError:
            continue
        prm = await handle_protected_resource_response(resp)
        if prm is not None:
            return prm
    return None


async def _discover_auth_server(
    client: httpx.AsyncClient, auth_server_url: str | None, server_url: str
) -> OAuthMetadata | None:
    """Try the RFC 8414 / OIDC authorization-server metadata URLs in order."""
    for url in build_oauth_authorization_server_metadata_discovery_urls(
        auth_server_url, server_url
    ):
        try:
            resp = await client.send(create_oauth_metadata_request(url))
        except httpx.HTTPError:
            continue
        should_continue, metadata = await handle_auth_metadata_response(resp)
        if metadata is not None:
            return metadata
        if not should_continue:
            break
    return None


# ── start ────────────────────────────────────────────────────────────────────


async def start_oauth_flow(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    vault_id: str,
    body: OAuthStartRequest,
    provider_apps: Sequence[OAuthProviderApp] | None = None,
) -> OAuthStartResponse:
    """Begin an interactive authorization-code flow and return the authorize URL.

    Verifies vault ownership, discovers the target's OAuth metadata, obtains a
    client, generates PKCE + state, persists the encrypted flow (TTL ~10 min),
    and returns the provider authorization URL.

    The client is resolved in priority order: a caller-supplied ``client_id`` →
    an operator-configured provider app (``provider_apps``, default
    ``Settings.oauth_provider_apps``) for servers without DCR → Dynamic Client
    Registration → otherwise an error asking for client credentials.
    """
    settings = get_settings()
    apps = provider_apps if provider_apps is not None else settings.oauth_provider_apps
    allow_insecure = settings.oauth_allow_insecure_host_set
    await _guard_url(body.target_url, allow_insecure=allow_insecure, label="target_url")

    # Ownership check (raises NotFoundError if the vault isn't this account's)
    # and opportunistic prune of expired flows.
    async with pool.acquire() as conn:
        await queries.get_vault(conn, vault_id, account_id=account_id)
        await queries.delete_expired_oauth_flows(conn)

    # follow_redirects is OFF: is_safe_url validates the literal host, but a 3xx
    # to an internal host would bypass it (the redirect target is never guarded).
    # RFC 9728/8414 discovery hits well-known paths directly, so no redirect is
    # needed for a spec-compliant provider.
    async with httpx.AsyncClient(
        timeout=_OAUTH_HTTP_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        prm = await _discover_protected_resource(client, body.target_url)
        auth_server_url = (
            str(prm.authorization_servers[0]) if prm and prm.authorization_servers else None
        )
        # The authorization server is discovered from the target's (untrusted)
        # metadata, then fetched server-side — guard it before the fetch.
        if auth_server_url:
            await _guard_url(
                auth_server_url, allow_insecure=allow_insecure, label="authorization_server"
            )
        metadata = await _discover_auth_server(client, auth_server_url, body.target_url)
        if metadata is None:
            raise OAuthFlowError(
                "could not discover OAuth metadata for this server",
                detail={"target_url": body.target_url},
            )

        # The token endpoint is persisted verbatim and POSTed at completion —
        # guard the discovered value now so the SSRF check covers that use.
        await _guard_url(
            str(metadata.token_endpoint), allow_insecure=allow_insecure, label="token_endpoint"
        )

        scope = body.scope or get_client_metadata_scopes(None, prm, metadata)
        # RFC 8707 resource indicator: the MCP server's canonical URI.
        resource = str(prm.resource) if prm else body.target_url

        # Resolved client identity. ``method`` is stored as a plain string and
        # interpreted by ``build_token_endpoint_post`` (anything other than
        # client_secret_basic/post behaves as a public "none" client).
        client_id: str
        client_secret: str | None
        method: str
        provider_app = _match_provider_app(apps, metadata, body.target_url)
        if body.client_id:
            # Caller supplied a pre-registered client (server without DCR).
            client_id = body.client_id
            client_secret = body.client_secret.get_secret_value() if body.client_secret else None
            method = body.token_endpoint_auth_method or (
                "client_secret_post" if client_secret else "none"
            )
        elif provider_app:
            # Operator-registered app for this provider — the user supplies
            # nothing (the CMA model for non-DCR providers like Google). Bind
            # the secret to the discovered token endpoint before using it, so
            # spoofed issuer/authz metadata can't redirect it to an attacker.
            _assert_token_endpoint_bound(provider_app, metadata)
            client_id = provider_app.client_id
            client_secret = (
                provider_app.client_secret.get_secret_value()
                if provider_app.client_secret
                else None
            )
            method = provider_app.token_endpoint_auth_method
            if provider_app.scope:
                scope = provider_app.scope
        elif metadata.registration_endpoint:
            # Dynamic Client Registration (RFC 7591) — public client + PKCE.
            # The registration endpoint is fetched server-side; guard it.
            await _guard_url(
                str(metadata.registration_endpoint),
                allow_insecure=allow_insecure,
                label="registration_endpoint",
            )
            client_metadata = OAuthClientMetadata(
                redirect_uris=[AnyUrl(body.redirect_uri)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
                scope=scope,
            )
            reg_resp = await client.send(
                create_client_registration_request(
                    metadata, client_metadata, _origin(auth_server_url or body.target_url)
                )
            )
            client_info = await handle_registration_response(reg_resp)
            if not client_info.client_id:
                raise OAuthFlowError(
                    "dynamic client registration returned no client_id",
                    detail={"target_url": body.target_url},
                )
            client_id = client_info.client_id
            client_secret = client_info.client_secret
            method = client_info.token_endpoint_auth_method or "none"
        else:
            raise OAuthFlowError(
                "this MCP server does not support automatic client registration; "
                "provide an OAuth client ID (and secret) under OAuth client credentials",
                detail={"target_url": body.target_url},
            )

    pkce = PKCEParameters.generate()
    state = secrets.token_urlsafe(32)

    auth_params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": body.redirect_uri,
        "state": state,
        "code_challenge": pkce.code_challenge,
        "code_challenge_method": "S256",
    }
    if scope:
        auth_params["scope"] = scope
    if resource:
        auth_params["resource"] = resource
    # Provider-specific extras (e.g. Google needs access_type=offline +
    # prompt=consent to return a refresh token). Standard params win on conflict.
    if provider_app and provider_app.authorize_params:
        for k, v in provider_app.authorize_params.items():
            auth_params.setdefault(k, v)
    authorization_url = f"{metadata.authorization_endpoint}?{urlencode(auth_params)}"

    # Persist the flow state, encrypted under the per-account subkey (it carries
    # the PKCE code_verifier and possibly a client_secret).
    flow_payload: dict[str, Any] = {
        "code_verifier": pkce.code_verifier,
        "client_id": client_id,
        "token_endpoint": str(metadata.token_endpoint),
        "token_endpoint_auth_method": method,
    }
    if client_secret:
        flow_payload["client_secret"] = client_secret
    if scope:
        flow_payload["scope"] = scope
    if resource:
        flow_payload["resource"] = resource
    if body.display_name:
        flow_payload["display_name"] = body.display_name

    blob = crypto_box.derive_account_subkey(account_id).encrypt_dict(flow_payload)
    expires_at = datetime.now(UTC) + timedelta(seconds=OAUTH_FLOW_TTL_SECONDS)

    async with pool.acquire() as conn:
        flow_id = await queries.insert_oauth_flow(
            conn,
            account_id=account_id,
            vault_id=vault_id,
            target_url=body.target_url,
            state=state,
            redirect_uri=body.redirect_uri,
            blob=blob,
            expires_at=expires_at,
        )

    log.info(
        "vault.oauth_flow_started",
        flow_id=flow_id,
        vault_id=vault_id,
        target_url=body.target_url,
        registered=bool(not body.client_id),
    )
    return OAuthStartResponse(flow_id=flow_id, state=state, authorization_url=authorization_url)


# ── complete ─────────────────────────────────────────────────────────────────


def _token_endpoint_auth_model(method: str, client_secret: str | None) -> TokenEndpointAuth:
    """Reconstruct the typed token_endpoint_auth from the stored flow fields."""
    if method == "client_secret_basic" and client_secret:
        return TokenEndpointAuthBasic(
            method="client_secret_basic", client_secret=SecretStr(client_secret)
        )
    if method == "client_secret_post" and client_secret:
        return TokenEndpointAuthPost(
            method="client_secret_post", client_secret=SecretStr(client_secret)
        )
    return TokenEndpointAuthNone(method="none")


async def _exchange_code(
    flow_payload: dict[str, Any], *, redirect_uri: str, code: str
) -> OAuthToken:
    """Exchange the authorization code for tokens at the stored token endpoint.

    Reuses the same client-auth handling as the refresh path. The
    ``redirect_uri`` is the one persisted at start (byte-identical to the one
    registered and sent to /authorize).
    """
    token_endpoint = str(flow_payload["token_endpoint"])
    client_id = str(flow_payload["client_id"])
    body: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": str(flow_payload["code_verifier"]),
    }
    resource = flow_payload.get("resource")
    if resource:
        body["resource"] = str(resource)
    endpoint_auth: dict[str, str] = {
        "method": str(flow_payload.get("token_endpoint_auth_method", "none"))
    }
    secret = flow_payload.get("client_secret")
    if secret:
        endpoint_auth["client_secret"] = str(secret)
    post_kwargs = build_token_endpoint_post(body, client_id=client_id, endpoint_auth=endpoint_auth)

    # follow_redirects OFF: the token_endpoint was SSRF-guarded at start and is
    # stored encrypted, but a token-exchange POST should never be redirected
    # anyway — a 3xx here can only point somewhere we didn't validate.
    async with httpx.AsyncClient(
        timeout=_OAUTH_HTTP_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        try:
            resp = await client.post(token_endpoint, **post_kwargs)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("vault.oauth_exchange_http_error", token_endpoint=token_endpoint)
            raise OAuthFlowError(
                "OAuth token exchange failed",
                detail={"token_endpoint": token_endpoint},
            ) from exc
        try:
            return await handle_token_response_scopes(resp)
        except OAuthTokenError as exc:
            raise OAuthFlowError(
                "OAuth token endpoint returned an invalid response",
                detail={"token_endpoint": token_endpoint},
            ) from exc


async def _store_oauth_credential(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    vault_id: str,
    target_url: str,
    token: OAuthToken,
    flow_payload: dict[str, Any],
    existing: VaultCredential | None,
) -> VaultCredential:
    """Create or rotate the ``oauth2_refresh`` credential for ``target_url``.

    ``existing`` is the active credential observed before the exchange (already
    confirmed by the caller to be ``oauth2_refresh`` or ``None``). On a create
    that loses a race with a concurrent completion, the active-row unique index
    is the serialization point: catch the conflict, re-read, and rotate — so a
    double-submit doesn't 409 the user after they consented.
    """
    expires_at = (
        datetime.now(UTC) + timedelta(seconds=token.expires_in) if token.expires_in else None
    )
    client_secret = flow_payload.get("client_secret")
    endpoint_auth = _token_endpoint_auth_model(
        str(flow_payload.get("token_endpoint_auth_method", "none")),
        str(client_secret) if client_secret else None,
    )
    display_name = flow_payload.get("display_name")

    secret_fields: dict[str, Any] = {
        "access_token": SecretStr(token.access_token),
        "token_endpoint": str(flow_payload["token_endpoint"]),
        "client_id": str(flow_payload["client_id"]),
        "token_endpoint_auth": endpoint_auth,
        "scope": token.scope or flow_payload.get("scope"),
        "resource": flow_payload.get("resource"),
        "expires_at": expires_at,
    }
    if token.refresh_token:
        secret_fields["refresh_token"] = SecretStr(token.refresh_token)

    if existing is None:
        create_body = VaultCredentialCreate(
            target_url=target_url,
            auth_type="oauth2_refresh",
            display_name=display_name,
            **secret_fields,
        )
        try:
            return await create_vault_credential(
                pool, crypto_box, account_id=account_id, vault_id=vault_id, body=create_body
            )
        except ConflictError:
            # Lost a race with a concurrent completion for the same target_url —
            # the active-row unique index serialized us. Re-read and rotate.
            async with pool.acquire() as conn:
                raced = await queries.get_active_credential_by_target_url(
                    conn, account_id=account_id, vault_id=vault_id, target_url=target_url
                )
            if raced is None or raced.auth_type != "oauth2_refresh":
                raise
            existing = raced

    # Rotate. Omit fields the exchange didn't return (e.g. no new refresh_token)
    # so they're preserved — EXCEPT expires_at, which is written even when None
    # so a token that came back without expires_in clears the prior (now-stale,
    # possibly-past) value instead of inheriting it. Inheriting a past expires_at
    # would make is_expiring() perpetually true and force a refresh before every
    # call (a hard failure when the provider issued no refresh_token).
    update_fields = {k: v for k, v in secret_fields.items() if v is not None or k == "expires_at"}
    if display_name:
        update_fields["display_name"] = display_name
    return await update_vault_credential(
        pool,
        crypto_box,
        account_id=account_id,
        vault_id=vault_id,
        credential_id=existing.id,
        body=VaultCredentialUpdate(**update_fields),
    )


async def complete_oauth_flow(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    vault_id: str,
    body: OAuthCompleteRequest,
) -> VaultCredential:
    """Finish an interactive flow: exchange the code and store the credential.

    Reads the flow under a row lock and decrypts it, rejects a conflicting
    ``auth_type`` at the target URL *before* burning the single-use code,
    exchanges the code for tokens, then stores them as an ``oauth2_refresh``
    credential — creating one or **rotating** the existing credential for the
    same ``(vault_id, target_url)`` — and finally deletes the flow row. The flow
    is consumed only on success: a transient exchange failure leaves it reusable
    within its TTL instead of forcing the user to re-consent from scratch.
    """
    # Read + decrypt the flow under a row lock. NOT deleted yet — single-use is
    # enforced by deleting on success (below), so a transient exchange failure
    # doesn't burn the flow + auth code.
    async with pool.acquire() as conn, conn.transaction():
        flow = await queries.get_oauth_flow_for_complete(
            conn, account_id=account_id, vault_id=vault_id, state=body.state
        )
        if flow is None:
            raise ValidationError(
                "unknown or expired OAuth flow",
                detail={"state": body.state},
            )
        flow_id, target_url, redirect_uri, blob = flow
        flow_payload = crypto_box.derive_account_subkey(account_id).decrypt_dict(blob)

    # Reject a different auth_type at this target_url BEFORE the exchange burns
    # the single-use code (auth_type is immutable — an OAuth connect can't
    # overwrite e.g. a bearer credential, so don't waste the user's consent).
    async with pool.acquire() as conn:
        existing = await queries.get_active_credential_by_target_url(
            conn, account_id=account_id, vault_id=vault_id, target_url=target_url
        )
    if existing is not None and existing.auth_type != "oauth2_refresh":
        raise ConflictError(
            f"an active {existing.auth_type} credential already exists for {target_url}; "
            "archive it before connecting via OAuth",
            detail={
                "vault_id": vault_id,
                "target_url": target_url,
                "auth_type": existing.auth_type,
            },
        )

    # Token exchange (network; no locks held; flow row intact for retry on a
    # transient failure).
    token = await _exchange_code(flow_payload, redirect_uri=redirect_uri, code=body.code)

    # Store as an oauth2_refresh credential (create or rotate), then consume the
    # flow — single-use is enforced on success.
    cred = await _store_oauth_credential(
        pool,
        crypto_box,
        account_id=account_id,
        vault_id=vault_id,
        target_url=target_url,
        token=token,
        flow_payload=flow_payload,
        existing=existing,
    )

    async with pool.acquire() as conn:
        await queries.delete_oauth_flow(conn, flow_id)

    log.info(
        "vault.oauth_flow_completed",
        vault_id=vault_id,
        target_url=target_url,
        credential_id=cred.id,
        rotated=existing is not None,
    )
    return cred
