"""E2E tests for the interactive OAuth "Connect" flow (services.vault_oauth).

Real testcontainer Postgres + a mocked OAuth/MCP provider. The provider is an
``httpx.MockTransport`` patched onto ``aios.services.vault_oauth.httpx.AsyncClient``
so the real MCP SDK request-builders/response-handlers run against it — exercising
discovery (RFC 9728 → RFC 8414), Dynamic Client Registration (RFC 7591), PKCE, and
the authorization-code token exchange end-to-end without a browser or network.
"""

from __future__ import annotations

import json as _json
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import SecretStr

from aios.crypto.vault import EncryptedBlob
from aios.errors import ConflictError, OAuthFlowError, ValidationError
from aios.models.vaults import OAuthCompleteRequest, OAuthStartRequest, VaultCredentialCreate
from aios.services import vault_oauth as svc
from aios.services import vaults as vaults_svc

ACCOUNT_ID = "acc_test_stub"
TARGET_URL = "https://mock-mcp.example.com/mcp"
REDIRECT_URI = "https://console.example.com/api/auth/mcp-oauth/callback"


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


def _make_handler(*, registration: bool = True, token_calls: list[dict[str, Any]] | None = None):
    """A MockTransport handler emulating a spec-compliant MCP OAuth provider."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/.well-known/oauth-protected-resource" in path:
            return httpx.Response(
                200,
                json={
                    "resource": TARGET_URL,
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["read", "write"],
                },
            )
        if (
            "/.well-known/oauth-authorization-server" in path
            or "/.well-known/openid-configuration" in path
        ):
            metadata: dict[str, Any] = {
                "issuer": "https://auth.example.com",
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
                "scopes_supported": ["read", "write"],
                "response_types_supported": ["code"],
            }
            if registration:
                metadata["registration_endpoint"] = "https://auth.example.com/register"
            return httpx.Response(200, json=metadata)
        if path.endswith("/register"):
            return httpx.Response(
                201,
                json={
                    "client_id": "dyn-client-123",
                    "redirect_uris": [REDIRECT_URI],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                },
            )
        if path.endswith("/token"):
            if token_calls is not None:
                token_calls.append(dict(httpx.QueryParams(request.read().decode())))
            return httpx.Response(
                200,
                json={
                    "access_token": "at-xyz",
                    "refresh_token": "rt-xyz",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "read write",
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler


# Captured before any patching — the factory uses the real class so patching
# `vault_oauth.httpx.AsyncClient` (the shared module attribute) doesn't recurse.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


@contextmanager
def _patched_provider(**kwargs: Any):
    handler = _make_handler(**kwargs)

    def factory(*_a: Any, **_k: Any) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

    with patch("aios.services.vault_oauth.httpx.AsyncClient", factory):
        yield


async def _vault(pool: Any, name: str) -> str:
    v = await vaults_svc.create_vault(pool, display_name=name, metadata={}, account_id=ACCOUNT_ID)
    return str(v.id)


def _decrypt_flow(crypto_box: Any, blob: EncryptedBlob) -> dict[str, Any]:
    return crypto_box.derive_account_subkey(ACCOUNT_ID).decrypt_dict(blob)


class TestStart:
    async def test_builds_authorize_url_and_persists_encrypted_flow(
        self, pool: Any, crypto_box: Any
    ) -> None:
        vault_id = await _vault(pool, "oauth-start")
        with _patched_provider():
            res = await svc.start_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthStartRequest(target_url=TARGET_URL, redirect_uri=REDIRECT_URI),
            )

        parsed = urlparse(res.authorization_url)
        assert (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            == "https://auth.example.com/authorize"
        )
        q = parse_qs(parsed.query)
        assert q["response_type"] == ["code"]
        assert q["client_id"] == ["dyn-client-123"]
        assert q["redirect_uri"] == [REDIRECT_URI]
        assert q["state"] == [res.state]
        assert q["code_challenge_method"] == ["S256"]
        assert q["code_challenge"][0]
        assert q["resource"] == [TARGET_URL]
        assert q["scope"] == ["read write"]

        # The flow row exists and decrypts to the PKCE verifier + registered client.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ciphertext, nonce, redirect_uri, target_url FROM oauth_flows WHERE state = $1",
                res.state,
            )
        assert row is not None
        assert row["redirect_uri"] == REDIRECT_URI
        payload = _decrypt_flow(
            crypto_box,
            EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"])),
        )
        assert payload["client_id"] == "dyn-client-123"
        assert payload["token_endpoint"] == "https://auth.example.com/token"
        assert len(payload["code_verifier"]) >= 43

    async def test_rejects_insecure_target(self, pool: Any, crypto_box: Any) -> None:
        vault_id = await _vault(pool, "oauth-ssrf")
        for bad in ("http://mcp.example.com/mcp", "https://localhost/mcp", "https://127.0.0.1/mcp"):
            with pytest.raises(OAuthFlowError):
                await svc.start_oauth_flow(
                    pool,
                    crypto_box,
                    account_id=ACCOUNT_ID,
                    vault_id=vault_id,
                    body=OAuthStartRequest(target_url=bad, redirect_uri=REDIRECT_URI),
                )

    async def test_requires_client_when_no_dcr(self, pool: Any, crypto_box: Any) -> None:
        vault_id = await _vault(pool, "oauth-nodcr")
        with _patched_provider(registration=False), pytest.raises(OAuthFlowError):
            await svc.start_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthStartRequest(target_url=TARGET_URL, redirect_uri=REDIRECT_URI),
            )


class TestComplete:
    async def _start(self, pool: Any, crypto_box: Any, vault_id: str) -> str:
        with _patched_provider():
            res = await svc.start_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthStartRequest(
                    target_url=TARGET_URL, redirect_uri=REDIRECT_URI, display_name="My Notion"
                ),
            )
        return res.state

    async def test_creates_oauth_credential(self, pool: Any, crypto_box: Any) -> None:
        vault_id = await _vault(pool, "oauth-complete")
        state = await self._start(pool, crypto_box, vault_id)

        token_calls: list[dict[str, Any]] = []
        with _patched_provider(token_calls=token_calls):
            cred = await svc.complete_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthCompleteRequest(state=state, code="auth-code-1"),
            )

        assert cred.auth_type == "oauth2_refresh"
        assert cred.display_name == "My Notion"
        assert cred.target_url == TARGET_URL
        # The exchange POST carried the auth-code grant + PKCE verifier.
        assert token_calls and token_calls[0]["grant_type"] == "authorization_code"
        assert token_calls[0]["code"] == "auth-code-1"
        assert token_calls[0]["redirect_uri"] == REDIRECT_URI
        assert token_calls[0]["code_verifier"]

        # Stored payload decrypts to the exchanged tokens.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT ciphertext, nonce FROM vault_credentials WHERE id = $1", cred.id
            )
            flow_left = await conn.fetchrow("SELECT 1 FROM oauth_flows WHERE state = $1", state)
        payload = _json.loads(
            crypto_box.derive_account_subkey(ACCOUNT_ID).decrypt(
                EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
            )
        )
        assert payload["access_token"] == "at-xyz"
        assert payload["refresh_token"] == "rt-xyz"
        assert payload["client_id"] == "dyn-client-123"
        assert payload["token_endpoint"] == "https://auth.example.com/token"
        assert flow_left is None, "flow row should be deleted (single-use)"

    async def test_rotates_existing_credential_without_conflict(
        self, pool: Any, crypto_box: Any
    ) -> None:
        vault_id = await _vault(pool, "oauth-rotate")
        # First connect.
        state1 = await self._start(pool, crypto_box, vault_id)
        with _patched_provider():
            first = await svc.complete_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthCompleteRequest(state=state1, code="code-1"),
            )
        # Second connect to the same target_url must rotate, not 409.
        state2 = await self._start(pool, crypto_box, vault_id)
        with _patched_provider():
            second = await svc.complete_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthCompleteRequest(state=state2, code="code-2"),
            )
        assert second.id == first.id, "re-connect should rotate the same credential row"
        creds = await vaults_svc.list_vault_credentials(pool, vault_id, account_id=ACCOUNT_ID)
        assert len([c for c in creds if c.target_url == TARGET_URL]) == 1

    async def test_rejects_different_auth_type_at_same_url(
        self, pool: Any, crypto_box: Any
    ) -> None:
        vault_id = await _vault(pool, "oauth-collide")
        # A bearer credential already occupies this target_url.
        await vaults_svc.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACCOUNT_ID,
            vault_id=vault_id,
            body=VaultCredentialCreate(
                target_url=TARGET_URL,
                auth_type="bearer_header",
                token=SecretStr("tok"),
            ),
        )
        state = await self._start(pool, crypto_box, vault_id)

        with _patched_provider(), pytest.raises(ConflictError):
            await svc.complete_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthCompleteRequest(state=state, code="code-x"),
            )

    async def test_rejects_unknown_state(self, pool: Any, crypto_box: Any) -> None:
        vault_id = await _vault(pool, "oauth-badstate")
        with pytest.raises(ValidationError):
            await svc.complete_oauth_flow(
                pool,
                crypto_box,
                account_id=ACCOUNT_ID,
                vault_id=vault_id,
                body=OAuthCompleteRequest(state="nope", code="code"),
            )
