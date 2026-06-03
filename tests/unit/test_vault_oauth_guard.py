"""Unit tests for the interactive-OAuth SSRF guard, the operator client_secret
endpoint-binding, and the provider-app secret/method validator.

These cover the security-critical pure functions in ``services.vault_oauth`` and
the ``OAuthProviderApp`` model without a DB or network — the end-to-end flow is
exercised in ``tests/e2e/test_vault_oauth.py``.
"""

from __future__ import annotations

import pytest
from mcp.shared.auth import OAuthMetadata
from pydantic import SecretStr, ValidationError

from aios.errors import OAuthFlowError
from aios.models.vaults import OAuthProviderApp
from aios.services.vault_oauth import _assert_token_endpoint_bound, _guard_url


def _metadata(
    token_endpoint: str,
    *,
    issuer: str = "https://accounts.google.com",
    authz: str = "https://accounts.google.com/o/oauth2/v2/auth",
) -> OAuthMetadata:
    return OAuthMetadata.model_validate(
        {
            "issuer": issuer,
            "authorization_endpoint": authz,
            "token_endpoint": token_endpoint,
            "response_types_supported": ["code"],
        }
    )


def _app(match: str, token_endpoint_hosts: tuple[str, ...] = ()) -> OAuthProviderApp:
    return OAuthProviderApp(
        match=match,
        client_id="cid",
        client_secret=SecretStr("sec"),
        token_endpoint_hosts=list(token_endpoint_hosts),
    )


# ── SSRF guard (_guard_url) ──────────────────────────────────────────────────


class TestGuardUrl:
    async def test_requires_https(self) -> None:
        with pytest.raises(OAuthFlowError):
            await _guard_url(
                "http://mcp.example.com/x", allow_insecure=frozenset(), label="target_url"
            )

    async def test_blocks_loopback_via_is_safe_url(self) -> None:
        with pytest.raises(OAuthFlowError):
            await _guard_url(
                "https://127.0.0.1/mcp", allow_insecure=frozenset(), label="token_endpoint"
            )

    async def test_allowlist_bypasses_https_and_private_block(self) -> None:
        # Host form and host:port form both bypass (matches the dev fleet config).
        await _guard_url(
            "http://workspace-mcp/mcp",
            allow_insecure=frozenset({"workspace-mcp"}),
            label="target_url",
        )
        await _guard_url(
            "http://workspace-mcp:8000/mcp",
            allow_insecure=frozenset({"workspace-mcp:8000"}),
            label="target_url",
        )

    async def test_error_carries_label(self) -> None:
        with pytest.raises(OAuthFlowError) as ei:
            await _guard_url(
                "http://x.example/y", allow_insecure=frozenset(), label="registration_endpoint"
            )
        assert "registration_endpoint" in str(ei.value)


# ── operator client_secret endpoint-binding ─────────────────────────────────


class TestTokenEndpointBinding:
    def test_google_split_apex_passes_with_token_hosts(self) -> None:
        # Google authorizes at accounts.google.com but issues tokens at
        # oauth2.googleapis.com — only allowed when listed in token_endpoint_hosts.
        app = _app("accounts.google.com", ("oauth2.googleapis.com",))
        _assert_token_endpoint_bound(app, _metadata("https://oauth2.googleapis.com/token"))

    def test_google_split_apex_rejected_without_token_hosts(self) -> None:
        app = _app("accounts.google.com")
        with pytest.raises(OAuthFlowError):
            _assert_token_endpoint_bound(app, _metadata("https://oauth2.googleapis.com/token"))

    def test_subhost_of_match_passes(self) -> None:
        app = _app("accounts.google.com")
        _assert_token_endpoint_bound(app, _metadata("https://sso.accounts.google.com/token"))

    def test_exact_match_passes(self) -> None:
        app = _app("auth.example.com")
        _assert_token_endpoint_bound(
            app,
            _metadata(
                "https://auth.example.com/token",
                issuer="https://auth.example.com",
                authz="https://auth.example.com/authorize",
            ),
        )

    def test_attacker_token_endpoint_rejected(self) -> None:
        # Spoofed issuer/authz = accounts.google.com selects the operator app,
        # but token_endpoint points at an attacker host -> refuse (would leak the
        # operator client_secret).
        app = _app("accounts.google.com", ("oauth2.googleapis.com",))
        with pytest.raises(OAuthFlowError):
            _assert_token_endpoint_bound(app, _metadata("https://attacker.example/token"))


# ── provider-app secret/method validator (#6) ───────────────────────────────


class TestProviderAppValidator:
    def test_client_secret_post_requires_secret(self) -> None:
        with pytest.raises(ValidationError):
            OAuthProviderApp(
                match="accounts.google.com",
                client_id="cid",
                token_endpoint_auth_method="client_secret_post",
            )

    def test_client_secret_basic_requires_secret(self) -> None:
        with pytest.raises(ValidationError):
            OAuthProviderApp(
                match="accounts.google.com",
                client_id="cid",
                token_endpoint_auth_method="client_secret_basic",
            )

    def test_none_method_allows_missing_secret(self) -> None:
        app = OAuthProviderApp(
            match="accounts.google.com", client_id="cid", token_endpoint_auth_method="none"
        )
        assert app.client_secret is None

    def test_confidential_method_with_secret_ok(self) -> None:
        app = OAuthProviderApp(
            match="accounts.google.com",
            client_id="cid",
            client_secret=SecretStr("s"),
            token_endpoint_auth_method="client_secret_post",
        )
        assert app.token_endpoint_auth_method == "client_secret_post"
