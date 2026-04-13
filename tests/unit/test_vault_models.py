"""Unit tests for vault Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from aios.models.vaults import (
    VaultCreate,
    VaultCredentialCreate,
    VaultCredentialUpdate,
    VaultUpdate,
)


class TestVaultCreate:
    def test_valid_create(self) -> None:
        v = VaultCreate(display_name="my vault")
        assert v.display_name == "my vault"
        assert v.metadata == {}

    def test_with_metadata(self) -> None:
        v = VaultCreate(display_name="v", metadata={"team": "infra"})
        assert v.metadata == {"team": "infra"}

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            VaultCreate(display_name="")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VaultCreate(display_name="v", bogus="x")  # type: ignore[call-arg]


class TestVaultUpdate:
    def test_all_optional(self) -> None:
        u = VaultUpdate()
        assert u.display_name is None
        assert u.metadata is None


class TestVaultCredentialCreate:
    def test_static_bearer_valid(self) -> None:
        c = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com",
            auth_type="static_bearer",
            token=SecretStr("my-token"),
        )
        assert c.auth_type == "static_bearer"
        assert c.token is not None
        assert c.token.get_secret_value() == "my-token"

    def test_mcp_oauth_valid(self) -> None:
        c = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com",
            auth_type="mcp_oauth",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
        )
        assert c.auth_type == "mcp_oauth"
        assert c.access_token is not None

    def test_rejects_bad_auth_type(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                mcp_server_url="https://x.com",
                auth_type="unknown",  # type: ignore[arg-type]
                token=SecretStr("t"),
            )

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                mcp_server_url="",
                auth_type="static_bearer",
                token=SecretStr("t"),
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                mcp_server_url="https://x.com",
                auth_type="static_bearer",
                token=SecretStr("t"),
                bogus="x",  # type: ignore[call-arg]
            )


class TestVaultCredentialUpdate:
    def test_all_optional(self) -> None:
        u = VaultCredentialUpdate()
        assert u.display_name is None
        assert u.token is None
        assert u.access_token is None

    def test_partial_update(self) -> None:
        u = VaultCredentialUpdate(token=SecretStr("new-token"))
        assert u.token is not None
        assert u.token.get_secret_value() == "new-token"
        assert "token" in u.model_fields_set
        assert "access_token" not in u.model_fields_set
