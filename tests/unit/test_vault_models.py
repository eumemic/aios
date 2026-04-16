"""Unit tests for vault Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

from aios.models.vaults import (
    TokenEndpointAuth,
    TokenEndpointAuthBasic,
    TokenEndpointAuthNone,
    TokenEndpointAuthPost,
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


class TestTokenEndpointAuth:
    def test_none_variant_no_secret(self) -> None:
        v = TokenEndpointAuthNone(method="none")
        assert v.method == "none"

    def test_none_variant_rejects_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthNone(method="none", client_secret=SecretStr("x"))  # type: ignore[call-arg]

    def test_basic_variant_carries_secret(self) -> None:
        v = TokenEndpointAuthBasic(
            method="client_secret_basic",
            client_secret=SecretStr("shh"),
        )
        assert v.method == "client_secret_basic"
        assert v.client_secret.get_secret_value() == "shh"

    def test_post_variant_carries_secret(self) -> None:
        v = TokenEndpointAuthPost(
            method="client_secret_post",
            client_secret=SecretStr("shh"),
        )
        assert v.method == "client_secret_post"
        assert v.client_secret.get_secret_value() == "shh"

    def test_basic_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthBasic(method="client_secret_basic")  # type: ignore[call-arg]

    def test_post_requires_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            TokenEndpointAuthPost(method="client_secret_post")  # type: ignore[call-arg]

    def test_discriminator_dispatches_none(self) -> None:
        adapter = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "none"})
        assert isinstance(v, TokenEndpointAuthNone)

    def test_discriminator_dispatches_basic(self) -> None:
        adapter = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "client_secret_basic", "client_secret": "shh"})
        assert isinstance(v, TokenEndpointAuthBasic)
        assert v.client_secret.get_secret_value() == "shh"

    def test_discriminator_dispatches_post(self) -> None:
        adapter = TypeAdapter(TokenEndpointAuth)
        v = adapter.validate_python({"method": "client_secret_post", "client_secret": "shh"})
        assert isinstance(v, TokenEndpointAuthPost)

    def test_discriminator_rejects_unknown_method(self) -> None:
        adapter = TypeAdapter(TokenEndpointAuth)
        with pytest.raises(ValidationError):
            adapter.validate_python({"method": "bogus"})

    def test_secret_masked_in_repr(self) -> None:
        v = TokenEndpointAuthPost(
            method="client_secret_post",
            client_secret=SecretStr("super-secret"),
        )
        assert "super-secret" not in repr(v)


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

    def test_mcp_oauth_with_typed_token_endpoint_auth_basic(self) -> None:
        c = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com",
            auth_type="mcp_oauth",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthBasic(
                method="client_secret_basic",
                client_secret=SecretStr("shh"),
            ),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthBasic)
        assert c.token_endpoint_auth.client_secret.get_secret_value() == "shh"

    def test_mcp_oauth_with_typed_token_endpoint_auth_post(self) -> None:
        c = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com",
            auth_type="mcp_oauth",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("shh"),
            ),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthPost)

    def test_mcp_oauth_with_typed_token_endpoint_auth_none(self) -> None:
        c = VaultCredentialCreate(
            mcp_server_url="https://mcp.example.com",
            auth_type="mcp_oauth",
            access_token=SecretStr("access-tok"),
            client_id="client-123",
            token_endpoint="https://issuer.example/token",
            token_endpoint_auth=TokenEndpointAuthNone(method="none"),
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthNone)

    def test_token_endpoint_auth_accepts_dict_form(self) -> None:
        c = VaultCredentialCreate.model_validate(
            {
                "mcp_server_url": "https://mcp.example.com",
                "auth_type": "mcp_oauth",
                "access_token": "tok",
                "client_id": "cid",
                "token_endpoint": "https://issuer.example/token",
                "token_endpoint_auth": {
                    "method": "client_secret_basic",
                    "client_secret": "shh",
                },
            }
        )
        assert isinstance(c.token_endpoint_auth, TokenEndpointAuthBasic)

    def test_rejects_flat_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialCreate(
                mcp_server_url="https://mcp.example.com",
                auth_type="mcp_oauth",
                access_token=SecretStr("access-tok"),
                client_id="client-123",
                client_secret=SecretStr("flat"),  # type: ignore[call-arg]
            )

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

    def test_partial_update_token_endpoint_auth(self) -> None:
        u = VaultCredentialUpdate(
            token_endpoint_auth=TokenEndpointAuthPost(
                method="client_secret_post",
                client_secret=SecretStr("rotated"),
            ),
        )
        assert "token_endpoint_auth" in u.model_fields_set
        assert isinstance(u.token_endpoint_auth, TokenEndpointAuthPost)

    def test_rejects_flat_client_secret(self) -> None:
        with pytest.raises(ValidationError):
            VaultCredentialUpdate(client_secret=SecretStr("flat"))  # type: ignore[call-arg]
