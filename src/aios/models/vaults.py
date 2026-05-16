"""Vault and vault credential resources.

Vaults are named collections of credentials for authenticated outbound
services (MCP servers, HTTP APIs). Each credential is keyed by
``target_url`` and encrypted at rest via the CryptoBox. Secrets
(tokens, client secrets, passwords) are write-only — never returned in
API responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

AuthType = Literal["bearer_header", "oauth2_refresh", "basic"]


# ── Token endpoint auth (for OAuth refresh) ─────────────────────────────────


class TokenEndpointAuthNone(BaseModel):
    """Public OAuth client — no credentials sent on the refresh call."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["none"]


class TokenEndpointAuthBasic(BaseModel):
    """OAuth client_secret_basic — HTTP Basic header on the refresh call."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["client_secret_basic"]
    client_secret: SecretStr


class TokenEndpointAuthPost(BaseModel):
    """OAuth client_secret_post — client_secret in the form body."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["client_secret_post"]
    client_secret: SecretStr


TokenEndpointAuth = Annotated[
    TokenEndpointAuthNone | TokenEndpointAuthBasic | TokenEndpointAuthPost,
    Field(discriminator="method"),
]


# ── Vault ────────────────────────────────────────────────────────────────────


class VaultCreate(BaseModel):
    """Request body for ``POST /v1/vaults``."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VaultUpdate(BaseModel):
    """Request body for ``PUT /v1/vaults/{vault_id}``."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    metadata: dict[str, Any] | None = None


class Vault(BaseModel):
    """Read view of a vault."""

    id: str
    display_name: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


# ── Vault Credential ────────────────────────────────────────────────────────


class VaultCredentialCreate(BaseModel):
    """Request body for ``POST /v1/vaults/{vault_id}/credentials``.

    All secret fields are write-only. The ``target_url`` is immutable
    after creation. The service layer validates required fields per
    ``auth_type``.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=128)
    target_url: str = Field(min_length=1)
    auth_type: AuthType
    metadata: dict[str, Any] = Field(default_factory=dict)

    # oauth2_refresh fields
    access_token: SecretStr | None = None
    expires_at: datetime | None = None
    client_id: str | None = None
    refresh_token: SecretStr | None = None
    token_endpoint: str | None = None
    token_endpoint_auth: TokenEndpointAuth | None = None
    scope: str | None = None
    resource: str | None = None

    # bearer_header fields
    token: SecretStr | None = None

    # basic fields
    username: SecretStr | None = None
    password: SecretStr | None = None


class VaultCredentialUpdate(BaseModel):
    """Request body for ``PUT /v1/vaults/{vault_id}/credentials/{id}``.

    ``target_url`` and ``auth_type`` are immutable — not accepted here.
    Omitted secret fields are preserved (decrypt-merge-encrypt).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None

    # oauth2_refresh fields (all optional — omitted = preserve)
    access_token: SecretStr | None = None
    expires_at: datetime | None = None
    client_id: str | None = None
    refresh_token: SecretStr | None = None
    token_endpoint: str | None = None
    token_endpoint_auth: TokenEndpointAuth | None = None
    scope: str | None = None
    resource: str | None = None

    # bearer_header
    token: SecretStr | None = None

    # basic
    username: SecretStr | None = None
    password: SecretStr | None = None


class VaultCredential(BaseModel):
    """Read view of a vault credential. Secrets are never returned."""

    id: str
    vault_id: str
    display_name: str | None
    target_url: str
    auth_type: AuthType
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
