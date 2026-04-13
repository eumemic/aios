"""Vault and vault credential resources.

Vaults are named collections of credentials for MCP server authentication.
Each credential is keyed by ``mcp_server_url`` and encrypted at rest via the
CryptoBox. Secrets (tokens, client secrets) are write-only — never returned
in API responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

AuthType = Literal["mcp_oauth", "static_bearer"]


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

    All secret fields are write-only. The ``mcp_server_url`` is immutable
    after creation. The service layer validates required fields per
    ``auth_type``.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=128)
    mcp_server_url: str = Field(min_length=1)
    auth_type: AuthType
    metadata: dict[str, Any] = Field(default_factory=dict)

    # mcp_oauth fields
    access_token: SecretStr | None = None
    expires_at: datetime | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    token_endpoint: str | None = None
    token_endpoint_auth: str | None = None
    scope: str | None = None
    resource: str | None = None

    # static_bearer fields
    token: SecretStr | None = None


class VaultCredentialUpdate(BaseModel):
    """Request body for ``PUT /v1/vaults/{vault_id}/credentials/{id}``.

    ``mcp_server_url`` and ``auth_type`` are immutable — not accepted here.
    Omitted secret fields are preserved (decrypt-merge-encrypt).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None

    # mcp_oauth fields (all optional — omitted = preserve)
    access_token: SecretStr | None = None
    expires_at: datetime | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    token_endpoint: str | None = None
    token_endpoint_auth: str | None = None
    scope: str | None = None
    resource: str | None = None

    # static_bearer
    token: SecretStr | None = None


class VaultCredential(BaseModel):
    """Read view of a vault credential. Secrets are never returned."""

    id: str
    vault_id: str
    display_name: str | None
    mcp_server_url: str
    auth_type: AuthType
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
