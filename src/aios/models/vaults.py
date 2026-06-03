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

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

AuthType = Literal["bearer_header", "oauth2_refresh", "basic", "custom_header"]


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


class _VaultCredentialSecrets(BaseModel):
    """Shared secret-bearing fields for ``VaultCredentialCreate`` /
    ``VaultCredentialUpdate``.

    All fields are optional on both bodies: omitted on Create means
    "not configured", omitted on Update means "preserve existing".
    The service layer validates which fields are required per
    ``auth_type``.
    """

    model_config = ConfigDict(extra="forbid")

    # oauth2_refresh fields
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

    # custom_header
    header_name: str | None = None
    header_value: SecretStr | None = None


class VaultCredentialCreate(_VaultCredentialSecrets):
    """Request body for ``POST /v1/vaults/{vault_id}/credentials``.

    All secret fields are write-only. The ``target_url`` is immutable
    after creation. The service layer validates required fields per
    ``auth_type``.
    """

    display_name: str | None = Field(default=None, max_length=128)
    target_url: str = Field(min_length=1)
    auth_type: AuthType
    metadata: dict[str, Any] = Field(default_factory=dict)


class VaultCredentialUpdate(_VaultCredentialSecrets):
    """Request body for ``PUT /v1/vaults/{vault_id}/credentials/{id}``.

    ``target_url`` and ``auth_type`` are immutable — not accepted here.
    Omitted secret fields are preserved (decrypt-merge-encrypt).
    """

    display_name: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


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


# ── Interactive OAuth "Connect" flow ────────────────────────────────────────


class OAuthStartRequest(BaseModel):
    """Begin an interactive OAuth authorization-code flow for an MCP server.

    With the token fields left blank, the server discovers the target's OAuth
    metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
    the supplied ``client_id``/``client_secret``, and returns an
    ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
    console's callback and is reused verbatim on completion.
    """

    model_config = ConfigDict(extra="forbid")

    target_url: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    display_name: str | None = Field(default=None, max_length=128)
    scope: str | None = None
    # For MCP servers that do NOT support Dynamic Client Registration: a
    # pre-registered OAuth client. Omit for DCR-capable servers.
    client_id: str | None = None
    client_secret: SecretStr | None = None
    token_endpoint_auth_method: (
        Literal["none", "client_secret_basic", "client_secret_post"] | None
    ) = None


class OAuthStartResponse(BaseModel):
    """The authorization URL to redirect the user to, plus the flow's CSRF state."""

    flow_id: str
    state: str
    authorization_url: str


class OAuthCompleteRequest(BaseModel):
    """Finish an interactive OAuth flow: exchange the returned code for tokens.

    The ``state`` correlates back to the in-progress flow (and guards CSRF);
    ``code`` is the authorization code the provider returned to the callback.
    """

    model_config = ConfigDict(extra="forbid")

    state: str = Field(min_length=1)
    code: str = Field(min_length=1)


class OAuthProviderApp(BaseModel):
    """An operator-registered OAuth client app for a provider that does NOT
    support Dynamic Client Registration (Google, Microsoft, Slack, …).

    When the interactive Connect flow discovers a server whose OAuth issuer /
    authorization host / target URL host matches ``match``, it uses this app's
    ``client_id`` / ``client_secret`` instead of registering a client — so end
    users sign in without supplying any credentials (the CMA model). Configured
    by the operator via the ``AIOS_OAUTH_PROVIDER_APPS`` JSON env var.
    """

    model_config = ConfigDict(extra="forbid")

    match: str = Field(
        min_length=1,
        description="Host (or host suffix) to match against the discovered OAuth "
        "issuer, authorization endpoint, or target URL — e.g. 'accounts.google.com'.",
    )
    client_id: str = Field(min_length=1)
    client_secret: SecretStr | None = None
    token_endpoint_auth_method: Literal["none", "client_secret_basic", "client_secret_post"] = (
        "client_secret_post"
    )
    token_endpoint_hosts: list[str] = Field(
        default_factory=list,
        description="Token-endpoint hostnames trusted for this app, beyond ``match``. "
        "Required when the provider issues tokens from a different host than it "
        "authorizes at — e.g. Google authorizes at accounts.google.com but issues "
        "tokens at oauth2.googleapis.com, so set "
        "token_endpoint_hosts=['oauth2.googleapis.com']. The operator's client_secret "
        "is sent to the discovered token endpoint only if its host is ``match`` (or a "
        "sub-host) or appears here — closing the exfiltration where attacker metadata "
        "selects this app via a spoofed issuer/authorization host but points the token "
        "endpoint elsewhere.",
    )
    scope: str | None = Field(
        default=None,
        description="Override the discovered scope (some providers, e.g. Google, "
        "require specific scopes the MCP server may not advertise).",
    )
    authorize_params: dict[str, str] = Field(
        default_factory=dict,
        description="Extra query params to add to the authorization URL. Provider "
        "quirks live here — e.g. Google needs {'access_type': 'offline', 'prompt': "
        "'consent'} to return a refresh token (standard providers issue one without).",
    )

    @model_validator(mode="after")
    def _require_secret_for_confidential_method(self) -> OAuthProviderApp:
        """A confidential client-auth method needs a secret — fail fast at config
        load rather than silently POSTing an empty secret (and storing the
        credential as a public ``none`` client) at the user's first Connect."""
        if (
            self.token_endpoint_auth_method in ("client_secret_basic", "client_secret_post")
            and not self.client_secret
        ):
            raise ValueError(
                f"token_endpoint_auth_method={self.token_endpoint_auth_method!r} "
                "requires client_secret"
            )
        return self
