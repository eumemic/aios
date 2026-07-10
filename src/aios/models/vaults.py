"""Vault and vault credential resources.

Vaults are named collections of credentials for authenticated outbound
services (MCP servers, HTTP APIs). Most credentials are keyed by an
immutable ``target_url`` and consumed worker-side as outbound auth headers.
The ``environment_variable`` kind is different: it has no ``target_url`` and
is keyed by ``secret_name`` (the env var materialized into the sandbox),
carrying an ``allowed_hosts`` egress scope instead. Secrets are encrypted at
rest via the CryptoBox and are write-only — never returned in API responses.
"""

from __future__ import annotations

from aios.actors import Actor

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from aios.models.environments import HOSTNAME_RE
from aios.sandbox.env_keys import (
    RESERVED_SANDBOX_ENV_KEYS as _RESERVED_SANDBOX_ENV_KEYS,
)

AuthType = Literal[
    "bearer_header",
    "oauth2_refresh",
    "basic",
    "custom_header",
    "environment_variable",
]

# Env var names the harness injects into every sandbox. An
# ``environment_variable`` credential may not claim one as its ``secret_name``:
# a collision either hijacks a load-bearing sandbox variable (e.g. ``PATH``
# repointed → unqualified-binary takeover) or is silently shadowed by the
# harness's own merge order — both defects, and ``secret_name`` is immutable
# post-create.
#
# Derived (not hand-mirrored) from the producers' own declared key constants in
# the dependency-free ``aios.sandbox.env_keys`` module — the single source of
# truth that ``sandbox.setup`` / ``sandbox.egress_ca`` / ``sandbox.spec`` /
# ``workflows.run_sandbox`` all build their env off. That module imports nothing
# heavy, so pulling it in here doesn't cycle via ``aios.config`` (the reason
# ``sandbox.setup`` couldn't be imported) or drag cryptography's x509 machinery
# into every models import (the reason ``sandbox.egress_ca`` couldn't). Adding a
# new injected key to a producer flows into this blocklist with no second edit,
# so the set can no longer drift — the invariant is held by construction.
RESERVED_SANDBOX_ENV_KEYS = _RESERVED_SANDBOX_ENV_KEYS

# A path-prefix segment: RFC 3986 ``pchar`` minus percent-encoding. ``%`` is
# excluded so stored entries stay canonical (the swap proxy owns request-side
# percent-decoding); ``/`` is the segment separator, handled by the split.
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._~!$&'()*+,;=:@-]+$")

# POSIX portable env var name: a leading letter/underscore, then word chars.
_SECRET_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A final hostname label that is a pure numeric token in any base — decimal
# ("10", "2130706433"), hex ("0x7f000001", "0xA"), or octal ("0177"). The last
# octet of every IPv4-literal spelling is such a token, so rejecting these
# labels rejects IP literals without a libc parse. A must-contain-a-letter
# test would be both too weak (hex octets contain letters) and too strict
# (rejects legitimate digit-hyphen labels), which is why this is a token match.
_NUMERIC_LABEL_RE = re.compile(r"0[xX][0-9A-Fa-f]+|[0-9]+")

# Upper bound on one ``allowed_hosts`` entry (host[/path]). The host part is
# already capped at 253; this bounds the optional path prefix.
_MAX_ALLOWED_HOST_ENTRY_LEN = 512


def _validate_credential_host(host: str) -> str:
    """Validate the host part of one ``allowed_hosts`` entry.

    Bare hostname only (same grammar as environment ``allowed_hosts``), and
    DNS-names-only: the swap proxy (#876) matches on the request ``Host``/SNI
    name, so an IP-literal entry could never match as intended. Rejecting a
    final label that is a pure numeric token (decimal, hex, or octal) rejects
    every IP-literal spelling — dotted-quad, integer, and hex/octal — without
    a libc parse. Resolve-time SSRF — a NAME that *resolves* to an internal
    address, incl. DNS rebinding — is the egress boundary's job (#876), not
    create-time's. Returns the host unchanged (stored as-given, like
    environment ``allowed_hosts``; downstream comparisons case-fold both
    sides).
    """
    # HOSTNAME_RE rejects the empty string and anything with a port/slash/
    # wildcard; the explicit length bound is the only thing the regex lacks.
    # fullmatch, not match: the trailing ``$`` would otherwise admit a single
    # trailing newline ("host\n") and store an unmatchable control char.
    if len(host) > 253 or not HOSTNAME_RE.fullmatch(host):
        raise ValueError(
            f"invalid host {host!r}: a bare hostname is required "
            "(no scheme, port, path, wildcard, or IPv6)"
        )
    if _NUMERIC_LABEL_RE.fullmatch(host.rsplit(".", 1)[-1]):
        raise ValueError(f"invalid host {host!r}: a DNS hostname is required, not an IP literal")
    return host


def parse_allowed_host_entry(entry: str) -> tuple[str, str | None]:
    """Split one ``allowed_hosts`` entry into ``(host, path_prefix)``.

    Grammar: ``host`` or ``host/<path-prefix>``. A bare host (and the
    explicit-whole-host spelling ``host/``) returns ``(host, None)``;
    otherwise the prefix is returned in canonical leading-slash form
    (``/repos/eumemic``). One trailing slash is dropped, so ``host/`` ≡
    ``host`` and ``host/foo/`` ≡ ``host/foo`` — exactly one parse per
    semantics.

    This is the single grammar authority: the egress swap proxy (#876) and
    the cred-⊆-env check (#879) import it rather than re-deriving the split,
    so the stored entry and the runtime matcher can never disagree.

    Raises ``ValueError`` on a malformed entry.
    """
    if not entry or len(entry) > _MAX_ALLOWED_HOST_ENTRY_LEN:
        raise ValueError(f"invalid allowed_hosts entry {entry!r}: empty or too long")
    host_part, sep, rest = entry.partition("/")
    host = _validate_credential_host(host_part)
    if not sep:
        return host, None
    if rest.endswith("/"):
        rest = rest[:-1]
    if not rest:
        return host, None
    segments = rest.split("/")
    for seg in segments:
        if not seg:
            raise ValueError(f"invalid allowed_hosts entry {entry!r}: empty path segment")
        if seg in (".", ".."):
            raise ValueError(f"invalid allowed_hosts entry {entry!r}: '.'/'..' path segment")
        # The pchar charset excludes ``%`` (and every shell/URL metachar), so
        # percent-encoding and illegal characters are both rejected here.
        # fullmatch, not match: ``$`` would otherwise admit a trailing newline.
        if not _PATH_SEGMENT_RE.fullmatch(seg):
            raise ValueError(f"invalid allowed_hosts entry {entry!r}: illegal path character")
    return host, "/" + "/".join(segments)


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
    created_by: Actor | None = None
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

    # environment_variable
    secret_value: SecretStr | None = None


class VaultCredentialCreate(_VaultCredentialSecrets):
    """Request body for ``POST /v1/vaults/{vault_id}/credentials``.

    All secret fields are write-only. The structural fields — ``target_url``,
    ``secret_name``, ``allowed_hosts``, and ``auth_type`` — are immutable after
    creation; only the secret (and ``display_name``/``metadata``) can be
    rotated via PUT, so changing a credential's egress scope means archiving
    and recreating it. The service layer validates required secret fields per
    ``auth_type``; this model validates the structural shape (which kind
    carries ``target_url`` vs ``secret_name``/``allowed_hosts``).
    """

    display_name: str | None = Field(default=None, max_length=128)
    auth_type: AuthType
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Header credentials (everything except ``environment_variable``) carry a
    # ``target_url``; ``environment_variable`` carries ``secret_name`` +
    # ``allowed_hosts`` instead. The ``_validate_shape`` validator enforces
    # the xor.
    target_url: str | None = Field(default=None, min_length=1)
    secret_name: str | None = Field(default=None, max_length=128)
    allowed_hosts: list[str] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> VaultCredentialCreate:
        if self.auth_type == "environment_variable":
            if self.target_url is not None:
                raise ValueError("environment_variable credentials must not set target_url")
            if not self.secret_name:
                raise ValueError("environment_variable credentials require secret_name")
            if not _SECRET_NAME_RE.fullmatch(self.secret_name):
                raise ValueError(
                    f"invalid secret_name {self.secret_name!r}: must be a POSIX env var name "
                    "([A-Za-z_][A-Za-z0-9_]*)"
                )
            if self.secret_name in RESERVED_SANDBOX_ENV_KEYS:
                raise ValueError(
                    f"secret_name {self.secret_name!r} is reserved by the sandbox runtime"
                )
            if not self.allowed_hosts:
                raise ValueError(
                    "environment_variable credentials require a non-empty allowed_hosts"
                )
            # Canonicalize every entry so there is one stored spelling per
            # semantics (``host/`` → ``host``, ``host/foo/`` → ``host/foo``),
            # then drop cross-entry duplicates (the list is a set of egress
            # scopes; ``dict.fromkeys`` preserves first-seen order).
            canonical: list[str] = []
            for entry in self.allowed_hosts:
                host, prefix = parse_allowed_host_entry(entry)
                canonical.append(host if prefix is None else host + prefix)
            self.allowed_hosts = list(dict.fromkeys(canonical))
        else:
            if not self.target_url:
                raise ValueError(f"{self.auth_type} credentials require target_url")
            if self.secret_name is not None or self.allowed_hosts is not None:
                raise ValueError(
                    f"{self.auth_type} credentials must not set secret_name/allowed_hosts"
                )
        return self


class VaultCredentialUpdate(_VaultCredentialSecrets):
    """Request body for ``PUT /v1/vaults/{vault_id}/credentials/{id}``.

    ``target_url``, ``secret_name``, ``allowed_hosts``, and ``auth_type`` are
    immutable — not accepted here. Omitted secret fields are preserved
    (decrypt-merge-encrypt).
    """

    display_name: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class VaultCredential(BaseModel):
    """Read view of a vault credential. Secrets are never returned.

    ``target_url`` is null for ``environment_variable`` credentials;
    ``secret_name``/``allowed_hosts`` are null for every other kind.
    """

    id: str
    vault_id: str
    display_name: str | None
    target_url: str | None
    auth_type: AuthType
    secret_name: str | None = None
    allowed_hosts: list[str] | None = None
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
