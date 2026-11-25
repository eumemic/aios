"""Per-account model-provider config: encrypted API key + proxy base URL.

Model API keys have historically been worker-global process env vars,
resolved by LiteLLM from the model string's provider prefix. A
``model_providers`` row lets an account configure its own credentials for a
given LiteLLM provider (``anthropic``, ``openai``, ``openrouter``, …),
resolved nearest-ancestor-wins up the account tree at model-call time (see
``aios.db.queries.model_providers.resolve_model_provider``) — a child
without its own row inherits its nearest configured ancestor's, falling
back to the worker's env vars when no account in the chain has one.

Resolution is **row-atomic**: the winning account's ``api_key`` and
``api_base`` always come from the SAME row, never combined field-by-field
across levels. A child's ``api_base`` paired with an ancestor's ``api_key``
is a key-exfiltration primitive (the child could redirect inference to an
endpoint it controls while an ancestor's real key rides along) — see
``provider_auth_conflict`` below, which the harness evaluates whenever an
agent's ``litellm_extra`` redirects the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from aios.models.attenuation import api_base_of

CREDENTIAL_KWARGS = frozenset(
    {
        "api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_region_name",
        "vertex_credentials",
        "vertex_project",
        "vertex_location",
        "azure_ad_token",
        "client_id",
        "client_secret",
        "tenant_id",
        "extra_headers",
    }
)
ENDPOINT_KWARGS = frozenset({"api_base", "base_url", "aws_bedrock_runtime_endpoint"})
REASONING_KWARGS = frozenset({"reasoning_effort", "thinking", "output_config"})


# The CREDENTIAL_KWARGS members that actually carry a secret. Presence of a
# non-secret family member alone (``aws_region_name``, ``extra_headers``, …)
# suppresses row injection but does NOT stop LiteLLM's env-key fallback, so it
# must never exempt the redirect guard.
SECRET_CREDENTIAL_KWARGS = frozenset(
    {"api_key", "aws_secret_access_key", "vertex_credentials", "azure_ad_token", "client_secret"}
)


def _inline_secret_present(litellm_extra: dict[str, Any]) -> bool:
    """A real (non-degenerate) inline secret: a stripped-non-empty string.

    A PRESENT-but-degenerate ``api_key`` (None, "", whitespace, non-str) is
    treated by LiteLLM as absent (env fallback), so it disqualifies outright.
    """
    if "api_key" in litellm_extra:
        key = litellm_extra["api_key"]
        return isinstance(key, str) and bool(key.strip())
    return any(
        isinstance(v, str) and bool(v.strip())
        for k, v in litellm_extra.items()
        if k in SECRET_CREDENTIAL_KWARGS
    )


def _secret_value_present(value: Any) -> bool:
    """A usable secret value: a stripped-non-empty string, or (``vertex_credentials``
    may be a parsed service-account JSON) a non-empty dict."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    return False


def credentials_hold_secret(credentials: dict[str, Any]) -> bool:
    """Whether a stored credential set carries at least one real secret.

    Only ``SECRET_CREDENTIAL_KWARGS`` count. A set made solely of non-secret
    family keys (``extra_headers``, ``aws_region_name``, ``vertex_project``, …)
    gives LiteLLM nothing to authenticate with, so it falls back to the
    worker's environment key — it must never count as credentialed.
    """
    return any(_secret_value_present(credentials.get(key)) for key in SECRET_CREDENTIAL_KWARGS)


def credential_flags(credentials: dict[str, Any]) -> tuple[bool, bool]:
    """``(api_key_set, credentials_set)`` for a decrypted credential set.

    Computed at write time from the plaintext and persisted beside the blob,
    so the read view reports what the stored set actually holds (an
    encrypted ``{}`` is non-empty ciphertext, so blob length cannot answer).
    ``credentials_set`` agrees with :attr:`ProviderAuth.has_credentials`.
    """
    return _secret_value_present(credentials.get("api_key")), credentials_hold_secret(credentials)


def validate_row_content(
    credentials: dict[str, Any] | None,
    litellm_defaults: dict[str, Any],
    *,
    require_key_with_endpoint: bool = True,
) -> None:
    forbidden = (CREDENTIAL_KWARGS | ENDPOINT_KWARGS) & litellm_defaults.keys()
    if forbidden:
        raise ValueError(
            f"litellm_defaults contains credential or endpoint keys: {sorted(forbidden)}"
        )
    if credentials is not None:
        unknown = credentials.keys() - (CREDENTIAL_KWARGS | ENDPOINT_KWARGS)
        if unknown:
            raise ValueError(f"credentials contains unsupported keys: {sorted(unknown)}")
        if (
            require_key_with_endpoint
            and api_base_of(credentials) is not None
            and not credentials.get("api_key")
        ):
            raise ValueError("credentials endpoint requires api_key")
        if (
            require_key_with_endpoint
            and CREDENTIAL_KWARGS & credentials.keys()
            and not credentials_hold_secret(credentials)
        ):
            # Non-secret family keys alone would make the row look configured
            # while LiteLLM silently authenticates with the worker env key.
            raise ValueError(
                "credentials require a non-empty secret "
                f"(one of {sorted(SECRET_CREDENTIAL_KWARGS)})"
            )


class ModelProviderCreate(BaseModel):
    """Request body for ``POST /v1/model-providers``.

    ``provider`` is a LiteLLM provider name (e.g. ``anthropic``, ``openai``,
    ``openrouter``) — lower-cased and stripped so it matches what
    ``litellm.get_llm_provider`` returns at resolve time regardless of the
    caller's casing. ``api_key`` is write-only and required in v1 (a keyless
    arm for unauthenticated self-hosted endpoints is a documented future
    extension, not yet supported).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    api_key: SecretStr | None = Field(default=None, min_length=1)
    api_base: str | None = None
    credentials: dict[str, Any] | None = None
    litellm_defaults: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> ModelProviderCreate:
        credentials = dict(self.credentials or {})
        if self.api_key is not None:
            credentials["api_key"] = self.api_key.get_secret_value()
        if self.api_base is not None:
            credentials["api_base"] = self.api_base
        validate_row_content(credentials or None, self.litellm_defaults)
        return self

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        return v.strip().lower()


class ModelProviderUpdate(BaseModel):
    """Request body for ``PUT /v1/model-providers/{id}``.

    ``api_key`` omitted → keep the existing key (rotation is opt-in via an
    explicit value; there is no way to clear it back to unset in v1 — archive
    and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
    clear (checked via ``model_fields_set``, not a sentinel default, since
    ``None`` is itself a valid target value).
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = Field(default=None, min_length=1)
    api_base: str | None = None
    credentials: dict[str, Any] | None = None
    litellm_defaults: dict[str, Any] | None = None
    version: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional optimistic-concurrency guard: the version the caller last "
            "read. When set and the row has moved on, the update is rejected "
            "with 409 and nothing is written."
        ),
    )

    @model_validator(mode="after")
    def _validate_content(self) -> ModelProviderUpdate:
        # ``credentials`` is a PARTIAL update merged onto the stored set
        # (omitted keys kept, explicit null removes), so the endpoint-needs-key
        # rule can only be judged on the merged result — the service does that.
        # Here: key-family shape only.
        validate_row_content(
            self.credentials,
            self.litellm_defaults or {},
            require_key_with_endpoint=False,
        )
        return self

    @field_validator("litellm_defaults")
    @classmethod
    def _null_defaults_clear(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        # Explicit ``null`` means "clear the defaults": the column is
        # ``jsonb NOT NULL``, so it is stored as ``{}``, never written as NULL.
        return {} if v is None else v


class ModelProvider(BaseModel):
    """Read view of a model-provider config. ``api_key`` is never returned."""

    id: str
    provider: str
    api_base: str | None = None
    api_key_set: bool
    credentials_set: bool = False
    litellm_defaults: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProviderAuth:
    """Resolved credentials for one model call, as decrypted at call time.

    ``api_key`` has ``repr=False`` so it never leaks into a log line or
    exception traceback via an accidental ``repr(auth)``. No ``depth`` field:
    the guard's only question is "is the owner the caller's own account or
    an ancestor's," which ``owner_account_id`` already answers by comparison
    against the caller's ``account_id`` — carrying depth as well would be a
    redundant boolean-in-disguise (self vs. not-self) smuggled in as an int.
    """

    api_key: str = field(repr=False)
    api_base: str | None
    owner_account_id: str
    credentials: dict[str, Any] = field(default_factory=dict, repr=False)
    litellm_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def has_credentials(self) -> bool:
        """Whether the matched row carries a real secret (vs. a defaults-only row).

        Only a non-empty ``SECRET_CREDENTIAL_KWARGS`` value counts: non-secret
        family keys alone (``extra_headers``, ``aws_region_name``, …) leave
        LiteLLM on the env-key fallback. A row without a secret never
        constrains endpoint identity and never satisfies credential
        admission: the call is credential-wise exactly as if no row matched.
        """
        return _secret_value_present(self.api_key) or credentials_hold_secret(self.credentials)


def provider_auth_conflict(
    *,
    litellm_extra: dict[str, Any] | None,
    resolved: ProviderAuth | None,
    account_id: str,
    account_is_root: bool,
) -> bool:
    """Would this call send an api_key owned above ``account_id`` to a redirected endpoint?

    ``litellm_extra`` redirects the call's ``api_base`` (or LiteLLM's
    ``base_url`` alias). If it does, the effective ``api_key`` LiteLLM will
    actually send must be traceable to something the account holds itself:

    * Inline ``litellm_extra["api_key"]`` never exempts this guard: inline
      credentials are agent metadata, not account configuration, and are not
      authoritative under non-legacy credential policies.
    * a ``model_providers`` row **owned by this account itself**
      (``resolved.owner_account_id == account_id``) — the account's own key,
      redirected by the account's own agent, is not privilege escalation.
      This exemption relies on ``resolved.api_key`` never being empty: both
      ``ModelProviderCreate`` and ``ModelProviderUpdate`` enforce
      ``min_length=1`` on ``api_key`` for exactly this reason — an
      own-row exemption over an empty key would fall back to LiteLLM's env
      resolution the same way a falsy ``litellm_extra["api_key"]`` would,
      reopening this guard via a self-created empty row instead of an
      ancestor's.

    Any other case is a conflict: a row owned by a strict ancestor
    (``resolved is not None and resolved.owner_account_id != account_id``),
    or no usable row at all on a non-root account, which means the call
    would fall back to the worker's environment key — env keys are
    root-owned, so a non-root account redirecting onto them is the same
    exfiltration shape as an ancestor row. "No usable row" deliberately
    covers BOTH a genuine no-row lookup AND a resolution failure (e.g. an
    unresolvable model string): the two are structurally indistinguishable
    to this function (``resolved`` is ``None`` either way) and MUST be
    treated identically — a bare pass on ``resolved is None`` would be the
    same bypass class as the falsy-api_key case above, just reached via an
    unresolvable ``model`` string instead of a null key.
    """
    redirect = api_base_of(litellm_extra)
    if redirect is None:
        return False
    if litellm_extra is not None and _inline_secret_present(litellm_extra):
        # Agent endpoint + credential is an independent source; the resolved row
        # is suppressed wholesale by merge_provider_config.
        return False
    if resolved is not None:
        return resolved.owner_account_id != account_id
    return not account_is_root
