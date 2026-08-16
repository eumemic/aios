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

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from aios.models.attenuation import api_base_of


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
    api_key: SecretStr = Field(min_length=1)
    api_base: str | None = None

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


class ModelProvider(BaseModel):
    """Read view of a model-provider config. ``api_key`` is never returned."""

    id: str
    provider: str
    api_base: str | None = None
    api_key_set: bool
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
    if resolved is not None:
        return resolved.owner_account_id != account_id
    return not account_is_root
