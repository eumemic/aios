"""Account and account-key resource models."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AccountConfig(BaseModel):
    """Per-account configuration bag.

    An unset item inherits from the parent account; see
    ``queries.resolve_effective_timezone``. Update semantics (per-item merge)
    are documented on ``UpdateAccountRequest``.
    """

    model_config = ConfigDict(extra="forbid")

    timezone: str | None = Field(
        default=None,
        description=(
            "IANA timezone name (e.g. 'America/Los_Angeles') used to render the "
            "per-message received-at timestamp for this account's agents. Unset "
            "inherits the parent account's timezone; the root falls back to UTC."
        ),
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str | None) -> str | None:
        # Fail hard at config-set time so the render path (which runs on every
        # wake) never has to defend against an unknown zone.
        if v is not None:
            try:
                ZoneInfo(v)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"unknown IANA timezone: {v!r}") from exc
        return v


class Account(BaseModel):
    id: str
    parent_account_id: str | None
    can_mint_children: bool
    display_name: str
    metadata: dict[str, Any]
    config: AccountConfig = Field(default_factory=AccountConfig)
    created_at: datetime
    archived_at: datetime | None = None


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=128)


class BootstrapResponse(BaseModel):
    account_id: str
    key_id: str
    # Returned exactly once at mint; not recoverable after this response.
    plaintext_key: str


class MintAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=128)
    # Defaults to False — child accounts can't mint grandchildren unless the
    # parent explicitly delegates. Two-level trees are the common case.
    can_mint_children: bool = False


class MintAccountResponse(BaseModel):
    account_id: str
    key_id: str
    # The first key on a freshly-minted account. Returned exactly once.
    plaintext_key: str


class MintKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=64)


class MintKeyResponse(BaseModel):
    key_id: str
    plaintext_key: str


class AccountUsage(BaseModel):
    """Per-account resource counts as returned by ``GET /v1/accounts/{id}/usage``."""

    agents: int
    environments: int
    sessions: int
    vaults: int
    memory_stores: int
    skills: int
    session_templates: int
    connections: int


class AccountKeySummary(BaseModel):
    """Key metadata as returned by the management API.

    Intentionally omits the bytes ``hash`` column — operators have no use
    for the on-disk hash, and surfacing it widens the audit footprint.
    """

    key_id: str
    label: str
    created_at: datetime
    revoked_at: datetime | None = None


class UpdateAccountRequest(BaseModel):
    """Body for ``PATCH /v1/accounts/{id}``.

    Partial update: omitted fields are preserved. All fields are optional;
    submitting none is a no-op that returns the account row unchanged.
    ``config`` is *merged* into the stored config (only the keys present in
    the submitted object are written), so setting one config item never
    disturbs the others.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    can_mint_children: bool | None = None
    config: AccountConfig | None = None
