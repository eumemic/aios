"""Account and account-key resource models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Account(BaseModel):
    id: str
    parent_account_id: str | None
    can_mint_children: bool
    display_name: str
    metadata: dict[str, Any]
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

    Partial update: omitted fields are preserved. Both fields are
    optional; at least one must be non-null. Submitting both as null
    is a no-op that returns the account row unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    can_mint_children: bool | None = None
