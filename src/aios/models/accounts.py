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
