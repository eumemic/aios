"""Operator-managed inbound approval ledger models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class InboundGrantAction(BaseModel):
    """Identify the canonical chat whose grant is being changed."""

    model_config = ConfigDict(extra="forbid")
    chat_id: str


class InboundGrant(BaseModel):
    """Audited inbound approval state."""

    id: str
    account_id: str
    connection_id: str
    chat_id: str
    status: Literal["pending", "active", "revoked"]
    approved_by: str | None = None
    approved_at: datetime | None = None
    approved_via_channel: str | None = None
    created_at: datetime
    updated_at: datetime
