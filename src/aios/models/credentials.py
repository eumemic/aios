"""Credential resource: encrypted API keys for upstream model providers.

The plaintext value never leaves a `Create`/`Update` request body. The `Read`
view is what every other endpoint returns; it intentionally has no `value`
field. The plaintext is only re-derived inside the harness via the vault when
calling LiteLLM.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class CredentialCreate(BaseModel):
    """Request body for `POST /v1/credentials`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    provider: str = Field(
        min_length=1,
        max_length=64,
        description="Free-form identifier for documentation/UI grouping (e.g. 'anthropic').",
    )
    value: SecretStr = Field(
        description="The plaintext API key. Never echoed back; encrypted before storage.",
    )


class CredentialUpdate(BaseModel):
    """Request body for `PATCH /v1/credentials/{id}`."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(default=None, min_length=1, max_length=64)
    value: SecretStr | None = None


class Credential(BaseModel):
    """Read view of a credential. Note: `value` is never present."""

    id: str
    name: str
    provider: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
