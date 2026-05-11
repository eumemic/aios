"""Per-connection scoped bearer tokens for connector containers (#301).

Each connector container (Telegram, Signal, Echo, …) authenticates as
its connection via a bearer token issued from this resource.  The token
resolves to a single ``connection_id``; routes that take a
``ConnectorAuthDep`` use that id to scope the request — inbound is
appended to that connection's session, the calls stream emits only that
connection's pending tool calls.

Plaintext is returned ONCE on issue.  The DB stores only a SHA-256 hash;
losing the plaintext means rotating the token.

Revocation is soft (``revoked_at``) so audit trails survive.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConnectorToken(BaseModel):
    """Read view of a connector token.  Never carries plaintext."""

    id: str
    connection_id: str
    label: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ConnectorTokenIssue(BaseModel):
    """Request body for ``POST /v1/connector-tokens``."""

    model_config = ConfigDict(extra="forbid")

    connection_id: str
    label: str | None = Field(default=None, max_length=128)


class ConnectorTokenIssued(BaseModel):
    """Response body for ``POST /v1/connector-tokens``.

    Includes the plaintext token — this is the ONLY time it's surfaced.
    Subsequent ``GET`` returns the read view without plaintext.
    """

    id: str
    connection_id: str
    label: str | None
    plaintext: str = Field(
        description="The bearer token value.  Save this — it cannot be recovered.",
    )
    created_at: datetime
