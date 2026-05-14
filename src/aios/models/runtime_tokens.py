"""Per-connector-type bearer tokens for runtime containers (#328 PR 5).

A runtime token authenticates one connector container that hosts N
connections of a single ``connector`` type — Telegram, Signal, Echo, …
The token resolves to ``(token_id, connector)``; the container then
discovers its active connections via SSE (``GET
/v1/connectors/connections``) and operates on each by ``connection_id``
in the body / query string of the runtime-scoped routes.

This is the successor to the per-connection :class:`ConnectorToken`
(#301): one bearer per container, not one bearer per connection.

Plaintext is returned ONCE on issue.  The DB stores only a SHA-256 hash;
losing the plaintext means rotating the token.  Revocation is soft
(``revoked_at``) so audit trails survive.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RuntimeToken(BaseModel):
    """Read view of a runtime token.  Never carries plaintext."""

    id: str
    connector: str
    label: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class RuntimeTokenIssue(BaseModel):
    """Request body for ``POST /v1/runtime-tokens``."""

    model_config = ConfigDict(extra="forbid")

    connector: str
    label: str | None = Field(default=None, max_length=128)


class RuntimeTokenIssued(BaseModel):
    """Response body for ``POST /v1/runtime-tokens``.

    Includes the plaintext token — this is the ONLY time it's surfaced.
    Subsequent ``GET`` returns the read view without plaintext.
    """

    id: str
    connector: str
    label: str | None
    plaintext: str = Field(
        description="The bearer token value.  Save this — it cannot be recovered.",
    )
    created_at: datetime
