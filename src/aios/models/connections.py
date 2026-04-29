"""Connection resource and inbound-message DTOs.

A *connection* is a registered ``(connector, account)`` pair used by the
inbound channel router. The address scheme used by the routing layer is::

    {connector}/{account}/{path}

where ``path`` is whatever sub-segments the connector emits for inbound
messages (typically a chat or thread id).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Connection ids use this prefix. The stable id is also used as the
# per-connection instruction alias key when connector MCP instructions are
# rendered into a session prompt.
CONNECTION_SERVER_NAME_PREFIX = "conn_"


class ConnectionCreate(BaseModel):
    """Request body for ``POST /v1/connections``.

    ``connector`` and ``account`` may not contain ``/`` — they form the
    leading two segments of channel addresses (``{connector}/{account}/{path}``)
    and a ``/`` would create ambiguous segment boundaries that confuse
    routing-rule prefix matching.
    """

    model_config = ConfigDict(extra="forbid")

    connector: str = Field(min_length=1, max_length=64)
    account: str = Field(min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("connector", "account")
    @classmethod
    def _no_slash(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("must not contain '/'")
        return v


class ConnectionUpdate(BaseModel):
    """Request body for ``PUT /v1/connections/{id}``.

    ``connector`` and ``account`` are immutable after creation.
    """

    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] | None = None


class Connection(BaseModel):
    """Read view of a connection."""

    id: str
    connector: str
    account: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class InboundMessage(BaseModel):
    """Request body for ``POST /v1/connections/{id}/messages``.

    ``path`` carries the chat id (and any connector-defined sub-segments)
    that, combined with the connection's ``connector`` and ``account``,
    forms the channel address used for routing.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboundMessageResponse(BaseModel):
    """Response from ``POST /v1/connections/{id}/messages``."""

    session_id: str
    event_id: str
    created_session: bool
