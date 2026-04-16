"""Connection resource and inbound-message DTOs.

A *connection* is a registered ``(connector, account)`` pair plus the
MCP URL the connector exposes for the agent to send replies back through.
The address scheme used by the routing layer is::

    {connector}/{account}/{path}

where ``path`` is whatever sub-segments the connector emits for inbound
messages (typically a chat or thread id).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConnectionCreate(BaseModel):
    """Request body for ``POST /v1/connections``."""

    model_config = ConfigDict(extra="forbid")

    connector: str = Field(min_length=1, max_length=64)
    account: str = Field(min_length=1, max_length=256)
    mcp_url: str = Field(min_length=1)
    vault_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConnectionUpdate(BaseModel):
    """Request body for ``PUT /v1/connections/{id}``.

    ``connector`` and ``account`` are immutable after creation.
    """

    model_config = ConfigDict(extra="forbid")

    mcp_url: str | None = Field(default=None, min_length=1)
    vault_id: str | None = None
    metadata: dict[str, Any] | None = None


class Connection(BaseModel):
    """Read view of a connection."""

    id: str
    connector: str
    account: str
    mcp_url: str
    vault_id: str
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
