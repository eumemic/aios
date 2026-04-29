"""Connection resource and inbound-message DTOs.

A *connection* is a registered ``(connector, account)`` pair used by the
inbound channel router. The address scheme used by the routing layer is::

    {connector}/{account}/{path}

where ``path`` is whatever sub-segments the connector emits for inbound
messages (typically a chat or thread id).

``mcp_url`` / ``vault_id`` are optional compatibility fields for older
connector setups that projected MCP tools from connections. New
channel-aware MCP integrations should leave them unset, declare normal
agent ``mcp_servers``, and use session vaults for credentials.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Legacy prefix for MCP server names projected from connection rows. New
# channel-aware MCP integrations should use normal agent ``mcp_servers`` plus
# ``mcp_toolset.channel_context`` instead of relying on this namespace.
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
    mcp_url: str | None = Field(default=None, min_length=1)
    vault_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("connector", "account")
    @classmethod
    def _no_slash(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("must not contain '/'")
        return v

    @model_validator(mode="after")
    def _legacy_mcp_pair(self) -> ConnectionCreate:
        if (self.mcp_url is None) != (self.vault_id is None):
            raise ValueError("mcp_url and vault_id must be provided together")
        return self


class ConnectionUpdate(BaseModel):
    """Request body for ``PUT /v1/connections/{id}``.

    ``connector`` and ``account`` are immutable after creation. Explicit
    ``null`` for a legacy MCP field clears it; omitting the field preserves it.
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
    mcp_url: str | None = None
    vault_id: str | None = None
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
