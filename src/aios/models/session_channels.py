"""Session-owned connector channel state.

``session_channels`` is the MCP-inbound replacement for legacy
``channel_bindings``. A row records one channel that a session may focus with
``switch_channel``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

NotificationMode = Literal["focal_candidate", "silent"]


class SessionChannel(BaseModel):
    """Read view of one session materialized channel."""

    id: str
    session_id: str
    mcp_server_name: str
    mcp_server_url: str
    account_id: str
    path: str
    address: str
    display_name: str | None = None
    notification_mode: NotificationMode = "focal_candidate"
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime
    archived_at: datetime | None = None
