"""Channel binding resource.

An explicit, immutable ``address → session_id`` mapping used by the
channel resolver as the fast path before falling back to routing rules.
There is no PUT — to re-route an address, archive the binding and create
a new one (or rely on a routing rule).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChannelBindingCreate(BaseModel):
    """Request body for ``POST /v1/channel-bindings``."""

    model_config = ConfigDict(extra="forbid")

    address: str = Field(min_length=1)
    session_id: str


class ChannelBinding(BaseModel):
    """Read view of a channel binding."""

    id: str
    address: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
