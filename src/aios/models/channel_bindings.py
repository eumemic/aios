"""Channel binding resource.

An explicit, immutable ``address → session_id`` mapping used by the
channel resolver as the fast path before falling back to routing rules.
There is no PUT — to re-route an address, archive the binding and create
a new one (or rely on a routing rule).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NotificationMode = Literal["focal_candidate", "silent"]


class ChannelBindingCreate(BaseModel):
    """Request body for ``POST /v1/channel-bindings``.

    ``address`` is the full display form ``{connector}/{account}/{path}``.
    The service layer parses it, resolves the owning connection, and
    stores the binding as ``(connection_id, path)`` internally.
    """

    model_config = ConfigDict(extra="forbid")

    address: str = Field(min_length=1)
    session_id: str


class ChannelBinding(BaseModel):
    """Read view of a channel binding.

    ``address`` is computed on read (``{connector}/{account}/{path}``) —
    storage is normalized as ``(connection_id, path)`` so the display form
    stays a single source of truth: the owning connection.
    """

    id: str
    connection_id: str
    path: str
    address: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    notification_mode: NotificationMode = "focal_candidate"
