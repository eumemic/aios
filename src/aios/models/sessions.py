"""Session resource: a running agent instance + its event log handle.

A session references an agent and an environment, and tracks its current
status (`running`, `idle`, `terminated`) plus the workspace volume path the
sandbox uses on the host. The harness lease columns and `container_id` are
internal — they live in the DB row but are not exposed on the wire shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SessionStatus = Literal["running", "idle", "terminated"]


class SessionCreate(BaseModel):
    """Request body for `POST /v1/sessions`."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    environment_id: str
    agent_version: int | None = Field(
        default=None,
        description=(
            "Pin to a specific agent version. Omit or pass null for 'latest' "
            "(auto-updating — the session uses whatever version is current)."
        ),
    )
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_message: str | None = Field(
        default=None,
        description=(
            "Convenience: when set, the server appends a user.message event "
            "with this content immediately after creating the session and "
            "enqueues a wake job. Equivalent to a follow-up POST /messages."
        ),
    )


class SessionUpdate(BaseModel):
    """Request body for ``PUT /v1/sessions/{id}``.

    All fields are optional; omitted fields are preserved. Changing
    ``agent_id`` resets ``agent_version`` to null (latest) unless
    ``agent_version`` is also provided.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    agent_version: int | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None


class Session(BaseModel):
    """Read view of a session. Internal-only columns are not exposed."""

    id: str
    agent_id: str
    environment_id: str
    agent_version: int | None
    title: str | None
    metadata: dict[str, Any]
    status: SessionStatus
    stop_reason: str | None
    last_event_seq: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class SessionUserMessage(BaseModel):
    """Request body for `POST /v1/sessions/{id}/messages`."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionInterruptRequest(BaseModel):
    """Request body for `POST /v1/sessions/{id}/interrupt`."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = None
