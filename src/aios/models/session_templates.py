"""Session template resource — frozen recipe for per_chat session spawn.

A *session template* captures the agent + environment + bound vaults +
attached memory stores that a per_chat connection should use when
spawning a new session for an unseen chat partner.  ``agent_version``
is captured at spawn time from the live template; later template edits
do not retroactively migrate already-spawned sessions.

Templates can be soft-deleted (``archived_at``) even when referenced —
existing per_chat connections keep working with their already-spawned
sessions; new chat sessions for those connections start failing the
inbound-handler lookup until the connection is reconfigured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionTemplateCreate(BaseModel):
    """Request body for ``POST /v1/session-templates``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    agent_id: str
    environment_id: str
    agent_version: int | None = Field(
        default=None,
        description=(
            "Pin to a specific agent version. Omit or pass null for 'latest' "
            "— the spawn captures whatever version is current at spawn time."
        ),
    )
    vault_ids: list[str] = Field(default_factory=list)
    memory_store_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    archive_when_idle: bool = Field(
        default=False,
        description=(
            "Copied down to every session this template spawns: when true, each "
            "spawned session self-archives the first time it goes idle."
        ),
    )


class SessionTemplateUpdate(BaseModel):
    """Request body for ``PUT /v1/session-templates/{id}``.

    Updates apply to future spawns only — already-spawned sessions are
    not retroactively migrated (see module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    agent_id: str | None = None
    agent_version: int | None = None
    environment_id: str | None = None
    vault_ids: list[str] | None = None
    memory_store_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    archive_when_idle: bool | None = None


class SessionTemplate(BaseModel):
    """Read view of a session template."""

    id: str
    name: str
    agent_id: str
    agent_version: int | None
    environment_id: str
    vault_ids: list[str]
    memory_store_ids: list[str]
    metadata: dict[str, Any]
    archive_when_idle: bool = False
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
