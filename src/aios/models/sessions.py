"""Session resource: a running agent instance + its event log handle.

A session references an agent and an environment, and tracks its current
status (see :data:`SessionStatus`) plus the workspace volume path the
sandbox uses on the host. The harness lease columns and `container_id`
are internal — they live in the DB row but are not exposed on the wire
shape.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aios.models.events import Event

SessionStatus = Literal["pending", "running", "idle", "rescheduling", "terminated"]


class SessionUsage(BaseModel):
    """Cumulative token usage across all model calls in a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


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
    vault_ids: list[str] = Field(
        default_factory=list,
        description="Vault ids to bind to this session for MCP credential resolution.",
    )

    workspace_path: str | None = Field(
        default=None,
        description=(
            "Absolute host path to use as the session workspace. "
            "If omitted, defaults to workspace_root/<session_id>. "
            "The directory must exist; aios will not create it."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables injected into the sandbox container.",
    )
    initial_message: str | None = Field(
        default=None,
        description=(
            "Convenience: when set, the server appends a user.message event "
            "with this content immediately after creating the session and "
            "enqueues a wake job. Equivalent to a follow-up POST /messages."
        ),
    )

    @field_validator("workspace_path")
    @classmethod
    def _validate_workspace_path(cls, v: str | None) -> str | None:
        if v is None:
            return None
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("workspace_path must be an absolute path")
        if not p.is_dir():
            raise ValueError(f"workspace_path directory does not exist: {v}")
        return v


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
    vault_ids: list[str] | None = None


class Session(BaseModel):
    """Read view of a session. Internal-only columns are not exposed."""

    id: str
    agent_id: str
    environment_id: str
    agent_version: int | None
    title: str | None
    metadata: dict[str, Any]
    status: SessionStatus
    stop_reason: dict[str, Any] | None
    vault_ids: list[str] = Field(default_factory=list)
    last_event_seq: int
    usage: SessionUsage = Field(default_factory=SessionUsage)
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    focal_channel: str | None = None


class SessionUserMessage(BaseModel):
    """Request body for `POST /v1/sessions/{id}/messages`."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionInterruptRequest(BaseModel):
    """Request body for `POST /v1/sessions/{id}/interrupt`."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = None


class ToolResultRequest(BaseModel):
    """Request body for ``POST /v1/sessions/{id}/tool-results``."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="The tool_call_id from the assistant's tool_calls.")
    content: str = Field(description="The result of executing the tool.")
    is_error: bool = Field(default=False, description="True if the tool execution failed.")


class WaitResponse(BaseModel):
    """Response for ``GET /v1/sessions/{id}/wait``."""

    events: list[Event]
    session_status: SessionStatus
    session_stop_reason: dict[str, Any] | None
    next_after: int


class ContextResponse(BaseModel):
    """Response for ``GET /v1/sessions/{id}/context``.

    The exact chat-completions payload the worker would send to LiteLLM
    if a step ran right now.  Dry-run: no side effects — no events
    appended, no status bump, no skill files provisioned.
    """

    session_id: str
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class ToolConfirmationRequest(BaseModel):
    """Request body for ``POST /v1/sessions/{id}/tool-confirmations``.

    Used for built-in tools with ``permission: "always_ask"``. The client
    inspects the pending tool call and either allows it (the worker will
    execute it) or denies it (the model receives an error with the deny
    message).
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="The tool_call_id to confirm or deny.")
    result: Literal["allow", "deny"]
    deny_message: str | None = Field(
        default=None,
        description="When result='deny', an optional message explaining why. Shown to the model.",
    )
