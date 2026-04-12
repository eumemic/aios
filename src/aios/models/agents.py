"""Agent resource: model + system prompt + tools + credential reference.

Phase 1 stores agents as a single mutable row (no version table yet). Phase 4
introduces immutable agent versions; the wire-level Read shape will gain a
`version` field at that point. The `model` field is a free-form LiteLLM
model string (e.g. ``anthropic/claude-opus-4-6``, ``ollama_chat/llama3.3``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The set of built-in tool types aios v1 ships. Each entry on an agent's
# `tools` list is an object with at minimum a `type` field; future phases can
# extend the entry with per-tool configuration (e.g. timeouts, denylists).
ToolType = Literal["bash", "read", "write", "edit", "cancel"]


class ToolSpec(BaseModel):
    """One entry in an agent's `tools` list.

    Currently just a `type` discriminator. Other fields are reserved for
    future per-tool config and will be added without breaking the schema.
    """

    model_config = ConfigDict(extra="forbid")

    type: ToolType


class AgentCreate(BaseModel):
    """Request body for `POST /v1/agents`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model: str = Field(
        min_length=1,
        description="LiteLLM model string, e.g. 'anthropic/claude-opus-4-6'.",
    )
    system: str = Field(default="", description="System prompt; empty by default.")
    tools: list[ToolSpec] = Field(default_factory=list)
    credential_id: str | None = Field(
        default=None,
        description="Optional credential id used when calling the model.",
    )
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    window_min: int = Field(default=50_000, ge=1)
    window_max: int = Field(default=150_000, ge=1)


class AgentUpdate(BaseModel):
    """Request body for `PATCH /v1/agents/{id}`.

    All fields are optional; omitted fields are preserved. In Phase 4 this
    will allocate a new immutable agent version; in Phase 1 it just mutates
    the row in place.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1)
    system: str | None = None
    tools: list[ToolSpec] | None = None
    credential_id: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
    window_min: int | None = Field(default=None, ge=1)
    window_max: int | None = Field(default=None, ge=1)


class Agent(BaseModel):
    """Read view of an agent."""

    id: str
    name: str
    model: str
    system: str
    tools: list[ToolSpec]
    credential_id: str | None
    description: str | None
    metadata: dict[str, Any]
    window_min: int
    window_max: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
