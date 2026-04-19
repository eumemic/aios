"""Agent resource: model + system prompt + tools + MCP servers.

Agents are versioned: every update creates a new immutable version. The
``agents`` table holds the latest config; the ``agent_versions`` table stores
the full history. The ``model`` field is a free-form LiteLLM model string
(e.g. ``anthropic/claude-opus-4-6``, ``ollama_chat/llama3.3``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aios.models.skills import AgentSkillRef

# Built-in tool types. Custom tools use type="custom" with extra fields.
BuiltinToolType = Literal[
    "bash",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "web_fetch",
    "web_search",
    "search_events",
    "cancel",
]

# Permission policy for built-in tools. Custom tools are always client-controlled
# and ignore this field.
PermissionPolicy = Literal["always_allow", "always_ask"]


# ── Triage gate config ────────────────────────────────────────────────────────


class TriageConfig(BaseModel):
    """Pre-inference gate that decides whether the agent should respond.

    When present, the harness runs one cheap model call before the main
    inference to classify the latest stimulus as ``respond`` or
    ``ignore``. On ``ignore`` the step emits a ``triage_decision``
    lifecycle event (carrying a ``reacting_to`` watermark) and returns
    without calling the main model. The user message itself is still
    appended to the log unconditionally by the API — the gate only
    controls the reply, not recording.

    Intended for group-chat scenarios where many messages arrive but
    few are addressed to the agent.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        min_length=1,
        description=(
            "LiteLLM model string for the triage call. Use a small, fast model "
            "(e.g. 'ollama_chat/llama3.2:1b', 'openrouter/anthropic/claude-haiku-4-5')."
        ),
    )
    system: str = Field(
        default="",
        description=(
            "System prompt for the triage classifier. Should instruct the model "
            'to emit a JSON object {"decision": "respond"|"ignore", "reason": "..."}.'
        ),
    )


# ── MCP server declaration ────────────────────────────────────────────────────


class McpServerSpec(BaseModel):
    """One entry in an agent's ``mcp_servers`` list.

    Declares a remote MCP server reachable via streamable HTTP transport.
    The ``name`` is used to cross-reference from ``mcp_toolset`` tool entries
    and to namespace discovered tools as ``mcp__<name>__<tool_name>``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["url"] = "url"
    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _reject_reserved_prefix(cls, v: str) -> str:
        from aios.models.connections import CONNECTION_SERVER_NAME_PREFIX

        if v.startswith(CONNECTION_SERVER_NAME_PREFIX):
            raise ValueError(
                f"mcp_server name prefix {CONNECTION_SERVER_NAME_PREFIX!r} is reserved"
            )
        return v


# ── MCP toolset config (permission policies for discovered tools) ──────────


class McpPermissionPolicy(BaseModel):
    """Wrapper matching Anthropic's ``{type: "always_allow"}`` shape."""

    model_config = ConfigDict(extra="forbid")

    type: PermissionPolicy


class McpToolsetConfig(BaseModel):
    """Default config for all tools discovered from an MCP server."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None


class McpToolConfig(BaseModel):
    """Per-tool override within an ``mcp_toolset`` entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None


# ── Tool declaration ──────────────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """One entry in an agent's ``tools`` list.

    For built-in tools, ``type`` is the tool name (``"bash"``, ``"read"``,
    etc.). For custom (client-executed) tools, ``type`` is ``"custom"`` and
    ``name``, ``description``, and ``input_schema`` are required. For MCP
    toolsets, ``type`` is ``"mcp_toolset"`` and ``mcp_server_name`` is
    required.

    ``enabled`` controls whether the tool is included in the schema sent to
    the model. Disabled tools are invisible to the model.

    ``permission`` controls execution policy for built-in tools:
    ``None`` or ``"always_allow"`` executes immediately (current default);
    ``"always_ask"`` idles the session with ``requires_action`` until the
    client confirms or denies.
    """

    model_config = ConfigDict(extra="forbid")

    type: BuiltinToolType | Literal["custom", "mcp_toolset"]
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    enabled: bool = True
    permission: PermissionPolicy | None = None

    # mcp_toolset fields
    mcp_server_name: str | None = None
    default_config: McpToolsetConfig | None = None
    configs: list[McpToolConfig] | None = None

    @model_validator(mode="after")
    def _check_type_fields(self) -> ToolSpec:
        if self.type == "custom":
            missing = [
                f for f in ("name", "description", "input_schema") if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(f"custom tools require: {', '.join(missing)}")
        elif self.type == "mcp_toolset":
            if self.mcp_server_name is None:
                raise ValueError("mcp_toolset requires mcp_server_name")
        return self


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
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    window_min: int = Field(default=50_000, ge=1)
    window_max: int = Field(default=150_000, ge=1)
    triage: TriageConfig | None = None


class AgentUpdate(BaseModel):
    """Request body for ``PUT /v1/agents/{id}``.

    All config fields are optional; omitted fields are preserved. The
    ``version`` field is required for optimistic concurrency — it must match
    the current version. If the update produces a change, a new version is
    created; otherwise the existing version is returned unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(description="Current version for optimistic concurrency.")
    name: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1)
    system: str | None = None
    tools: list[ToolSpec] | None = None
    skills: list[AgentSkillRef] | None = None
    mcp_servers: list[McpServerSpec] | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
    window_min: int | None = Field(default=None, ge=1)
    window_max: int | None = Field(default=None, ge=1)
    triage: TriageConfig | None = None


class Agent(BaseModel):
    """Read view of an agent (always the latest version)."""

    id: str
    version: int
    name: str
    model: str
    system: str
    tools: list[ToolSpec]
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec]
    description: str | None
    metadata: dict[str, Any]
    window_min: int
    window_max: int
    triage: TriageConfig | None = None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class AgentVersion(BaseModel):
    """Read view of a specific agent version from the version history."""

    agent_id: str
    version: int
    model: str
    system: str
    tools: list[ToolSpec]
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec]
    window_min: int
    window_max: int
    triage: TriageConfig | None = None
    created_at: datetime
