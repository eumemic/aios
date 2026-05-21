"""Agent resource: model + system prompt + tools + MCP servers.

Agents are versioned: every update creates a new immutable version. The
``agents`` table holds the latest config; the ``agent_versions`` table stores
the full history. The ``model`` field is a free-form LiteLLM model string
(e.g. ``anthropic/claude-opus-4-6``, ``ollama_chat/llama3.3``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    "schedule_wake",
    "wake_session",
    "http_request",
]

# Permission policy for built-in tools. Custom tools are always client-controlled
# and ignore this field.
PermissionPolicy = Literal["always_allow", "always_ask"]


# ── MCP server declaration ────────────────────────────────────────────────────


class McpServerSpec(BaseModel):
    """One entry in an agent's ``mcp_servers`` list.

    Declares a remote MCP server reachable via streamable HTTP transport.
    The ``name`` is used to cross-reference from ``mcp_toolset`` tool entries
    and to namespace discovered tools as ``mcp__<name>__<tool_name>``.

    ``include_instructions`` controls whether the server's
    ``InitializeResult.instructions`` (per MCP spec) is rendered into the
    system prompt.  Defaults true so connector-mounted servers — and any
    third-party server that ships useful affordance prose — light up
    automatically.  Set false to opt out per agent (unfamiliar prose,
    noisy servers).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["url"] = "url"
    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1)
    include_instructions: bool = True


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


# ── HTTP server declaration ──────────────────────────────────────────────────


class HttpPermissionPolicy(BaseModel):
    """Wrapper matching the ``{type: "always_allow"}`` shape used for MCP."""

    model_config = ConfigDict(extra="forbid")

    type: PermissionPolicy


class HttpRouteSpec(BaseModel):
    """One entry in an ``HttpServerSpec.routes`` allowlist.

    ``path_pattern`` is a glob against the request path (``*`` matches one
    segment, ``**`` matches any number of segments).  ``description`` is
    operator-authored prose rendered into the system prompt so the agent
    knows what the route does and how to call it.  ``permission_policy``
    gates *invocation*: ``always_ask`` leaves the call unresolved in the
    event log until the client confirms via
    ``POST /sessions/:id/tool-confirmations``.
    """

    model_config = ConfigDict(extra="forbid")

    path_pattern: str = Field(min_length=1)
    description: str | None = None
    enabled: bool = True
    permission_policy: HttpPermissionPolicy | None = None


class HttpServerSpec(BaseModel):
    """One entry in an agent's ``http_servers`` list.

    Declares an authenticated HTTP endpoint the agent can reach via the
    ``http_request`` built-in tool.  ``base_url`` is the common URL
    prefix the agent's ``path`` argument is appended to; ``routes`` is
    the allowlist of path patterns the broker permits.  Credentials are
    resolved at request time from the session's bound vaults, keyed on
    ``base_url``.  Secret never enters the sandbox — the worker authors
    the ``Authorization`` header from the vault credential.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=1)
    description: str | None = None
    routes: list[HttpRouteSpec] = Field(default_factory=list)


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
    ``"always_ask"`` leaves the call unresolved in the event log until
    the client confirms or denies via
    ``POST /sessions/:id/tool-confirmations``. Pending calls surface on
    ``Session.awaiting`` so clients can list what they need to act on.
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
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    litellm_extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific LiteLLM kwargs merged into every model "
            "request for this agent.  Common shapes: OpenRouter "
            "``extra_body.provider.order`` for provider pinning, "
            "Anthropic ``thinking``, OpenAI ``reasoning_effort``, raw "
            "sampling knobs (``temperature``, ``max_tokens``), "
            "``api_base`` for self-hosted inference.  Validated by "
            "LiteLLM / the provider; bad kwargs surface as tool-path "
            "errors the model sees.  Security: ``api_base`` redirects "
            "the model call — treat operator-set agents as trusted "
            "and don't accept this field from untrusted principals."
        ),
    )
    window_min: int = Field(default=50_000, ge=1)
    window_max: int = Field(default=150_000, ge=1)


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
    http_servers: list[HttpServerSpec] | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
    litellm_extra: dict[str, Any] | None = None
    window_min: int | None = Field(default=None, ge=1)
    window_max: int | None = Field(default=None, ge=1)


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
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    description: str | None
    metadata: dict[str, Any]
    litellm_extra: dict[str, Any] = Field(default_factory=dict)
    window_min: int
    window_max: int
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
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    litellm_extra: dict[str, Any] = Field(default_factory=dict)
    window_min: int
    window_max: int
    created_at: datetime


# ── Tool-name + permission helpers ───────────────────────────────────────────


def is_mcp_tool_name(name: str) -> bool:
    """True if ``name`` is the namespaced form ``mcp__<server>__<tool>``."""
    return name.startswith("mcp__")


def resolve_permission(name: str, agent_tools: list[ToolSpec]) -> PermissionPolicy | None:
    """Look up the permission policy for a built-in or custom tool by name."""
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.permission
    return None


def resolve_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> PermissionPolicy | None:
    """Look up the permission policy for an MCP tool by namespaced name.

    Returns the matched ``mcp_toolset`` entry's
    ``default_config.permission_policy.type`` (or its bare ``permission``
    if no policy is set), or ``None`` if no entry matches. ``None``
    callers fall back to ``AIOS_DEFAULT_MCP_PERMISSION_POLICY``.
    """
    server_name = name.split("__", 2)[1]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            cfg = spec.default_config
            if cfg and cfg.permission_policy:
                return cfg.permission_policy.type
            return spec.permission
    return None
