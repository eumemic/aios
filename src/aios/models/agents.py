"""Agent resource: model + system prompt + tools + MCP servers.

Agents are versioned: every update creates a new immutable version. The
``agents`` table holds the latest config; the ``agent_versions`` table stores
the full history. The ``model`` field is a free-form LiteLLM model string
(e.g. ``anthropic/claude-opus-4-6``, ``ollama_chat/llama3.3``).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, get_args

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
    "schedule_wake",
    "wake_session",
    "wake_self",
    "http_request",
    "trigger_create",
    "trigger_remove",
    "trigger_update",
    "trigger_list",
    "create_workflow",
    "update_workflow",
    "create_run",
    "await_run",
    "cancel_run",
]

# Permission policy for built-in tools. Custom tools are always client-controlled
# and ignore this field.
PermissionPolicy = Literal["always_allow", "always_ask"]

# Transport classification — which callers may invoke a tool.
#   "agent_tool": model only (the LLM's tool-call surface).
#   "cli":        sandbox-side ``tool`` CLI only (bash inside the session).
#   "both":       reachable from either.
# The substrate's security frontier: outbound-side-effect tools live as
# ``agent_tool`` so the model is the bottleneck for irreversible effects.
# Enforcement is structural (the broker refuses non-CLI tools). Built-ins
# get a registry default; an agent's ``ToolSpec`` /
# ``McpToolsetConfig`` / ``McpToolConfig`` can override per-tool.
ToolTransport = Literal["cli", "agent_tool", "both"]

_BUILTIN_NAMES: frozenset[str] = frozenset(get_args(BuiltinToolType))

# Header names the MCP streamable-http transport authors on every request
# (see ``mcp.client.streamable_http._prepare_headers``). A spec header named
# after one of these never reaches the wire — the transport overwrites it
# per-request — yet it would still fragment the connection-pool key
# (``_headers_key``). Compared case-insensitively; HTTP header names are too.
_RESERVED_MCP_HEADERS: frozenset[str] = frozenset(
    {"accept", "content-type", "mcp-session-id", "mcp-protocol-version"}
)

# RFC 7230 ``token``: the legal character set for an HTTP header field name.
# Excludes whitespace, control chars, ``:``, and non-ASCII by construction.
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


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

    ``headers`` are extra NON-SECRET HTTP headers sent on every request to
    this server — toolset selectors (e.g. GitHub's
    ``X-MCP-Toolsets: discussions,issues``), format hints, API-version
    pins.  Do NOT put secrets here: this dict is stored in plaintext agent
    JSON.  Real credentials belong in the vault path; a vault-derived auth
    header overrides a same-named entry here (auth headers win on
    collision).  Names must be valid HTTP tokens and values printable ASCII
    (validated below) so they can't fail only at connection time; headers
    the MCP transport authors itself (Accept, Content-Type, Mcp-Session-Id,
    Mcp-Protocol-Version) are rejected — setting them here is a silent no-op.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["url"] = "url"
    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1)
    include_instructions: bool = True
    headers: dict[str, str] | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def _name_not_builtin(cls, v: str) -> str:
        # The ``tool`` CLI uses a flat top-level namespace (``tool <name>``
        # for built-ins, ``tool <server> <method>`` for MCP). Reserving
        # built-in names on ``McpServerSpec`` keeps the broker's
        # name-resolution unambiguous.
        if v in _BUILTIN_NAMES:
            raise ValueError(
                f"MCP server name {v!r} collides with a built-in tool name; "
                f"the `tool` CLI uses a flat top-level namespace where "
                f"built-in tool names are reserved. Pick a different name."
            )
        return v

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        # Reject HTTP-illegal headers at config time. httpx would otherwise
        # raise only when the connection is opened, where the error is caught
        # and the whole server silently dropped (logged at WARN). Fail loud at
        # the write boundary instead.
        if v is None:
            return None
        for name, value in v.items():
            if not _HEADER_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"invalid HTTP header name {name!r}: must be a non-empty RFC 7230 "
                    "token (ASCII letters, digits, and any of !#$%&'*+-.^_`|~)"
                )
            if name.lower() in _RESERVED_MCP_HEADERS:
                raise ValueError(
                    f"header {name!r} is authored by the MCP transport on every request; "
                    "setting it here has no effect on the wire. Remove it."
                )
            # Values: printable ASCII (0x20-0x7E) plus HTAB. CR/LF would enable
            # header injection; non-ASCII can't be encoded on the wire.
            bad = next((c for c in value if c != "\t" and not 0x20 <= ord(c) <= 0x7E), None)
            if bad is not None:
                raise ValueError(
                    f"invalid value for header {name!r}: character {bad!r} is not allowed "
                    "(only printable ASCII and tab — no control chars, CR, LF, or non-ASCII)"
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
    transport: ToolTransport | None = None


class McpToolConfig(BaseModel):
    """Per-tool override within an ``mcp_toolset`` entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None
    transport: ToolTransport | None = None


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


def validate_http_servers(servers: list[HttpServerSpec]) -> None:
    """Cross-item invariants for ingress ``http_servers`` lists.

    ``base_url`` is the credential-resolution key and the attenuation key, so
    each authored surface must contain it at most once.  Equality is exact
    string equality, matching run-time resolution and attenuation semantics.
    """
    seen: set[str] = set()
    for server in servers:
        if server.base_url in seen:
            raise ValueError(f"duplicate base_url {server.base_url!r}")
        seen.add(server.base_url)


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
    # Override the registry default transport for a built-in (or the
    # system ``"both"`` default for a custom tool). ``None`` = inherit.
    # Ignored for ``type == "mcp_toolset"`` — per-server / per-tool MCP
    # transport overrides live on ``default_config`` and ``configs[]``,
    # paralleling ``permission_policy``.
    transport: ToolTransport | None = None

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

    @model_validator(mode="after")
    def _validate_http_servers(self) -> AgentCreate:
        validate_http_servers(self.http_servers)
        return self


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

    @model_validator(mode="after")
    def _validate_http_servers(self) -> AgentUpdate:
        if self.http_servers is not None:
            validate_http_servers(self.http_servers)
        return self


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

    Precedence (mirrors the broker's resolution so the model path and CLI
    path agree on overrides): per-tool ``configs[]`` entry → ``default_config``
    → bare ``ToolSpec.permission``. Returns ``None`` when nothing is set;
    callers then fall back to ``AIOS_DEFAULT_MCP_PERMISSION_POLICY``.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return None
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.permission_policy.type if cfg.permission_policy else None
            if spec.default_config and spec.default_config.permission_policy:
                return spec.default_config.permission_policy.type
            return spec.permission
    return None


def resolve_mcp_transport(name: str, agent_tools: list[ToolSpec]) -> ToolTransport | None:
    """Look up the transport classification for an MCP tool by namespaced name.

    Precedence parallels :func:`resolve_mcp_permission`: per-tool
    ``configs[]`` entry → ``default_config.transport``. Returns ``None``
    when no override is set — callers fall back to the system default
    ``"both"``.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return None
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.transport
            if spec.default_config:
                return spec.default_config.transport
            return None
    return None


def resolve_mcp_enabled(name: str, agent_tools: list[ToolSpec]) -> bool:
    """Resolve whether an MCP tool is enabled given the agent's config.

    Precedence parallels :func:`resolve_mcp_permission`: per-tool
    ``configs[]`` entry → ``default_config.enabled`` → ``True`` (the
    field default). Returns ``False`` if no matching ``mcp_toolset``
    entry exists at all — a tool the agent hasn't declared a toolset
    for is implicitly off.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return False
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.enabled
            if spec.default_config is not None:
                return spec.default_config.enabled
            return True
    return False


def resolve_builtin_transport(name: str, agent_tools: list[ToolSpec]) -> ToolTransport | None:
    """Look up the transport override for a built-in or custom tool by name.

    Returns the matching ``ToolSpec.transport`` (which may be ``None`` if
    the operator left it unset, meaning "inherit the registry default").
    Returns ``None`` if no matching entry exists at all.
    """
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.transport
    return None
