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
    "list_related_sessions",
    "http_request",
    "trigger_create",
    "trigger_remove",
    "trigger_update",
    "trigger_list",
    "create_workflow",
    "update_workflow",
    "archive_workflow",
    "unarchive_workflow",
    "cancel_run",
    "resume_gate",
    "get_workflow",
    "list_workflows",
    "get_run",
    "list_runs",
    "list_run_events",
    "call_session",
    "call_agent",
    "call_workflow",
    "skill_upsert",
    "skill_archive",
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

# HTTP methods an ``http_request`` route may be scoped to. Mirrors the broker's
# ``_ALLOWED_METHODS`` tuple in ``tools/http_request.py``.
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

_BUILTIN_NAMES: frozenset[str] = frozenset(get_args(BuiltinToolType))

# Read-tolerance for the #1419 invoke*→call_* rename. Agent/workflow/run/session rows
# persisted before the rename carry these pre-rename builtin tool names in their `tools`
# JSONB; without a map they'd fail `ToolSpec` validation on read (a deploy-breaker —
# every agent that exposed workflow-launch/session-invoke to its model would 500). The
# `mode="before"` validator below maps them so old rows still load; the two-step
# create_run/await_run launch tools fold into the unified `call_workflow`. The data
# migration (0116) rewrites the persisted rows to canonical; once it has run everywhere
# this map + the validator can be removed.
_LEGACY_BUILTIN_RENAMES: dict[str, str] = {
    "invoke": "call_session",
    "invoke_agent": "call_agent",
    "invoke_workflow": "call_workflow",
    "create_run": "call_workflow",
    "await_run": "call_workflow",
}

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
    """Per-tool override within an ``mcp_toolset`` entry.

    ``read_allow`` opts a single discovered tool into the outbound-suppression
    read allowlist (#710): MCP has no HTTP-method convention, so when a session
    runs with ``outbound_suppression == "on"`` every MCP call is *default-deny*
    (suppressed with a synthesized success) UNLESS the operator marked the
    specific tool ``read_allow=True`` at config time. A read-allowed tool runs
    for real even under suppression. Default ``False`` — the safe choice for a
    protocol that can't self-describe side effects.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None
    transport: ToolTransport | None = None
    read_allow: bool = False


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

    ``methods`` scopes the route to a set of HTTP verbs so a surface can
    express read/write attenuation structurally — e.g. ``GET`` everywhere
    but ``POST`` only on a sandbox path (#828).  ``None`` (the default)
    means *all* methods are allowed (the method-dimension lattice top;
    backward-compatible with routes authored before method scoping).  A
    non-empty list restricts the route to exactly those verbs.  An empty
    list (``[]``) is *deny-all* — the method-dimension lattice bottom, and
    the natural result of intersecting two disjoint method sets during
    attenuation; it matches nothing.  The capability meet
    (:mod:`aios.models.attenuation`) intersects ``methods`` per route, so a
    child surface can narrow a parent route's verbs but never widen them.

    GraphQL caveat — **REST-only discipline for attenuated surfaces.**
    Method scoping confines REST read/write because the verb encodes the
    semantics.  A GraphQL endpoint serves both queries (reads) and
    mutations (writes) over a single ``POST`` path, so method scoping
    *cannot* separate read from write there, and the broker does not
    inspect request bodies.  An operator who needs to confine writes on a
    GraphQL surface must place reads and writes behind distinct
    ``base_url`` servers (each with its own credential/route allowlist) or
    accept that granting ``POST`` grants both.

    ``allow_query`` opts the route into permitting a query string on the
    request ``path``.  The default is ``False``: a ``?...`` is rejected at
    the route gate (#485) because ``httpx`` parses the query off the URL
    and an unanticipating allowlist — e.g. a read-only ``/lights/*`` — is
    bypassed when the upstream interprets ``?action=delete`` as a write.
    An operator sets ``allow_query=True`` only on routes where the query
    string cannot escalate beyond what the route already grants and where
    it is functionally required — e.g. a GitHub ``/repos/**`` route that
    already permits every verb and must follow cursor/``page`` pagination
    to read a full comment thread (#1156).  The path portion is still
    glob-matched against ``path_pattern`` (the query is stripped before
    the match), and ``.``/``..`` dot-segment rejection still applies, so a
    query allowance never widens the path-dimension grant.  It is
    launcher-verbatim under attenuation: a child surface cannot turn it on
    where the parent left it off.
    """

    model_config = ConfigDict(extra="forbid")

    path_pattern: str = Field(min_length=1)
    description: str | None = None
    enabled: bool = True
    permission_policy: HttpPermissionPolicy | None = None
    methods: list[HttpMethod] | None = None
    allow_query: bool = False
    # Outbound-suppression override (#710). When a session runs with
    # ``outbound_suppression == "on"`` the broker classifies each matched call
    # as read (passes through) or write (suppressed with a synthesized success
    # + audit event). The default classifier is the HTTP method: ``GET`` is a
    # read, ``POST``/``PUT``/``PATCH``/``DELETE`` are writes. ``suppress`` is the
    # per-route escape hatch for the cases the method doesn't predict: set it
    # ``True`` to suppress a side-effecting ``GET``, ``False`` to let a
    # read-only ``POST`` (a GraphQL/JSON-RPC query, a search endpoint) pass.
    # ``None`` (default) defers to the method classifier. Off when suppression
    # is off — this never gates a normal request.
    suppress: bool | None = None


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
    # Status code returned on a suppressed call's synthesized success response
    # (#710). The synthesized body is empty (``""``) so a JSON parse yields
    # nothing surprising; the status defaults to ``200``. Configurable per
    # http_server for surfaces whose success contract is e.g. ``201 Created`` —
    # the agent must observe a plausible success so its behavior validates.
    suppressed_response_status: int = Field(default=200, ge=100, le=599)


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


# ── names-only http_server declaration (Ask 3 from #939) ──────────────────────
#
# A workflow author may reference a grant the acting agent already holds by *name
# alone* — ``http_servers: ["davenant"]`` — instead of reconstructing the full
# ``HttpServerSpec(name=..., base_url=...)`` whose identity must match the agent's.
# The bare name is resolved against the acting agent's servers at the authoring
# edge (``aios.services.workflows``); the agent's ``base_url`` + frozen routes are
# then inherited launcher-frozen into storage exactly as the #949 identity-match
# path already does. This is **pure surface ergonomics**: a names-only entry can
# only resolve to a server the agent already has, so it grants no new authority
# and the run-time parent-wins-frozen resolution (keyed on the verbatim agent name)
# is untouched.
#
# Resolution lives in the service (it needs the acting agent); aliasing — declaring
# the agent's server under a *different* name with ``server_ref`` resolving against
# the alias — is an open fork (#953) deliberately NOT shipped here: it would require
# a run-time resolution change beyond the committed surface-only scope.

HttpServerRef = str | HttpServerSpec
"""An authoring-edge http_server entry: a bare name (names-only sugar, resolved
against the acting agent) or a full ``HttpServerSpec`` (identity-match, #949)."""


def resolve_http_server_refs(
    refs: list[HttpServerRef], agent_servers: list[HttpServerSpec]
) -> list[HttpServerSpec]:
    """Resolve names-only entries against the acting agent's ``http_servers``.

    Each ``str`` entry is replaced by ``HttpServerSpec(name=<name>, base_url=<the
    agent's base_url for that name>, routes=[])`` — an empty-routes identity spec
    the existing authoring gate then admits by identity and inherits frozen routes
    into. A name with no matching agent server raises ``ValueError`` (the author
    referenced a grant the agent does not hold). Full ``HttpServerSpec`` entries
    pass through verbatim (the #949 identity-match path).

    Resolution is by name; if the agent declares the same name at multiple
    ``base_url``s the first is taken (agents validate ``base_url`` uniqueness, not
    name uniqueness, so duplicate names are possible but the authoring gate then
    flags any genuine mismatch downstream).
    """
    by_name: dict[str, HttpServerSpec] = {}
    for s in agent_servers:
        by_name.setdefault(s.name, s)
    out: list[HttpServerSpec] = []
    for ref in refs:
        if isinstance(ref, str):
            agent_server = by_name.get(ref)
            if agent_server is None:
                raise ValueError(
                    f"http_servers references {ref!r}, which the acting agent does not grant"
                )
            out.append(HttpServerSpec(name=ref, base_url=agent_server.base_url, routes=[]))
        else:
            out.append(ref)
    return out


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

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_builtin_names(cls, data: Any) -> Any:
        """Map pre-#1419-rename builtin tool names to canonical ones (read-tolerance).

        Runs before field validation so a row persisted with a legacy ``type``
        (``invoke``/``invoke_agent``/``invoke_workflow``/``create_run``/``await_run``)
        still validates against the post-rename ``BuiltinToolType`` Literal. The
        ``create_run``/``await_run`` collapse to ``call_workflow`` is deduped at the list
        level (``to_openai_tools`` + migration 0116), not here. Temporary shim — remove
        once 0116 has rewritten all persisted rows. See :data:`_LEGACY_BUILTIN_RENAMES`.
        """
        if isinstance(data, dict):
            t = data.get("type")
            if isinstance(t, str) and t in _LEGACY_BUILTIN_RENAMES:
                data = {**data, "type": _LEGACY_BUILTIN_RENAMES[t]}
        return data

    @model_validator(mode="after")
    def _check_type_fields(self) -> ToolSpec:
        if self.type == "custom":
            missing = [
                f for f in ("name", "description", "input_schema") if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(f"custom tools require: {', '.join(missing)}")
            # ``mcp__`` is the reserved MCP namespace (see is_mcp_tool_name): a
            # custom tool with that prefix is classified as an MCP call by
            # _classify_tool_call BEFORE the custom fallback, so it routes to
            # the MCP dispatcher (erroring as an unknown server) and is never
            # held for the client. Reject it here, at the operator-controlled
            # definition layer, rather than letting it silently never work.
            if self.name is not None and is_mcp_tool_name(self.name):
                raise ValueError(
                    f"custom tool name {self.name!r} must not start with the reserved "
                    "'mcp__' prefix (it namespaces MCP-dispatched tools)"
                )
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


# ── Outbound-suppression classification (#710) ───────────────────────────────

# HTTP methods that pass through unchanged under outbound suppression by
# default (reads). Everything else (POST/PUT/PATCH/DELETE) is a write and is
# suppressed. A per-route ``suppress`` override flips either direction.
_SUPPRESSION_READ_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def http_route_suppressed(route: HttpRouteSpec, method: str) -> bool:
    """Decide whether a matched HTTP route is suppressed for ``method``.

    The default classifier is the HTTP method — ``GET`` (and the other safe
    verbs) read and pass through; ``POST``/``PUT``/``PATCH``/``DELETE`` write
    and are suppressed. The route's optional ``suppress`` override wins when
    set: ``True`` suppresses a side-effecting read, ``False`` lets a read-only
    write through. This is *only* consulted when the session's
    ``outbound_suppression`` is ``"on"`` — callers gate on that first.
    """
    if route.suppress is not None:
        return route.suppress
    return method.upper() not in _SUPPRESSION_READ_METHODS


def mcp_tool_suppressed(name: str, agent_tools: list[ToolSpec]) -> bool:
    """Decide whether an MCP tool call is suppressed under outbound suppression.

    MCP has no method convention, so the policy is *default-deny*: every MCP
    call is suppressed UNLESS its per-tool ``McpToolConfig.read_allow`` is set
    (an operator opt-in for a known-safe read). Resolution mirrors
    :func:`resolve_mcp_enabled`: a matching ``configs[]`` entry decides; absent
    one, the tool is suppressed. An undeclared/unmatched tool is suppressed
    (the safe default). Callers gate on ``outbound_suppression == "on"`` first.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return True
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return not cfg.read_allow
            return True
    return True


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
