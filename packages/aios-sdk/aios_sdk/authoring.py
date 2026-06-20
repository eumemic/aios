"""Typed, illegal-states-unrepresentable authoring facade over the generated SDK.

The generated authoring surface (:mod:`aios_sdk._generated`) is faithful to
``openapi.json`` but hostile to hand-authoring: every optional field is a
``None | X | Unset`` union and the tool ``type`` is the bare
``ToolSpecTypeType0 | ToolSpecTypeType1`` union, so illegal kind/field combos
slip through until the **server** rejects them.

This module is a *hand-written sibling* of the generated tree (it is **not**
under the ``aios_sdk/_generated`` excluded glob, so it is linted with
``ruff check`` and typed with ``mypy --strict``). It adds **zero** validation
logic: each constructor builds a plain ``dict`` with exactly the keys legal for
its kind and round-trips it through the generated ``from_dict`` constructor. The
generated model — and ultimately the **server's** ``_check_type_fields``
validator — remains the single source of truth.

Two correctness properties fall out of the structure, not out of any re-validation:

* **Variation is a discriminated KIND, not a flag.** The tool kind is selected by
  *which constructor you call* (:func:`define_tool_builtin` /
  :func:`define_tool_custom` / :func:`define_tool_mcp`), never by a boolean
  parameter — mirroring the source ``ToolSpec.type`` tagged union.
* **Illegal field combos are unrepresentable at author time.** Because
  :func:`define_tool_builtin` has no ``input_schema`` / ``name`` / ``description``
  params and :func:`define_tool_custom` has no ``mcp_server_name`` param, the
  runtime errors ``custom tools require: …`` and ``mcp_toolset requires
  mcp_server_name`` become unspellable rather than deferred ``ValueError``\\ s.

The facade does **not** re-validate kwargs (``litellm_extra``, reserved
``mcp__`` custom-tool names, etc.). Such combos — the ones the structure cannot
catch — surface from the server as raw errors (fail-hard preserved); the facade
never becomes a second validation authority that could drift from the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from aios_sdk._generated.models.agent_create import AgentCreate
from aios_sdk._generated.models.tool_spec import ToolSpec

if TYPE_CHECKING:
    from aios_sdk._generated.models.agent_skill_ref import AgentSkillRef
    from aios_sdk._generated.models.http_server_spec import HttpServerSpec
    from aios_sdk._generated.models.mcp_server_spec import McpServerSpec
    from aios_sdk._generated.models.mcp_tool_config import McpToolConfig
    from aios_sdk._generated.models.mcp_toolset_config import McpToolsetConfig

__all__ = [
    "AgentBuilder",
    "BuiltinToolName",
    "Permission",
    "Transport",
    "define_agent",
    "define_tool_builtin",
    "define_tool_custom",
    "define_tool_mcp",
]

# The built-in tool names, kept in sync with the source ``BuiltinToolType``
# (``src/aios/models/agents.py``) and the generated ``ToolSpecTypeType0`` enum.
# This ``Literal`` is an *ergonomic hint only*, not a second source of truth:
# the constructed ``ToolSpec`` is built from the generated model, whose
# ``ToolSpecTypeType0`` enum is regenerated from ``openapi.json``, so an unknown
# name fails ``ToolSpec.from_dict`` regardless of what this ``Literal`` allows.
BuiltinToolName = Literal[
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
    "create_run",
    "await_run",
    "cancel_run",
    "resume_gate",
    "get_workflow",
    "list_workflows",
    "get_run",
    "list_runs",
    "list_run_events",
    "invoke",
    "invoke_agent",
    "invoke_workflow",
]

# Execution policy for built-in tools.
Permission = Literal["always_allow", "always_ask"]

# Transport override (which call surface a tool is exposed on).
Transport = Literal["cli", "agent_tool", "both"]


def define_tool_builtin(
    type: BuiltinToolName,
    *,
    enabled: bool = True,
    permission: Permission | None = None,
    transport: Transport | None = None,
) -> ToolSpec:
    """A built-in tool, keyed by name (``"bash"``, ``"read"``, …).

    Accepts only the fields legal for a built-in: there is no ``name`` /
    ``description`` / ``input_schema`` / ``mcp_server_name`` parameter, so the
    ``custom`` / ``mcp_toolset`` field requirements are structurally
    unrepresentable here. The author passes plain strings; the generated enums
    are constructed internally by ``ToolSpec.from_dict``.
    """
    d: dict[str, Any] = {"type": type, "enabled": enabled}
    if permission is not None:
        d["permission"] = permission
    if transport is not None:
        d["transport"] = transport
    return ToolSpec.from_dict(d)


def define_tool_custom(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    enabled: bool = True,
    transport: Transport | None = None,
) -> ToolSpec:
    """A custom (client-executed) tool.

    ``name``, ``description`` and ``input_schema`` are required by signature, so
    the runtime ``custom tools require: …`` error is unrepresentable. There is
    no ``mcp_server_name`` parameter. A reserved ``mcp__`` ``name`` is *not*
    caught here — that combo is structurally representable, so the **server's**
    validator remains its single authority (fail-hard preserved).
    """
    d: dict[str, Any] = {
        "type": "custom",
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "enabled": enabled,
    }
    if transport is not None:
        d["transport"] = transport
    return ToolSpec.from_dict(d)


def define_tool_mcp(
    *,
    mcp_server_name: str,
    default_config: McpToolsetConfig | None = None,
    configs: list[McpToolConfig] | None = None,
    enabled: bool = True,
) -> ToolSpec:
    """An MCP toolset, keyed by ``mcp_server_name``.

    ``mcp_server_name`` is required by signature, so the runtime
    ``mcp_toolset requires mcp_server_name`` error is unrepresentable. There is
    no ``name`` / ``description`` / ``input_schema`` parameter.
    """
    d: dict[str, Any] = {
        "type": "mcp_toolset",
        "mcp_server_name": mcp_server_name,
        "enabled": enabled,
    }
    if default_config is not None:
        d["default_config"] = default_config.to_dict()
    if configs is not None:
        d["configs"] = [c.to_dict() for c in configs]
    return ToolSpec.from_dict(d)


class AgentBuilder:
    """Fluent accumulator that constructs a generated ``AgentCreate``.

    Each chained method appends to an internal ``dict``; :meth:`build` returns
    ``AgentCreate.from_dict(self._d)`` so the generated model — and the server —
    stay the validation authority. ``name`` and ``model`` are required up front
    (they have no default in ``AgentCreate``); everything else is optional.
    """

    def __init__(self, name: str, model: str) -> None:
        self._d: dict[str, Any] = {"name": name, "model": model}
        self._tools: list[ToolSpec] = []
        self._skills: list[AgentSkillRef] = []
        self._mcp_servers: list[McpServerSpec] = []
        self._http_servers: list[HttpServerSpec] = []

    def system(self, s: str) -> Self:
        self._d["system"] = s
        return self

    def tool(self, t: ToolSpec) -> Self:
        self._tools.append(t)
        return self

    def tools(self, *t: ToolSpec) -> Self:
        self._tools.extend(t)
        return self

    def skill(self, ref: AgentSkillRef) -> Self:
        self._skills.append(ref)
        return self

    def mcp_server(self, spec: McpServerSpec) -> Self:
        self._mcp_servers.append(spec)
        return self

    def http_server(self, spec: HttpServerSpec) -> Self:
        self._http_servers.append(spec)
        return self

    def window(self, min: int, max: int) -> Self:
        """Map to the generated ``window_min`` / ``window_max`` fields.

        There is no single ``window`` field on ``AgentCreate``.
        """
        self._d["window_min"] = min
        self._d["window_max"] = max
        return self

    def litellm_extra(self, d: dict[str, Any]) -> Self:
        self._d["litellm_extra"] = d
        return self

    def metadata(self, d: dict[str, Any]) -> Self:
        self._d["metadata"] = d
        return self

    def description(self, s: str) -> Self:
        self._d["description"] = s
        return self

    def build(self) -> AgentCreate:
        d = dict(self._d)
        if self._tools:
            d["tools"] = [t.to_dict() for t in self._tools]
        if self._skills:
            d["skills"] = [s.to_dict() for s in self._skills]
        if self._mcp_servers:
            d["mcp_servers"] = [s.to_dict() for s in self._mcp_servers]
        if self._http_servers:
            d["http_servers"] = [s.to_dict() for s in self._http_servers]
        return AgentCreate.from_dict(d)


def define_agent(name: str, model: str) -> AgentBuilder:
    """Entry point: start authoring an agent with the required ``name`` / ``model``."""
    return AgentBuilder(name, model)
