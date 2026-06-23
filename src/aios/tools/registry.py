"""Tool registry — name → ToolSpec (JSON schema + handler).

Design lifted from ``~/code/hermes-agent/tools/registry.py``: a module-level
singleton that tool modules register themselves against at import time.
The harness imports :mod:`aios.tools` once at worker startup, which causes
every built-in tool module to run its registration call.

Each :class:`ToolDefinition` holds:

* ``name`` — the string the model uses in ``tool_calls[*].function.name``
* ``description`` — what the tool does, shown to the model
* ``parameters_schema`` — JSON Schema describing the ``arguments`` shape
* ``handler`` — an async callable that runs the tool
* ``transport`` — which callers may invoke the tool (``cli`` /
  ``agent_tool`` / ``both``); see ``ToolTransport`` for the security
  frontier framing

The :func:`to_openai_tools` helper translates an agent's ``tools`` list
(``[{type: 'bash'}, {type: 'read'}, ...]``) into the
chat-completions/OpenAI ``tools`` parameter that LiteLLM expects, filtering
``cli``-only tools out of the model's view. :func:`effective_transport`
resolves a tool's transport given the agent's overrides — used by the
``ToolBroker`` (CLI side) and by MCP discovery (model side).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from aios.errors import AiosError
from aios.models.agents import PermissionPolicy, ToolTransport
from aios.models.agents import ToolSpec as AgentToolSpec


class ToolNotFoundError(AiosError):
    """Raised when a tool name is referenced but not registered."""

    error_type = "tool_not_found"
    status_code = 404


class DuplicateToolError(AiosError):
    """Raised when a tool tries to register under a name already in use."""

    error_type = "duplicate_tool"
    status_code = 500


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Rich return from a tool handler.

    Most handlers just return a result ``dict`` and :mod:`aios.harness.tool_dispatch`
    serialises it as JSON into the tool-role message's ``content``.
    Handlers that need to attach structured metadata or a plain-string
    content to the resulting event — notably ``switch_channel`` which
    stamps a marker consumed by the unread-derivation helpers — return
    ``ToolResult`` instead.

    * ``content`` — ``str``: used verbatim.  ``dict``: JSON-encoded.
      ``list[dict]``: passed through as a content-parts list (multimodal
      tool results, e.g. ``read`` returning an image; the chat-completions
      wire format already accepts ``content: list[dict]`` on tool messages
      and LiteLLM translates per provider).
    * ``metadata`` — merged into the event's ``data.metadata`` (stripped
      from the chat-completions wire shape by ``_strip_to_spec``).
    * ``is_error`` — records a handler-level error (separate from
      dispatch-level exceptions, which raise).
    """

    content: str | dict[str, Any] | list[dict[str, Any]]
    metadata: dict[str, Any] | None = None
    is_error: bool = False


# Handler signature: async (session_id, arguments) -> result dict | ToolResult.
# Plain dict → serialised as JSON into the tool message's ``content``.
# :class:`ToolResult` → richer shape with optional per-event metadata.
# Handlers do NOT call append_event themselves —
# :mod:`aios.harness.tool_dispatch` does that.
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | ToolResult]]

# Arg-aware permission classifier. Lets a tool declare a permission policy
# that depends on the parsed arguments and the agent config — used by
# ``http_request`` to gate on per-route policy in ``agent.http_servers``
# rather than a single tool-level policy. ``agent`` is left untyped here
# to avoid a circular import; consumers pass an
# :class:`aios.models.agents.Agent` or :class:`AgentVersion`.
ClassifyPermission = Callable[[dict[str, Any], Any], PermissionPolicy | None]


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """A registered tool's full definition.

    ``transport`` is the registry-level default classification: which
    callers may invoke this tool absent an agent-level override. See
    :data:`aios.models.agents.ToolTransport`. Outbound-side-effect tools
    ship as ``"agent_tool"`` so the model stays the bottleneck for
    irreversible effects; an operator can override per-agent via
    ``ToolSpec.transport``.

    ``executes`` is the execution-class of the handler — ``"worker"`` (the
    default: owner-agnostic network/credential tools the worker runs in-process)
    or ``"sandbox"`` (the filesystem tools that need a provisioned container).
    The workflow-run step reads it to route a ``tool('bash')`` call to the
    run-sandbox executor; the session path doesn't consult it (every built-in
    already runs against the session's own sandbox there).

    ``resumable`` marks a **pure-await** tool: its handler parks on a durable
    servicer edge (a pure read of the log), so the ghost-repair sweep RE-PARKS it
    on crash recovery instead of synthesizing an error result — re-reading the
    durable answer is side-effect-free, unlike bash/http (#1431). The sweep derives
    its resumable set from this flag (:meth:`ToolRegistry.resumable_tool_names`), so
    a new parking tool just sets ``resumable=True`` here at registration — there is
    no separate hand-maintained name list to keep in lockstep.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: ToolHandler
    transport: ToolTransport = "both"
    executes: Literal["worker", "sandbox"] = "worker"
    classify_permission: ClassifyPermission | None = None
    resumable: bool = False


@dataclass(slots=True)
class ToolRegistry:
    """Singleton registry. Import :data:`registry` from this module."""

    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        handler: ToolHandler,
        transport: ToolTransport = "both",
        executes: Literal["worker", "sandbox"] = "worker",
        classify_permission: ClassifyPermission | None = None,
        resumable: bool = False,
    ) -> None:
        """Register a tool. Raises :class:`DuplicateToolError` on name clash.

        Intended to be called at module import time from tool modules
        (e.g. :mod:`aios.tools.bash`). See :class:`ToolDefinition` for ``resumable``.
        """
        if name in self._tools:
            raise DuplicateToolError(
                f"tool {name!r} already registered",
                detail={"name": name},
            )
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            handler=handler,
            transport=transport,
            executes=executes,
            classify_permission=classify_permission,
            resumable=resumable,
        )

    def resumable_tool_names(self) -> frozenset[str]:
        """Names of the registered pure-await tools (``resumable=True``) — the
        ghost-repair sweep's re-park discriminant. Single source of truth: derived
        from the registrations, never a hand-maintained list (#1431)."""
        return frozenset(name for name, tool in self._tools.items() if tool.resumable)

    def get(self, name: str) -> ToolDefinition:
        """Return the registered tool. Raises :class:`ToolNotFoundError` if absent."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"tool {name!r} is not registered",
                detail={"name": name},
            )
        return tool

    def tool_executes_class(self, name: str) -> str:
        """Execution-class of ``name`` — ``"sandbox"`` for the filesystem built-ins,
        ``"worker"`` otherwise (including an unregistered name, so the run-step
        router never crashes on a tool its gate will reject as a value)."""
        tool = self._tools.get(name)
        return tool.executes if tool is not None else "worker"

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def clear(self) -> None:
        """Drop all registrations. Used by unit tests."""
        self._tools.clear()


registry = ToolRegistry()


def openai_tool_entry(tool: ToolDefinition) -> dict[str, Any]:
    """The chat-completions ``tools`` entry for a registered built-in.

    One source of truth for the registry → OpenAI tool-spec shape, shared by
    :func:`to_openai_tools` and the injected built-ins (``switch_channel`` and
    the workflow ``return``/``error`` completion tools).
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def to_openai_tools(agent_tools: list[AgentToolSpec]) -> list[dict[str, Any]]:
    """Translate an agent's declared tools into the OpenAI ``tools`` param.

    Takes the agent's ``tools: [{type: 'bash'}, {type: 'read'}, ...]`` list
    and returns the chat-completions shape:

    .. code-block::

        [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "...",
                    "parameters": {<json schema>},
                },
            },
            ...
        ]

    Tools whose effective transport is ``"cli"`` are filtered out — they
    aren't reachable from the model. Unknown tool types raise
    :class:`ToolNotFoundError`. If the agent declares no tools (or every
    tool resolves to ``"cli"``), returns an empty list.
    """
    result: list[dict[str, Any]] = []
    seen_builtins: set[str] = set()
    for entry in agent_tools:
        if not entry.enabled:
            continue
        if entry.type == "mcp_toolset":
            continue  # MCP tools are added via discovery, not from registry.
        if entry.type == "custom":
            # Custom tools carry their own schema — not in the registry.
            effective: ToolTransport = entry.transport or "both"
            if effective == "cli":
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description or "",
                        "parameters": entry.input_schema or {},
                    },
                }
            )
        else:
            # A pre-#1419 legacy collapse (invoke_workflow + create_run both →
            # call_workflow) can leave duplicate builtin entries after the read-tolerance
            # rename; emit each builtin name once so the model never sees a dup.
            if entry.type in seen_builtins:
                continue
            seen_builtins.add(entry.type)
            tool = registry.get(entry.type)
            effective = entry.transport or tool.transport
            if effective == "cli":
                continue
            result.append(openai_tool_entry(tool))
    return result


def transport_defaults() -> dict[str, ToolTransport]:
    """Each registered built-in's default transport, for the attenuation operator.

    The pure ``attenuate`` operator resolves a builtin ``ToolSpec.transport=None`` to
    its registry default; it takes this mapping as an argument rather than importing
    the runtime registry singleton (which would couple the bottom-of-graph operator to
    worker startup). Read once by the caller and passed in.
    """
    return {name: registry.get(name).transport for name in registry.names()}


def tool_executes_class(name: str) -> str:
    """Module-level accessor for a built-in's execution-class (``worker``/``sandbox``).

    Mirrors the other module-level registry helpers so callers can patch the
    function on the module. Defaults unregistered names to ``"worker"``.
    """
    return registry.tool_executes_class(name)


def effective_transport(name: str, agent_tools: list[AgentToolSpec]) -> ToolTransport:
    """Resolve a tool's effective transport given an agent's tool list.

    Dispatches by name kind:

    * MCP-namespaced (``mcp__<server>__<tool>``): per-tool ``configs[]``
      entry → ``default_config.transport`` → system default ``"both"``.
    * Built-in / custom: ``ToolSpec.transport`` override → registry
      default for built-ins, ``"both"`` for custom.

    Names not in the registry and not MCP-namespaced (shouldn't normally
    happen — caller should validate name resolution first) get the
    permissive ``"both"`` fallback; the broker's per-call resolution
    chain rejects unknown names before this is reached.
    """
    # Deferred to avoid extending the top imports for a single helper.
    # ``aios.models.agents`` is already imported at module load.
    from aios.models.agents import (
        is_mcp_tool_name,
        resolve_builtin_transport,
        resolve_mcp_transport,
    )

    if is_mcp_tool_name(name):
        return resolve_mcp_transport(name, agent_tools) or "both"
    override = resolve_builtin_transport(name, agent_tools)
    if override is not None:
        return override
    if registry.has(name):
        return registry.get(name).transport
    return "both"
