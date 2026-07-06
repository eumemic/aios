"""The single source of truth for the authority disposition of a tool call.

The *authority disposition of an unresolved tool call* — is it dispatched by
the harness now, an MCP ``always_allow`` immediate, a needs-confirmation
``always_ask``, a client-executed ``custom`` tool, or an unknown MCP tool — is
**one** domain concept.  Historically it was encoded as three independent
hand-rolled decision trees (``loop._classify_tool_call``,
``sessions._classify_awaiting``, ``sweep._was_dispatched``), each re-walking the
identical primitive ladder::

    is_mcp_tool_name → effective_mcp_permission
                     → registry.has → resolve_permission
                     → ToolDefinition.classify_permission   (route refinement)

They were coupled only by prose ("uses the same permission resolution as the
step function"), and they drifted: ``_was_dispatched`` was left without the
``classify_permission`` route refinement, so a confirmation-pending route-gated
``always_ask`` ``http_request`` was misclassified as *dispatched* by the
recovery sweep — which then fabricated an error tool-result that killed a parked
human-in-the-loop confirmation (the exact outcome
``sweep._is_client_result_pending``'s docstring forbids).

This module walks the ladder **once** and returns a :class:`ToolDisposition`.
The three consumers become thin, one-line projections of the disposition; the
route refinement now exists in exactly one place and *cannot* be present in two
copies and absent in the third.

Graph placement: this is a leaf — it imports no consumer.  The tool registry
and the argument parser are **late-imported** inside :func:`classify_tool_call`
(not at module load), preserving the existing cycle break documented in
``services.sessions`` (``aios.tools`` package-init → ``services.wake`` →
``services.sessions``).  Importing this module never triggers tool-package init.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from aios.models.agents import is_mcp_tool_name, resolve_permission
from aios.services.agents import effective_mcp_permission

if TYPE_CHECKING:
    from aios.models.agents import StepSurface


class ToolDisposition(Enum):
    """The five authority outcomes the system cares about for a tool call.

    * :attr:`IMMEDIATE` — built-in/custom-registered tool whose effective
      permission is ``always_allow`` (or whose ``always_ask`` gate has already
      been resolved): the harness dispatches it now.
    * :attr:`MCP_IMMEDIATE` — known MCP tool with ``always_allow``: dispatched
      now via the MCP path.
    * :attr:`NEEDS_CONFIRM` — built-in or MCP tool gated on an *unresolved*
      ``always_ask`` confirmation: the harness holds it, waiting on the USER.
    * :attr:`CUSTOM` — client-executed custom tool (name not in the registry,
      not MCP-namespaced): the harness holds it, waiting on the CLIENT.
    * :attr:`UNKNOWN_MCP` — MCP-namespaced tool whose server is not registered
      on the agent: routed to an immediate tool-error so the model
      self-corrects.  Only distinguishable when an ``mcp_server_map`` is
      supplied (the dispatch path has one; the read/sweep paths do not and
      treat an unknown MCP server like any other MCP tool).
    """

    IMMEDIATE = "immediate"
    MCP_IMMEDIATE = "mcp_immediate"
    NEEDS_CONFIRM = "needs_confirm"
    CUSTOM = "custom"
    UNKNOWN_MCP = "unknown_mcp"


def classify_tool_call(
    name: str,
    arguments: Any,
    agent: StepSurface,
    *,
    confirmation_resolved: bool,
    mcp_server_map: dict[str, Any] | None = None,
) -> ToolDisposition:
    """Classify a tool call into its authority :class:`ToolDisposition`.

    This is the **one** walk of the permission ladder shared by all three
    consumers.  It composes the four existing primitives exactly once:
    ``is_mcp_tool_name`` / ``effective_mcp_permission`` (MCP branch),
    ``registry.has`` + ``resolve_permission`` + ``ToolDefinition``-driven
    ``classify_permission`` route refinement (built-in branch).

    Parameters
    ----------
    name:
        The tool-call function name (namespaced ``mcp__<server>__<tool>`` for
        MCP tools).
    arguments:
        The raw ``function.arguments`` value (str or dict, provider-dependent),
        used for arg-aware route refinement.  ``None`` / unparseable args fall
        through to the tool's base permission — the schema validator then emits
        a typed error the model can self-correct from.
    agent:
        The agent (or frozen agent version) whose ``tools`` carry the
        permission policies.
    confirmation_resolved:
        Whether *this call's* ``always_ask`` gate has already been satisfied.
        The three consumers each supply the same bit from their own world:
        the loop's fresh dispatch is never pre-confirmed (``False``); the
        awaiting view supplies ``has_allow_lifecycle``; the sweep supplies
        ``tool_call_id in confirmed_ids``.  When ``True``, an otherwise
        ``always_ask`` call projects to an immediate disposition (it is no
        longer awaiting anyone).
    mcp_server_map:
        Optional map of registered MCP server names (the dispatch path passes
        ``{s.name: s for s in agent.mcp_servers}``).  When supplied, an
        MCP-namespaced tool whose server is absent classifies as
        :attr:`ToolDisposition.UNKNOWN_MCP`.  When ``None`` (the read and
        sweep paths, which carry no server map), the unknown-server case is not
        distinguished — it resolves through the normal MCP permission ladder,
        preserving those consumers' historical behavior.
    """
    # Late import: keeps this module a graph leaf and preserves the
    # aios.tools package-init cycle break (services.wake → services.sessions).
    from aios.tools.invoke import parse_arguments
    from aios.tools.registry import registry as tool_registry

    if is_mcp_tool_name(name):
        if mcp_server_map is not None:
            server_name = _mcp_server_name(name)
            if server_name is None or server_name not in mcp_server_map:
                return ToolDisposition.UNKNOWN_MCP
        if effective_mcp_permission(name, agent.tools) == "always_allow":
            return ToolDisposition.MCP_IMMEDIATE
        if confirmation_resolved:
            return ToolDisposition.MCP_IMMEDIATE
        return ToolDisposition.NEEDS_CONFIRM

    if not tool_registry.has(name):
        # Unknown bare name → client-executed custom tool.
        return ToolDisposition.CUSTOM

    perm_tool = resolve_permission(name, agent.tools)
    perm_route: str | None = None
    tool_def = tool_registry.get(name)
    if tool_def.classify_permission is not None:
        # Arg-aware refinement: tools like ``http_request`` resolve a per-call
        # policy from the parsed arguments + agent config (e.g. the matched
        # route's ``permission_policy``).  Malformed args fall through to the
        # base permission so the schema validator emits a typed error.
        args = parse_arguments(arguments)
        if args is not None:
            perm_route = tool_def.classify_permission(args, agent)

    if (perm_tool == "always_ask" or perm_route == "always_ask") and not confirmation_resolved:
        return ToolDisposition.NEEDS_CONFIRM
    return ToolDisposition.IMMEDIATE


def _mcp_server_name(name: str) -> str | None:
    """Server segment of a ``mcp__<server>__<tool>`` name, or ``None`` if malformed."""
    parts = name.split("__", 2)
    if len(parts) < 3 or not parts[1]:
        return None
    return parts[1]
