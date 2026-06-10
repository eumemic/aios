"""Runtime-bound capability attenuation — the pure operator + this process's defaults.

:mod:`aios.models.attenuation` is pure: it takes the operator-default MCP permission and
the builtin transport map as arguments. Every materialize edge (the workflow author edge,
``create_run``, ``create_child_session``) reads the *same* two runtime values — the
``AIOS_DEFAULT_MCP_PERMISSION_POLICY`` setting and the tool registry's transport defaults —
so this module binds them once and exposes ``clamp`` / ``normalize`` over ``Surface``,
keeping the ``or "always_ask"`` / ``transport_defaults()`` plumbing in one place.
"""

from __future__ import annotations

from aios.config import get_settings
from aios.models.agents import PermissionPolicy, ToolTransport
from aios.models.attenuation import Surface, attenuate, canonicalize
from aios.tools.registry import transport_defaults


def _defaults() -> tuple[PermissionPolicy, dict[str, ToolTransport]]:
    """The two operator defaults bound to this process: settings policy + registry transports."""
    return get_settings().default_mcp_permission_policy or "always_ask", transport_defaults()


def clamp(declared: Surface, launcher: Surface) -> Surface:
    """``attenuate(declared, launcher)`` bound to this process's operator defaults."""
    default_mcp, builtin_transports = _defaults()
    return attenuate(
        declared,
        launcher,
        default_mcp_permission=default_mcp,
        builtin_transports=builtin_transports,
    )


def normalize(declared: Surface) -> Surface:
    """``canonicalize(declared)`` bound to this process's defaults (the meet's identity element)."""
    default_mcp, builtin_transports = _defaults()
    return canonicalize(
        declared, default_mcp_permission=default_mcp, builtin_transports=builtin_transports
    )
