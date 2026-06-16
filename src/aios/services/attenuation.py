"""Runtime-bound capability attenuation — the pure operator + this process's defaults.

:mod:`aios.models.attenuation` is pure: it takes the operator-default MCP permission and
the builtin transport map as arguments. Every materialize edge (the workflow author edge,
``create_run``, ``create_child_session``) reads the *same* two runtime values — the
``AIOS_DEFAULT_MCP_PERMISSION_POLICY`` setting and the tool registry's transport defaults —
so this module binds them once and exposes ``clamp`` / ``normalize`` over ``Surface``,
keeping the ``or "always_ask"`` / ``transport_defaults()`` plumbing in one place.
"""

from __future__ import annotations

from typing import Any

from aios.config import get_settings
from aios.models.agents import PermissionPolicy, ToolTransport
from aios.models.attenuation import (
    Surface,
    api_base_of,
    api_base_trusted,
    attenuate,
    canonicalize,
)
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


def model_identity_trusted(
    child_litellm_extra: dict[str, Any] | None,
    launcher_litellm_extra: dict[str, Any] | None,
) -> bool:
    """The model-identity clamp (#823) bound to this process's operator allowlist.

    Admits a child's model identity iff its effective ``api_base`` equals the
    launcher's or sits in the operator ``trusted_inference_api_bases`` allowlist. The
    spawn edge (``workflows/step.py`` ``_open_agent_capability``) calls this before
    writing a named-agent child and **fails closed** — journaling a catchable
    ``untrusted_api_base`` rejection — when it returns ``False``, so the child's
    context is never sent to an untrusted endpoint.

    A workflow run (the launcher) carries no ``litellm_extra`` of its own, so
    ``launcher_litellm_extra`` is ``None`` in the run→agent() case and the equality
    arm reduces to "the child must not redirect" — the allowlist is the only way to
    admit a redirected endpoint.
    """
    allowlist = get_settings().trusted_inference_api_bases
    return api_base_trusted(
        api_base_of(child_litellm_extra),
        launcher_api_base=api_base_of(launcher_litellm_extra),
        allowlist=allowlist,
    )
