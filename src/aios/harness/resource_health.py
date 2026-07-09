"""Prelude health surface for degraded session resources (#1720).

A dead/expired GitHub credential or a down MCP server used to be a per-turn
rediscovery: the harness silently skipped (or slow-retried) the resource and
the only trace was a transient ``github_clone_failed``/``mcp_server_unavailable``
event the model would never see again next turn. Production evidence (#1720):
one session re-diagnosed the same dead token as a *novel* problem across ~174
turns because there was no persistent "repo X: AUTH-FAILED since <ts>" fact
anywhere in its context.

This module renders that fact as a compact, always-visible system-prompt
block — one line per degraded resource, e.g.::

    ━━━ Resource health ━━━
    repos: davenant-theme AUTH-FAILED since 2026-07-07T19:53:06Z
    mcp: github DEGRADED

— built fresh every step from the two in-memory breakers
(:class:`aios.sandbox.github_clone_breaker.GithubCloneBreaker` and
:class:`aios.mcp.pool.McpSessionPool`), so it disappears the step after the
resource heals (no separate "clear" bookkeeping needed here — the block is
just a projection of current breaker state, never persisted itself).
"""

from __future__ import annotations

from aios.harness._text import join_blocks

_HEADER = "━━━ Resource health ━━━"


def build_resource_health_block(
    *,
    degraded_repos: list[tuple[str, str]],
    degraded_mcp_server_names: list[str],
) -> str:
    """Render the standing degraded-resource block.

    ``degraded_repos`` is a list of ``(mount_path, since_iso)`` pairs — already
    formatted by the caller so this module has no dependency on the breaker's
    dataclass shape. ``degraded_mcp_server_names`` is the agent-declared
    server ``name`` (not URL) for each MCP server whose breaker is currently
    open. Returns ``""`` when nothing is degraded, so the augmenter leaves the
    system prompt unchanged in the common (healthy) case.
    """
    if not degraded_repos and not degraded_mcp_server_names:
        return ""
    lines = [_HEADER]
    for mount_path, since_iso in degraded_repos:
        lines.append(f"repos: {mount_path} AUTH-FAILED since {since_iso}")
    for name in degraded_mcp_server_names:
        lines.append(f"mcp: {name} DEGRADED")
    return "\n".join(lines)


def augment_with_resource_health(
    base_system: str,
    *,
    degraded_repos: list[tuple[str, str]],
    degraded_mcp_server_names: list[str],
) -> str:
    """Append the resource-health block to ``base_system`` if anything is degraded."""
    return join_blocks(
        base_system,
        build_resource_health_block(
            degraded_repos=degraded_repos,
            degraded_mcp_server_names=degraded_mcp_server_names,
        ),
    )
