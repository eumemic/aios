"""Unit coverage for :mod:`aios.harness.resource_health` (issue #1720).

Pins the prelude "resource health" surface: a persistent, always-visible
system-prompt line for a resource whose circuit breaker is open, so the
model doesn't silently re-diagnose the same dead credential/down MCP server
turn after turn with no memory of having already tried.
"""

from __future__ import annotations

from aios.harness.resource_health import (
    augment_with_resource_health,
    build_resource_health_block,
)


def test_empty_when_nothing_degraded() -> None:
    assert build_resource_health_block(degraded_repos=[], degraded_mcp_server_names=[]) == ""


def test_augment_leaves_system_prompt_unchanged_when_healthy() -> None:
    base = "you are a helpful agent"
    assert (
        augment_with_resource_health(base, degraded_repos=[], degraded_mcp_server_names=[]) == base
    )


def test_renders_auth_failed_only_for_classified_auth_failures() -> None:
    block = build_resource_health_block(
        degraded_repos=[
            ("/workspace/theme", "2026-07-07T19:53:06+00:00", True, "Authentication failed"),
            ("/workspace/api", "2026-07-08T01:00:00+00:00", True, "remote: Invalid username"),
        ],
        degraded_mcp_server_names=[],
    )
    lines = block.splitlines()
    assert lines[0] == "━━━ Resource health ━━━"
    assert "repos: /workspace/theme AUTH-FAILED since 2026-07-07T19:53:06+00:00" in lines
    assert "repos: /workspace/api AUTH-FAILED since 2026-07-08T01:00:00+00:00" in lines


def test_transient_failure_renders_clone_failing_with_real_cause_not_auth() -> None:
    """A non-auth (transient) failure must NOT be mislabeled AUTH-FAILED — it
    renders CLONE-FAILING with the real, token-redacted git error (#1720)."""
    block = build_resource_health_block(
        degraded_repos=[
            (
                "/workspace/api",
                "2026-07-08T01:00:00+00:00",
                False,
                "git clone timed out after 30.0s",
            ),
        ],
        degraded_mcp_server_names=[],
    )
    lines = block.splitlines()
    assert lines[0] == "━━━ Resource health ━━━"
    assert (
        "repos: /workspace/api CLONE-FAILING since 2026-07-08T01:00:00+00:00 "
        "(last: git clone timed out after 30.0s)" in lines
    )
    assert "AUTH-FAILED" not in block


def test_transient_failure_without_error_text_omits_last_clause() -> None:
    block = build_resource_health_block(
        degraded_repos=[("/workspace/api", "2026-07-08T01:00:00+00:00", False, "")],
        degraded_mcp_server_names=[],
    )
    assert (
        "repos: /workspace/api CLONE-FAILING since 2026-07-08T01:00:00+00:00" in block.splitlines()
    )
    assert "(last:" not in block


def test_renders_one_line_per_degraded_mcp_server() -> None:
    block = build_resource_health_block(
        degraded_repos=[], degraded_mcp_server_names=["github", "linear"]
    )
    lines = block.splitlines()
    assert lines[0] == "━━━ Resource health ━━━"
    assert "mcp: github DEGRADED" in lines
    assert "mcp: linear DEGRADED" in lines


def test_mixed_repo_and_mcp_degradation() -> None:
    block = build_resource_health_block(
        degraded_repos=[("/workspace/theme", "2026-07-07T19:53:06+00:00", True, "auth failed")],
        degraded_mcp_server_names=["github"],
    )
    assert block == (
        "━━━ Resource health ━━━\n"
        "repos: /workspace/theme AUTH-FAILED since 2026-07-07T19:53:06+00:00\n"
        "mcp: github DEGRADED"
    )


def test_augment_appends_block_when_degraded() -> None:
    base = "you are a helpful agent"
    out = augment_with_resource_health(
        base,
        degraded_repos=[("/workspace/theme", "2026-07-07T19:53:06+00:00", True, "auth failed")],
        degraded_mcp_server_names=[],
    )
    assert out.startswith(base)
    assert "━━━ Resource health ━━━" in out
    assert "repos: /workspace/theme AUTH-FAILED since 2026-07-07T19:53:06+00:00" in out


def test_degraded_mcp_names_match_by_identity_not_url() -> None:
    """#2233: two mounts on ONE url are separate circuits, so the prompt must
    name only the identity whose breaker is open. A url-only membership test
    would tell the model a healthy pinned mount is down.

    The unpinned arm stays coarse on purpose: an unpinned mount's identity is
    only knowable at resolution time, so any vault at its url matches it.
    """
    from unittest.mock import patch

    from aios.harness.step_context import _session_degraded_mcp_server_names
    from aios.mcp.pool import McpSessionPool
    from aios.models.agents import McpServerSpec

    url = "https://gmail/mcp"
    pool = McpSessionPool()
    pool.mark_unhealthy(url, "vlt_work", "hk", backoff_s=60.0)

    pinned = [
        McpServerSpec(name="work", url=url, vault_id="vlt_work"),
        McpServerSpec(name="home", url=url, vault_id="vlt_home"),
    ]
    unpinned = [McpServerSpec(name="either", url=url)]

    with patch("aios.harness.runtime.mcp_session_pool", pool):
        assert _session_degraded_mcp_server_names(pinned) == ["work"]
        assert _session_degraded_mcp_server_names(unpinned) == ["either"]
        assert _session_degraded_mcp_server_names([McpServerSpec(name="x", url="https://o")]) == []
