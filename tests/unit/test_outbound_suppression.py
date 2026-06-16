"""Unit tests for the outbound-suppression classification helpers (#710).

The pure decision functions live in :mod:`aios.models.agents`:

* ``http_route_suppressed`` — method-default with per-route override.
* ``mcp_tool_suppressed`` — default-deny with per-tool ``read_allow`` opt-in.
"""

from __future__ import annotations

import pytest

from aios.models.agents import (
    HttpRouteSpec,
    McpToolConfig,
    ToolSpec,
    http_route_suppressed,
    mcp_tool_suppressed,
)


def _route(pattern: str = "/x", *, suppress: bool | None = None) -> HttpRouteSpec:
    return HttpRouteSpec(path_pattern=pattern, suppress=suppress)


class TestHttpRouteSuppressed:
    @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "get"])
    def test_reads_pass_by_default(self, method: str) -> None:
        assert http_route_suppressed(_route(), method) is False

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "post"])
    def test_writes_suppressed_by_default(self, method: str) -> None:
        assert http_route_suppressed(_route(), method) is True

    def test_override_true_suppresses_a_read(self) -> None:
        assert http_route_suppressed(_route(suppress=True), "GET") is True

    def test_override_false_passes_a_write(self) -> None:
        assert http_route_suppressed(_route(suppress=False), "POST") is False


def _mcp_toolset(server: str, configs: list[McpToolConfig] | None = None) -> ToolSpec:
    return ToolSpec(
        type="mcp_toolset",
        mcp_server_name=server,
        configs=configs or [],
    )


class TestMcpToolSuppressed:
    def test_default_deny_unconfigured_tool(self) -> None:
        # A discovered tool with no matching config entry is suppressed.
        tools = [_mcp_toolset("gh")]
        assert mcp_tool_suppressed("mcp__gh__create_issue", tools) is True

    def test_read_allow_opts_in(self) -> None:
        tools = [_mcp_toolset("gh", [McpToolConfig(name="search", read_allow=True)])]
        assert mcp_tool_suppressed("mcp__gh__search", tools) is False

    def test_configured_without_read_allow_is_suppressed(self) -> None:
        tools = [_mcp_toolset("gh", [McpToolConfig(name="create_issue")])]
        assert mcp_tool_suppressed("mcp__gh__create_issue", tools) is True

    def test_unknown_server_is_suppressed(self) -> None:
        tools = [_mcp_toolset("gh", [McpToolConfig(name="search", read_allow=True)])]
        assert mcp_tool_suppressed("mcp__other__search", tools) is True

    def test_malformed_name_is_suppressed(self) -> None:
        assert mcp_tool_suppressed("mcp__gh", []) is True

    def test_read_allow_only_applies_to_named_tool(self) -> None:
        tools = [_mcp_toolset("gh", [McpToolConfig(name="search", read_allow=True)])]
        # A sibling tool not named in configs is still default-deny.
        assert mcp_tool_suppressed("mcp__gh__create_issue", tools) is True
