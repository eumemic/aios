"""Unit tests for MCP tool routing helpers in loop.py."""

from __future__ import annotations

from aios.harness.loop import (
    _is_mcp_tool,
    _tc_name,
    resolve_mcp_permission,
)
from aios.models.agents import (
    McpPermissionPolicy,
    McpToolsetConfig,
    ToolSpec,
)
from aios.tools.registry import to_openai_tools


class TestTcName:
    def test_extracts_name(self) -> None:
        tc = {"function": {"name": "bash", "arguments": "{}"}}
        assert _tc_name(tc) == "bash"

    def test_missing_function(self) -> None:
        assert _tc_name({}) == ""

    def test_missing_name(self) -> None:
        assert _tc_name({"function": {}}) == ""


class TestIsMcpTool:
    def test_mcp_tool(self) -> None:
        assert _is_mcp_tool("mcp__github__create_issue") is True

    def test_builtin_tool(self) -> None:
        assert _is_mcp_tool("bash") is False

    def test_custom_tool(self) -> None:
        assert _is_mcp_tool("get_weather") is False

    def test_mcp_prefix_only(self) -> None:
        assert _is_mcp_tool("mcp__") is True


class TestResolveMcpPermission:
    def test_default_returns_none(self) -> None:
        """No mcp_toolset entry → None (callers treat as always_ask)."""
        tools = [ToolSpec(type="bash")]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) is None

    def test_always_allow_from_default_config(self) -> None:
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="github",
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                ),
            ),
        ]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) == "always_allow"

    def test_always_ask_from_default_config(self) -> None:
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="github",
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="always_ask"),
                ),
            ),
        ]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) == "always_ask"

    def test_no_default_config_returns_flat_permission(self) -> None:
        """Falls back to ToolSpec.permission when no default_config."""
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="github",
                permission="always_allow",
            ),
        ]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) == "always_allow"

    def test_no_config_no_permission_returns_none(self) -> None:
        """No default_config, no flat permission → None (always_ask)."""
        tools = [
            ToolSpec(type="mcp_toolset", mcp_server_name="github"),
        ]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) is None

    def test_wrong_server_not_matched(self) -> None:
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="slack",
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                ),
            ),
        ]
        assert resolve_mcp_permission("mcp__github__create_issue", tools) is None


class TestToOpenaiToolsSkipsMcpToolset:
    def test_mcp_toolset_skipped(self) -> None:
        """mcp_toolset entries produce no output — MCP tools come from discovery."""
        result = to_openai_tools(
            [
                ToolSpec(type="mcp_toolset", mcp_server_name="github"),
            ]
        )
        assert result == []

    def test_mcp_toolset_skipped_among_others(self) -> None:
        """mcp_toolset entries are skipped but other tools pass through."""
        import aios.tools  # noqa: F401

        result = to_openai_tools(
            [
                ToolSpec(type="bash"),
                ToolSpec(type="mcp_toolset", mcp_server_name="github"),
            ]
        )
        assert len(result) == 1
        assert result[0]["function"]["name"] == "bash"
