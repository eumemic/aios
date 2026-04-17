"""Unit tests for MCP tool routing helpers in loop.py."""

from __future__ import annotations

from aios.harness.loop import (
    _hide_conn_tools_when_phone_down,
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

    def test_connection_provided_defaults_to_always_allow(self) -> None:
        """Tools from connection-provided MCP servers (names with the
        reserved ``conn_`` prefix) don't require per-call confirmation:
        the session's channel binding is the explicit routing consent.
        """
        # No agent-declared mcp_toolset entry references the connection —
        # the reserved-prefix validator forbids it — so normally we'd
        # fall through to None (= always_ask).  Connection servers are
        # the exception.
        assert (
            resolve_mcp_permission("mcp__conn_01HQR2K7VXBZ9MNPL3WYCT8F__send", []) == "always_allow"
        )

    def test_connection_provided_ignores_agent_overrides(self) -> None:
        """Even if someone (somehow) had a matching mcp_toolset entry, the
        connection branch wins — agent-declared tools can't target
        connection-derived servers by name."""
        # The validator prevents this from being constructible via API, but
        # test behaviour directly in case of bypass (e.g., legacy DB rows).
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="conn_foo",
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="always_ask"),
                ),
            ),
        ]
        assert resolve_mcp_permission("mcp__conn_foo__send", tools) == "always_allow"


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


def _tool(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name}}


class TestHideConnToolsWhenPhoneDown:
    """Slice 6: connection-provided MCP tools disappear from the model's
    tool list when the session's focal_channel is NULL.  Agent-declared
    MCP tools and built-ins stay visible.
    """

    def test_focal_set_keeps_all_tools(self) -> None:
        tools = [
            _tool("mcp__conn_abc__signal_send"),
            _tool("mcp__github__create_issue"),
            _tool("bash"),
        ]
        result = _hide_conn_tools_when_phone_down(tools, "signal/bot/alice")
        assert result == tools

    def test_focal_null_hides_conn_tools(self) -> None:
        tools = [
            _tool("mcp__conn_abc__signal_send"),
            _tool("mcp__conn_abc__signal_react"),
            _tool("mcp__github__create_issue"),
            _tool("bash"),
        ]
        result = _hide_conn_tools_when_phone_down(tools, None)
        names = [t["function"]["name"] for t in result]  # type: ignore[index]
        assert "mcp__conn_abc__signal_send" not in names
        assert "mcp__conn_abc__signal_react" not in names
        # Agent-declared MCP and built-ins survive.
        assert "mcp__github__create_issue" in names
        assert "bash" in names

    def test_empty_list(self) -> None:
        assert _hide_conn_tools_when_phone_down([], None) == []
        assert _hide_conn_tools_when_phone_down([], "signal/bot/alice") == []
