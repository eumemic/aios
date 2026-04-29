"""Unit tests for MCP tool routing helpers in loop.py."""

from __future__ import annotations

from datetime import datetime

from aios.harness.loop import (
    _hide_focal_channel_tools_when_phone_down,
    _is_mcp_tool,
    _tc_name,
    mcp_channel_context_by_server,
    resolve_mcp_permission,
)
from aios.models.agents import (
    McpChannelContext,
    McpPermissionPolicy,
    McpToolsetConfig,
    ToolSpec,
)
from aios.models.connections import Connection
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

    def test_focal_channel_toolset_defaults_to_always_allow(self) -> None:
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="signal",
                channel_context=McpChannelContext(type="focal"),
            ),
        ]
        assert resolve_mcp_permission("mcp__signal__signal_send", tools) == "always_allow"

    def test_explicit_permission_beats_focal_default(self) -> None:
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="signal",
                permission="always_ask",
                channel_context=McpChannelContext(type="focal"),
            ),
        ]
        assert resolve_mcp_permission("mcp__signal__signal_send", tools) == "always_ask"

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

    def test_legacy_connection_projection_can_default_to_always_allow(self) -> None:
        """Legacy connection-projected servers are authorized by the runtime
        with an explicit server-name set, not by a magic name prefix.
        """
        assert (
            resolve_mcp_permission(
                "mcp__conn_01HQR2K7VXBZ9MNPL3WYCT8F__send",
                [],
                always_allow_server_names={"conn_01HQR2K7VXBZ9MNPL3WYCT8F"},
            )
            == "always_allow"
        )

    def test_normal_toolset_policy_beats_legacy_projection(self) -> None:
        """If an agent declares a server/toolset, that normal MCP config wins."""
        tools = [
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="conn_foo",
                default_config=McpToolsetConfig(
                    permission_policy=McpPermissionPolicy(type="always_ask"),
                ),
            ),
        ]
        assert (
            resolve_mcp_permission(
                "mcp__conn_foo__send",
                tools,
                always_allow_server_names={"conn_foo"},
            )
            == "always_ask"
        )


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


class TestHideFocalChannelToolsWhenPhoneDown:
    """Focal-channel MCP tools disappear from the model's tool list when
    the session's focal_channel is NULL. Other MCP tools and built-ins stay
    visible.
    """

    def test_focal_set_keeps_all_tools(self) -> None:
        tools = [
            _tool("mcp__signal__signal_send"),
            _tool("mcp__github__create_issue"),
            _tool("bash"),
        ]
        result = _hide_focal_channel_tools_when_phone_down(
            tools, "signal/bot/alice", {"signal": "focal"}
        )
        assert result == tools

    def test_focal_null_hides_channel_aware_tools(self) -> None:
        tools = [
            _tool("mcp__signal__signal_send"),
            _tool("mcp__signal__signal_react"),
            _tool("mcp__github__create_issue"),
            _tool("bash"),
        ]
        result = _hide_focal_channel_tools_when_phone_down(tools, None, {"signal": "focal"})
        names = [t["function"]["name"] for t in result]  # type: ignore[index]
        assert "mcp__signal__signal_send" not in names
        assert "mcp__signal__signal_react" not in names
        # Agent-declared MCP and built-ins survive.
        assert "mcp__github__create_issue" in names
        assert "bash" in names

    def test_empty_list(self) -> None:
        assert _hide_focal_channel_tools_when_phone_down([], None, {}) == []
        assert _hide_focal_channel_tools_when_phone_down([], "signal/bot/alice", {}) == []

    def test_contexts_come_from_toolset_config(self) -> None:
        contexts = mcp_channel_context_by_server(
            [
                ToolSpec(
                    type="mcp_toolset",
                    mcp_server_name="signal",
                    channel_context=McpChannelContext(type="focal"),
                ),
                ToolSpec(type="mcp_toolset", mcp_server_name="github"),
            ]
        )
        assert contexts == {"signal": "focal"}

    def test_agent_server_url_suppresses_legacy_connection_context(self) -> None:
        now = datetime(2026, 4, 16)
        connection = Connection(
            id="conn_01HQR2K7VXBZ9MNPL3WYCT8F",
            connector="signal",
            account="acct",
            mcp_url="https://m1",
            vault_id="vlt_x",
            metadata={},
            created_at=now,
            updated_at=now,
        )

        contexts = mcp_channel_context_by_server(
            [
                ToolSpec(
                    type="mcp_toolset",
                    mcp_server_name="signal",
                    channel_context=McpChannelContext(type="focal"),
                ),
            ],
            [connection],
            agent_mcp_server_names={"signal"},
            agent_mcp_server_urls={"https://m1"},
        )
        assert contexts == {"signal": "focal"}

    def test_channel_only_connection_does_not_add_legacy_context(self) -> None:
        now = datetime(2026, 4, 16)
        connection = Connection(
            id="conn_01HQR2K7VXBZ9MNPL3WYCT8F",
            connector="signal",
            account="acct",
            mcp_url=None,
            vault_id=None,
            metadata={},
            created_at=now,
            updated_at=now,
        )

        contexts = mcp_channel_context_by_server([], [connection])
        assert contexts == {}
