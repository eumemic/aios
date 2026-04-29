"""Unit tests for MCP-related models — McpServerSpec, ToolSpec mcp_toolset, serialization."""

from __future__ import annotations

import json

import pytest

from aios.models.agents import (
    AgentCreate,
    McpChannelContext,
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)


class TestMcpServerSpec:
    def test_basic_creation(self) -> None:
        spec = McpServerSpec(name="github", url="https://mcp.github.com/")
        assert spec.type == "url"
        assert spec.name == "github"
        assert spec.url == "https://mcp.github.com/"

    def test_type_defaults_to_url(self) -> None:
        spec = McpServerSpec(name="test", url="https://example.com")
        assert spec.type == "url"

    def test_name_required(self) -> None:
        with pytest.raises(ValueError):
            McpServerSpec(name="", url="https://example.com")

    def test_url_required(self) -> None:
        with pytest.raises(ValueError):
            McpServerSpec(name="test", url="")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            McpServerSpec(name="test", url="https://example.com", token="secret")  # type: ignore[call-arg]

    def test_round_trip(self) -> None:
        spec = McpServerSpec(name="github", url="https://mcp.github.com/")
        d = spec.model_dump()
        restored = McpServerSpec.model_validate(d)
        assert restored == spec

    def test_json_round_trip(self) -> None:
        spec = McpServerSpec(name="slack", url="https://mcp.slack.com/mcp")
        j = json.dumps(spec.model_dump())
        restored = McpServerSpec.model_validate_json(j)
        assert restored.name == "slack"

    def test_conn_prefix_is_ordinary_name(self) -> None:
        """The conn_ prefix is not a reserved server-name namespace."""
        assert McpServerSpec(name="conn_github", url="https://m").name == "conn_github"
        assert McpServerSpec(name="connector", url="https://m").name == "connector"
        assert McpServerSpec(name="my_conn", url="https://m").name == "my_conn"
        assert McpServerSpec(name="conn", url="https://m").name == "conn"


class TestMcpToolsetConfig:
    def test_defaults(self) -> None:
        cfg = McpToolsetConfig()
        assert cfg.enabled is True
        assert cfg.permission_policy is None

    def test_with_permission(self) -> None:
        cfg = McpToolsetConfig(
            permission_policy=McpPermissionPolicy(type="always_allow"),
        )
        assert cfg.permission_policy is not None
        assert cfg.permission_policy.type == "always_allow"


class TestMcpToolConfig:
    def test_per_tool_override(self) -> None:
        cfg = McpToolConfig(
            name="create_issue",
            permission_policy=McpPermissionPolicy(type="always_allow"),
        )
        assert cfg.name == "create_issue"
        assert cfg.enabled is True


class TestMcpChannelContext:
    def test_focal_context_default(self) -> None:
        ctx = McpChannelContext()
        assert ctx.type == "focal"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            McpChannelContext(type="focal", path="chat")  # type: ignore[call-arg]


class TestToolSpecMcpToolset:
    def test_basic_mcp_toolset(self) -> None:
        spec = ToolSpec(type="mcp_toolset", mcp_server_name="github")
        assert spec.type == "mcp_toolset"
        assert spec.mcp_server_name == "github"

    def test_mcp_toolset_requires_server_name(self) -> None:
        with pytest.raises(ValueError, match="mcp_toolset requires mcp_server_name"):
            ToolSpec(type="mcp_toolset")

    def test_mcp_toolset_with_default_config(self) -> None:
        spec = ToolSpec(
            type="mcp_toolset",
            mcp_server_name="github",
            default_config=McpToolsetConfig(
                permission_policy=McpPermissionPolicy(type="always_ask"),
            ),
        )
        assert spec.default_config is not None
        assert spec.default_config.permission_policy is not None
        assert spec.default_config.permission_policy.type == "always_ask"

    def test_mcp_toolset_with_configs(self) -> None:
        spec = ToolSpec(
            type="mcp_toolset",
            mcp_server_name="github",
            configs=[
                McpToolConfig(
                    name="create_issue",
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                ),
            ],
        )
        assert spec.configs is not None
        assert len(spec.configs) == 1
        assert spec.configs[0].name == "create_issue"

    def test_mcp_toolset_with_channel_context(self) -> None:
        spec = ToolSpec(
            type="mcp_toolset",
            mcp_server_name="signal",
            channel_context=McpChannelContext(type="focal"),
        )
        assert spec.channel_context is not None
        assert spec.channel_context.type == "focal"

    def test_channel_context_only_allowed_on_mcp_toolset(self) -> None:
        with pytest.raises(ValueError, match="channel_context"):
            ToolSpec(type="bash", channel_context=McpChannelContext(type="focal"))

    def test_mcp_toolset_round_trip(self) -> None:
        spec = ToolSpec(
            type="mcp_toolset",
            mcp_server_name="github",
            default_config=McpToolsetConfig(
                permission_policy=McpPermissionPolicy(type="always_ask"),
            ),
        )
        d = spec.model_dump()
        restored = ToolSpec.model_validate(d)
        assert restored.type == "mcp_toolset"
        assert restored.mcp_server_name == "github"
        assert restored.default_config is not None

    def test_mcp_toolset_json_round_trip(self) -> None:
        """Simulates JSONB storage and retrieval."""
        spec = ToolSpec(type="mcp_toolset", mcp_server_name="slack")
        j = json.dumps(spec.model_dump())
        restored = ToolSpec.model_validate_json(j)
        assert restored.type == "mcp_toolset"
        assert restored.mcp_server_name == "slack"


class TestAgentCreateWithMcpServers:
    def test_empty_mcp_servers_default(self) -> None:
        body = AgentCreate(name="test", model="gpt-4")
        assert body.mcp_servers == []

    def test_mcp_servers_accepted(self) -> None:
        body = AgentCreate(
            name="test",
            model="gpt-4",
            mcp_servers=[
                McpServerSpec(name="github", url="https://mcp.github.com/"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", mcp_server_name="github"),
            ],
        )
        assert len(body.mcp_servers) == 1
        assert body.mcp_servers[0].name == "github"


class TestBackwardCompat:
    """Old JSONB rows without mcp_servers or mcp_toolset entries."""

    def test_old_tools_list_no_mcp_fields(self) -> None:
        raw = [{"type": "bash"}, {"type": "read"}]
        specs = [ToolSpec.model_validate(t) for t in raw]
        assert all(s.mcp_server_name is None for s in specs)
        assert all(s.default_config is None for s in specs)
        assert all(s.configs is None for s in specs)
        assert all(s.channel_context is None for s in specs)
