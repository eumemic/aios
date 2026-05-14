"""Tests for the ``mcp`` CLI system-prompt hint predicate.

The hint is rendered into the system prompt whenever the agent has at
least one ``always_allow`` MCP toolset entry. Emitting it for an agent
whose toolset has only ``always_ask`` policies would lie to the model
(every CLI call would 403), so the predicate has to walk both
``default_config`` and per-tool ``configs`` overrides.
"""

from __future__ import annotations

from aios.harness.step_context import _has_always_allow_mcp_tool
from aios.models.agents import (
    McpPermissionPolicy,
    McpToolConfig,
    McpToolsetConfig,
    ToolSpec,
)


def _ts(
    *,
    enabled: bool = True,
    default_policy: str | None = "always_allow",
    configs: list[McpToolConfig] | None = None,
) -> ToolSpec:
    return ToolSpec(
        type="mcp_toolset",
        enabled=enabled,
        mcp_server_name="s",
        default_config=McpToolsetConfig(
            enabled=True,
            permission_policy=(
                McpPermissionPolicy(type=default_policy) if default_policy else None
            ),
        ),
        configs=configs,
    )


class TestHasAlwaysAllowMcpTool:
    def test_empty_tools(self) -> None:
        assert _has_always_allow_mcp_tool([]) is False

    def test_only_builtin_tool(self) -> None:
        assert _has_always_allow_mcp_tool([ToolSpec(type="bash")]) is False

    def test_disabled_toolset_does_not_count(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(enabled=False)]) is False

    def test_default_always_allow(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(default_policy="always_allow")]) is True

    def test_default_always_ask(self) -> None:
        assert _has_always_allow_mcp_tool([_ts(default_policy="always_ask")]) is False

    def test_per_tool_override_to_always_allow(self) -> None:
        spec = _ts(
            default_policy="always_ask",
            configs=[
                McpToolConfig(name="x", permission_policy=McpPermissionPolicy(type="always_allow"))
            ],
        )
        assert _has_always_allow_mcp_tool([spec]) is True

    def test_per_tool_override_disabled(self) -> None:
        # Disabled per-tool overrides shouldn't count even if their policy
        # is always_allow.
        spec = _ts(
            default_policy="always_ask",
            configs=[
                McpToolConfig(
                    name="x",
                    enabled=False,
                    permission_policy=McpPermissionPolicy(type="always_allow"),
                )
            ],
        )
        assert _has_always_allow_mcp_tool([spec]) is False
