"""Unit tests for ``AIOS_DEFAULT_MCP_PERMISSION_POLICY``.

The setting overrides the harness's hardcoded ``None → always_ask``
fallback when an MCP-namespaced tool has no matching ``mcp_toolset``
on the agent.

Default (unset): preserves current behaviour — unmounted toolsets gate
on confirmation.

When set to ``always_allow``: trusted-environment ergonomics — tools
mounted by attached connectors dispatch immediately without requiring
the agent author to declare an ``mcp_toolset`` per connector.

Explicit per-toolset policies still win — this only changes the
fallback for unmounted/unspecified servers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aios.harness import loop, runtime
from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolsetConfig,
    ToolSpec,
)


def _agent(*tools: ToolSpec, server_names: tuple[str, ...] = ()) -> MagicMock:
    agent = MagicMock()
    agent.mcp_servers = [
        McpServerSpec(name=name, url="https://example.com/mcp", type="url") for name in server_names
    ]
    agent.tools = list(tools)
    return agent


class TestDefaultMcpPolicy:
    """``_classify_tool_call`` consults the env-level fallback policy."""

    @pytest.fixture(autouse=True)
    def _stub_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = MagicMock()
        registry.states_for_connector.return_value = [MagicMock()]
        monkeypatch.setattr(runtime, "connector_subprocess_registry", registry)

    def test_unset_default_falls_back_to_always_ask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unmounted MCP toolset with no env override gates on confirmation."""
        from aios.config import Settings, get_settings

        monkeypatch.setattr(
            "aios.config.get_settings",
            lambda: Settings(default_mcp_permission_policy=None),
        )
        get_settings.cache_clear()

        agent = _agent()  # no mcp_toolset
        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc1",
                "function": {"name": "mcp__telegram__telegram_send", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map={},
        )
        assert kind == "needs_confirm"

    def test_env_default_always_allow_skips_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AIOS_DEFAULT_MCP_PERMISSION_POLICY=always_allow`` lets unmounted
        toolsets dispatch immediately."""
        from aios.config import Settings

        monkeypatch.setattr(
            "aios.config.get_settings",
            lambda: Settings(default_mcp_permission_policy="always_allow"),
        )

        agent = _agent()  # no mcp_toolset
        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc2",
                "function": {"name": "mcp__telegram__telegram_send", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map={},
        )
        assert kind == "mcp_immediate"

    def test_explicit_toolset_policy_overrides_env_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the agent declares an mcp_toolset with ``always_ask``, it wins
        over the env-level default."""
        from aios.config import Settings

        monkeypatch.setattr(
            "aios.config.get_settings",
            lambda: Settings(default_mcp_permission_policy="always_allow"),
        )

        agent = _agent(
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name="telegram",
                enabled=True,
                default_config=McpToolsetConfig(
                    enabled=True,
                    permission_policy=McpPermissionPolicy(type="always_ask"),
                ),
            ),
        )
        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc3",
                "function": {"name": "mcp__telegram__telegram_send", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map={},
        )
        assert kind == "needs_confirm"
