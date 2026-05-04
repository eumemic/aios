"""Unit test for unknown-MCP-tool handling.

The bug: the harness pattern-matches ``mcp__<server>__<tool>`` then
calls ``resolve_mcp_permission`` to choose a gate, never checking that
the named server is actually registered (as a connector instance OR as
one of the agent's HTTP MCP servers).  When the model hallucinates an
old or made-up MCP tool name, the harness:

  1. Returns ``None`` from ``resolve_mcp_permission`` (no toolset match).
  2. Caller defaults to ``always_ask``.
  3. Session parks in ``requires_action`` waiting for a confirmation
     that, if granted, would dispatch into ``mcp_tool.server_not_found``
     anyway.

Demonstrated live during PR #213 testing: the model parroted
``mcp__conn_<id>__signal_send`` from prior conversation context (the
pre-PR4 namespace) — the ``conn_<id>`` server doesn't exist anymore,
but the harness gated the call instead of erroring.

This test pins the contract: an MCP tool whose server is not known
must be classified ``unknown_mcp`` (immediate tool-error route), not
``needs_confirm``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aios.harness import loop, runtime
from aios.models.agents import McpServerSpec, ToolSpec


def _agent_with_mcp_toolset(server_name: str) -> MagicMock:
    """Build a minimal agent stand-in with one mcp_toolset entry."""
    agent = MagicMock()
    agent.mcp_servers = [McpServerSpec(name=server_name, url="https://example.com/mcp", type="url")]
    agent.tools = [
        ToolSpec(
            type="mcp_toolset",
            mcp_server_name=server_name,
            enabled=True,
        )
    ]
    return agent


class TestClassifyMcpTool:
    """``loop._classify_tool_call`` must reject unknown-server MCP names."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default to no connector registry (HTTP-only path)."""
        monkeypatch.setattr(runtime, "connector_subprocess_registry", None)

    def test_unknown_server_classified_as_unknown_mcp(self) -> None:
        """``mcp__nonexistent__foo`` against an agent with no matching server
        must classify as ``unknown_mcp`` so the harness emits an immediate
        tool error instead of gating on confirmation.
        """
        agent = _agent_with_mcp_toolset("github")
        # mcp_server_map mirrors what the dispatch path consults: name → URL
        # for HTTP MCP servers the agent declared.
        mcp_server_map = {"github": "https://api.githubcopilot.com/mcp/"}

        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc1",
                "function": {"name": "mcp__nonexistent__foo", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map=mcp_server_map,
        )
        assert kind == "unknown_mcp"

    def test_known_http_mcp_server_classified_normally(self) -> None:
        """Known HTTP MCP server falls into the existing buckets."""
        agent = _agent_with_mcp_toolset("github")
        mcp_server_map = {"github": "https://api.githubcopilot.com/mcp/"}

        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc2",
                "function": {"name": "mcp__github__create_issue", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map=mcp_server_map,
        )
        assert kind in {"mcp_immediate", "needs_confirm"}

    def test_known_connector_classified_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connector-registered server (no entry in mcp_server_map) is known."""
        agent = _agent_with_mcp_toolset("telegram")
        mcp_server_map: dict[str, str] = {}

        registry = MagicMock()
        registry.states_for_connector.return_value = [MagicMock()]
        monkeypatch.setattr(runtime, "connector_subprocess_registry", registry)

        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc3",
                "function": {"name": "mcp__telegram__telegram_send", "arguments": "{}"},
            },
            agent=agent,
            mcp_server_map=mcp_server_map,
        )
        assert kind in {"mcp_immediate", "needs_confirm"}

    def test_old_pre_pr4_conn_id_namespace_is_unknown(self) -> None:
        """Pre-PR4 ``mcp__conn_<id>__<tool>`` names hallucinated by models
        with stale conversation context must error immediately, not gate.
        """
        agent = _agent_with_mcp_toolset("github")
        mcp_server_map = {"github": "https://api.githubcopilot.com/mcp/"}

        kind = loop._classify_tool_call(
            tool_call={
                "id": "tc4",
                "function": {
                    "name": "mcp__conn_01KQS3BK4J7Y1ER372VJGBMCF8__signal_send",
                    "arguments": "{}",
                },
            },
            agent=agent,
            mcp_server_map=mcp_server_map,
        )
        assert kind == "unknown_mcp"
