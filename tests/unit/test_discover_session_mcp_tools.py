"""Unit tests for discover_session_mcp_tools.

Covers the collect-URLs-then-discover shape: agent-declared MCP filtered by
enabled mcp_toolset entries. MCP instructions are returned under normal
agent MCP server names.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.agents import McpChannelContext, McpServerSpec, ToolSpec


@pytest.fixture(autouse=True)
def _mock_crypto_box() -> Any:
    """Bypass the runtime crypto-box requirement — discovery doesn't
    actually decrypt anything when resolve_auth_for_url is mocked.
    """
    with patch("aios.harness.loop.runtime.require_crypto_box") as m:
        m.return_value = object()
        yield m


def _agent(
    mcp_servers: list[McpServerSpec] | None = None,
    tools: list[ToolSpec] | None = None,
) -> Any:
    return SimpleNamespace(
        mcp_servers=mcp_servers or [],
        tools=tools or [],
    )


class TestDiscoverSessionMcpTools:
    async def test_no_sources_returns_empty(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        tools, instructions = await discover_session_mcp_tools(
            pool=AsyncMock(),
            session_id="sess_x",
            agent=_agent(),
        )
        assert tools == []
        assert instructions == {}

    async def test_agent_only_enabled_server(self) -> None:
        """Only enabled mcp_toolset entries produce discovery; disabled
        ones are skipped.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[
                McpServerSpec(name="gh", url="https://mcp.github"),
                McpServerSpec(name="off", url="https://mcp.off"),
            ],
            tools=[
                ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh"),
                ToolSpec(type="mcp_toolset", enabled=False, mcp_server_name="off"),
            ],
        )

        async def _discover(
            url: str, name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "url": url}], None

        with (
            patch("aios.mcp.client.resolve_auth_for_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = {}
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )
        names = {t["name"] for t in tools}
        assert names == {"mcp__gh__t"}

    async def test_discovers_only_agent_servers(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )

        async def _discover(
            url: str, name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "url": url}], None

        with (
            patch("aios.mcp.client.resolve_auth_for_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = {}
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )
        urls = sorted(t["url"] for t in tools)
        assert urls == ["https://mcp.github"]

    async def test_focal_server_instructions_are_not_connection_aliased(
        self,
    ) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="signal", url="https://m1")],
            tools=[
                ToolSpec(
                    type="mcp_toolset",
                    enabled=True,
                    mcp_server_name="signal",
                    channel_context=McpChannelContext(type="focal"),
                )
            ],
        )

        async def _discover(
            url: str, name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "url": url}], "send via focal"

        with (
            patch("aios.mcp.client.resolve_auth_for_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = {}
            tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )

        assert tools == [{"name": "mcp__signal__t", "url": "https://m1"}]
        assert instructions == {"signal": "send via focal"}

    async def test_auth_resolved_per_url(self) -> None:
        """Each URL resolves auth independently — goes through
        resolve_auth_for_url once per server, not once per batch.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )

        seen: list[str] = []

        async def _fake_resolve(
            _pool: Any,
            _cb: Any,
            _sid: str,
            url: str,
        ) -> dict[str, str]:
            seen.append(url)
            return {"Authorization": f"Bearer token-for-{url}"}

        async def _discover(
            url: str, name: str, headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [{"name": f"mcp__{name}__t", "auth": headers["Authorization"]}], None

        with (
            patch("aios.mcp.client.resolve_auth_for_url", side_effect=_fake_resolve),
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            tools, _instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )
        assert seen == ["https://mcp.github"]
        assert {t["auth"] for t in tools} == {"Bearer token-for-https://mcp.github"}

    async def test_instructions_keyed_by_server_name(self) -> None:
        """Each server's ``InitializeResult.instructions`` flows into the
        returned dict under its server_name key.  Servers that supply no
        instructions are omitted — the harness uses dict membership as
        the trigger for rendering MCP server affordance prose.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="signal", url="https://mcp.signal")],
            tools=[
                ToolSpec(
                    type="mcp_toolset",
                    enabled=True,
                    mcp_server_name="signal",
                    channel_context=McpChannelContext(type="focal"),
                )
            ],
        )

        async def _discover(
            url: str, name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [], "## signal\n\nbe nice"

        with (
            patch("aios.mcp.client.resolve_auth_for_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = {}
            _tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )
        assert instructions == {"signal": "## signal\n\nbe nice"}

    async def test_empty_string_instructions_omitted(self) -> None:
        """An empty string is treated identically to ``None`` — no
        affordance block should be rendered for a server that returned
        ``""``.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="signal", url="https://mcp.signal")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="signal")],
        )

        async def _discover(
            _url: str, _name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            return [], ""

        with (
            patch("aios.mcp.client.resolve_auth_for_url", new_callable=AsyncMock) as resolve,
            patch("aios.mcp.client.discover_mcp_tools", side_effect=_discover),
        ):
            resolve.return_value = {}
            _tools, instructions = await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="sess_x",
                agent=agent,
            )
        assert instructions == {}
