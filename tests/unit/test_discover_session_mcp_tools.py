"""Unit tests for discover_session_mcp_tools.

Covers the collect-URLs-then-discover shape: agent-declared MCP
(filtered by enabled mcp_toolset entries) unioned with
connection-provided MCP (derived from session bindings → connections).
The MCP SDK and auth lookup are both mocked; only the orchestration
is under test here.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.agents import McpServerSpec, ToolSpec
from aios.models.connections import Connection


@pytest.fixture(autouse=True)
def _mock_crypto_box() -> Any:
    """Bypass the runtime crypto-box requirement — discovery doesn't
    actually decrypt anything when resolve_auth_for_url is mocked.
    """
    with patch("aios.harness.loop.runtime.require_crypto_box") as m:
        m.return_value = object()
        yield m


def _connection(cid: str, url: str) -> Connection:
    now = datetime(2026, 4, 16)
    return Connection(
        id=cid,
        connector="signal",
        account="acct",
        mcp_url=url,
        vault_id="vlt_x",
        metadata={},
        created_at=now,
        updated_at=now,
    )


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
            connections=[],
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
                connections=[],
            )
        names = {t["name"] for t in tools}
        assert names == {"mcp__gh__t"}

    async def test_connections_only(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        connections = [
            _connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F", "https://m1"),
            _connection("conn_01HQR2K7VXBZ9MNPL3WYCT8G", "https://m2"),
        ]

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
                agent=_agent(),
                connections=connections,
            )
        # Each connection contributes one tool, namespaced with its id.
        urls = {t["url"] for t in tools}
        assert urls == {"https://m1", "https://m2"}
        names = {t["name"] for t in tools}
        assert names == {
            "mcp__conn_01HQR2K7VXBZ9MNPL3WYCT8F__t",
            "mcp__conn_01HQR2K7VXBZ9MNPL3WYCT8G__t",
        }

    async def test_agent_and_connections_combined(self) -> None:
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )
        connections = [_connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F", "https://m1")]

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
                connections=connections,
            )
        urls = sorted(t["url"] for t in tools)
        assert urls == ["https://m1", "https://mcp.github"]

    async def test_auth_resolved_per_url(self) -> None:
        """Each URL resolves auth independently — goes through
        resolve_auth_for_url once per server, not once per batch.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )
        connections = [_connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F", "https://m1")]

        seen_urls: list[str] = []

        async def _fake_resolve(_pool: Any, _cb: Any, _sid: str, url: str) -> dict[str, str]:
            seen_urls.append(url)
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
                connections=connections,
            )
        assert sorted(seen_urls) == ["https://m1", "https://mcp.github"]
        auths = {t["auth"] for t in tools}
        assert auths == {
            "Bearer token-for-https://mcp.github",
            "Bearer token-for-https://m1",
        }

    async def test_instructions_keyed_by_server_name(self) -> None:
        """Each server's ``InitializeResult.instructions`` flows into the
        returned dict under its server_name key.  Servers that supply no
        instructions are omitted — the harness uses dict membership as
        the trigger for rendering a per-connector affordance block.
        """
        from aios.harness.loop import discover_session_mcp_tools

        agent = _agent(
            mcp_servers=[McpServerSpec(name="gh", url="https://mcp.github")],
            tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="gh")],
        )
        connections = [_connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F", "https://m1")]

        async def _discover(
            url: str, name: str, _headers: dict[str, str]
        ) -> tuple[list[dict[str, Any]], str | None]:
            # Agent server supplies nothing; connection server supplies prose.
            if name == "gh":
                return [], None
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
                connections=connections,
            )
        # 'gh' returned None → omitted; connection returned prose → present.
        assert instructions == {"conn_01HQR2K7VXBZ9MNPL3WYCT8F": "## signal\n\nbe nice"}

    async def test_empty_string_instructions_omitted(self) -> None:
        """An empty string is treated identically to ``None`` — no
        affordance block should be rendered for a server that returned
        ``""``.
        """
        from aios.harness.loop import discover_session_mcp_tools

        connections = [_connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F", "https://m1")]

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
                agent=_agent(),
                connections=connections,
            )
        assert instructions == {}
