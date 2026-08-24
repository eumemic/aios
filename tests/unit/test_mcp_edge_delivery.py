from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.harness.loop import discover_session_mcp_tools
from aios.mcp.pool import McpSessionPool
from aios.models.agents import McpServerSpec, ToolSpec
from tests.unit.test_discover_session_mcp_tools import _agent


@pytest.mark.asyncio
async def test_unrelated_prelude_cannot_consume_owners_down_edge() -> None:
    url = "https://shared.example/mcp"
    pool = McpSessionPool()
    pool.mark_unhealthy(url, "vlt_owner", "", backoff_s=60)
    unrelated = _agent(
        mcp_servers=[McpServerSpec(name="other", url="https://other/mcp")],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="other")],
    )
    owner = _agent(
        mcp_servers=[McpServerSpec(name="owner", url=url, vault_id="vlt_owner")],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="owner")],
    )
    emitted: list[dict[str, object]] = []

    async def discover(*_args: object, **_kwargs: object) -> tuple[list[dict[str, object]], None]:
        return [], None

    async def append(*args: object, **_kwargs: object) -> MagicMock:
        event = args[3]
        assert isinstance(event, dict)
        emitted.append(event)
        return MagicMock()

    prior = runtime.mcp_session_pool
    runtime.mcp_session_pool = pool
    try:
        with (
            patch(
                "aios.mcp.client.resolve_auth_for_mcp_mount",
                new_callable=AsyncMock,
                return_value=(None, {}),
            ),
            patch("aios.mcp.client.discover_mcp_tools", side_effect=discover),
            patch("aios.harness.loop.sessions_service.append_event", side_effect=append),
        ):
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="unrelated",
                agent=unrelated,
                account_id="acc_test_stub",
            )
            await discover_session_mcp_tools(
                pool=AsyncMock(),
                session_id="owner",
                agent=owner,
                account_id="acc_test_stub",
            )
    finally:
        runtime.mcp_session_pool = prior

    assert [
        event["server"] for event in emitted if event.get("event") == "mcp_server_unavailable"
    ] == ["owner"]
