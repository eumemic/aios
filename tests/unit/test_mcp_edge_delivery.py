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


@pytest.mark.asyncio
async def test_same_url_session_claims_only_its_resolved_identity() -> None:
    url = "https://shared.example/mcp"
    breaker_pool = McpSessionPool()
    breaker_pool.mark_unhealthy(url, "vlt_owner", "", backoff_s=60)
    unrelated = _agent(
        mcp_servers=[McpServerSpec(name="other", url=url)],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="other")],
    )
    owner = _agent(
        mcp_servers=[McpServerSpec(name="owner", url=url, vault_id="vlt_owner")],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="owner")],
    )
    emitted: list[tuple[str, str]] = []

    async def resolve(_pool, _crypto, session_id, _spec, **_kwargs):
        return ("vlt_other" if session_id == "unrelated" else "vlt_owner"), {}

    async def discover(*_args, **_kwargs):
        return [], None

    async def append(_pool, session_id, _type, event, **_kwargs):
        emitted.append((session_id, event["server"]))
        return MagicMock()

    prior = runtime.mcp_session_pool
    runtime.mcp_session_pool = breaker_pool
    try:
        with (
            patch("aios.mcp.client.resolve_auth_for_mcp_mount", side_effect=resolve),
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
                pool=AsyncMock(), session_id="owner", agent=owner, account_id="acc_test_stub"
            )
    finally:
        runtime.mcp_session_pool = prior

    assert emitted == [("owner", "owner")]


@pytest.mark.asyncio
async def test_same_identity_edge_is_delivered_only_to_producer() -> None:
    url = "https://shared.example/mcp"
    breaker_pool = McpSessionPool()
    sibling = _agent(
        mcp_servers=[McpServerSpec(name="sibling_mount", url=url, vault_id="v")],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="sibling_mount")],
    )
    origin = _agent(
        mcp_servers=[McpServerSpec(name="origin_mount", url=url, vault_id="v")],
        tools=[ToolSpec(type="mcp_toolset", enabled=True, mcp_server_name="origin_mount")],
    )
    emitted = []
    failed = False

    async def discover(*args, **kwargs):
        nonlocal failed
        if args[3] == "origin_mount" and not failed:
            failed = True
            breaker_pool.mark_unhealthy(url, "v", "", backoff_s=60)
            raise TimeoutError("down")
        return [], None

    async def append(_pool, session_id, _type, event, **kwargs):
        emitted.append((session_id, event["server"]))
        return MagicMock()

    prior = runtime.mcp_session_pool
    runtime.mcp_session_pool = breaker_pool
    try:
        with (
            patch(
                "aios.mcp.client.resolve_auth_for_mcp_mount",
                new_callable=AsyncMock,
                return_value=("v", {}),
            ),
            patch("aios.mcp.client.discover_mcp_tools", side_effect=discover),
            patch("aios.harness.loop.sessions_service.append_event", side_effect=append),
        ):
            await discover_session_mcp_tools(
                pool=AsyncMock(), session_id="origin", agent=origin, account_id="acc_test_stub"
            )
            # Re-seed an attributed edge to replay the scheduling case where the
            # sibling prelude reaches draining before its producer.
            from aios.mcp.pool import degraded_edge_owner

            breaker_pool.mark_healthy(url, "v", "")
            with degraded_edge_owner("origin", "origin_mount"):
                breaker_pool.mark_unhealthy(url, "v", "", backoff_s=60)
            emitted.clear()
            await discover_session_mcp_tools(
                pool=AsyncMock(), session_id="sibling", agent=sibling, account_id="acc_test_stub"
            )
            await discover_session_mcp_tools(
                pool=AsyncMock(), session_id="origin", agent=origin, account_id="acc_test_stub"
            )
    finally:
        runtime.mcp_session_pool = prior
    assert emitted == [("origin", "origin_mount")]
