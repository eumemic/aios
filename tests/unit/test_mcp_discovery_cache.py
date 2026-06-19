"""Unit tests for MCP discovery caching + per-server timeout / circuit breaker (#1391).

A single unresponsive MCP server used to stall the whole turn prelude: discovery
re-polled ``list_tools()`` live every step (no result cache) with no per-server
timeout, so one hung server starved the turn until the 960s step budget fired.

These tests cover the fix:

* the pool's ``(tools, instructions)`` result cache, keyed on transport identity
  + binding identity, so a static tool set is paid once per binding (no per-step
  RPC) and re-discovered only on a binding-identity bump or ``list_changed``;
* the per-server discovery timeout (``_DISCOVERY_TIMEOUT_S``) that bounds
  ``list_tools()``;
* the per-key circuit breaker that skips a recently-failed server fast.

All MCP SDK + httpx interactions are mocked. No network calls.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.mcp.client import (
    _DISCOVERY_TIMEOUT_S,
    _headers_key,
    discover_mcp_tools,
)
from aios.mcp.pool import McpSessionPool

URL = "https://m.example/"
EMPTY_KEY = _headers_key(None)


def _make_mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = f"desc {name}"
    tool.inputSchema = {"type": "object"}
    return tool


def _make_mock_session(tool_names: list[str], instructions: str | None = None) -> AsyncMock:
    session = AsyncMock()
    init = MagicMock()
    init.instructions = instructions
    session.initialize = AsyncMock(return_value=init)
    result = MagicMock()
    result.tools = [_make_mock_tool(n) for n in tool_names]
    session.list_tools = AsyncMock(return_value=result)
    return session


def _transport_mock() -> MagicMock:
    t = MagicMock()
    t.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    t.__aexit__ = AsyncMock(return_value=False)
    return t


def _session_ctx(session: AsyncMock) -> MagicMock:
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=session)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


@pytest.fixture
def pool_runtime() -> Any:
    """Install a fresh ``McpSessionPool`` on ``runtime`` for the test, restoring after."""
    prior = runtime.mcp_session_pool
    pool = McpSessionPool()
    runtime.mcp_session_pool = pool
    try:
        yield pool
    finally:
        runtime.mcp_session_pool = prior


# ── result cache ────────────────────────────────────────────────────────────


class TestDiscoveryResultCache:
    async def test_second_discovery_same_binding_skips_list_tools_rpc(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: with caching on, the SECOND discovery for the same
        binding identity serves from cache and issues NO ``list_tools()`` RPC —
        removing it from the per-step hot path."""
        session = _make_mock_session(["create_issue"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            tools1, instr1 = await discover_mcp_tools(URL, "v", {}, "github", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1
            tools2, instr2 = await discover_mcp_tools(URL, "v", {}, "github", binding_id="agt_1:3")

        # Cache hit: no further RPC, identical result.
        assert session.list_tools.await_count == 1
        assert tools2 == tools1
        assert instr2 == instr1
        assert tools1[0]["function"]["name"] == "mcp__github__create_issue"

    async def test_binding_version_bump_rediscovers(self, pool_runtime: McpSessionPool) -> None:
        """Acceptance: a binding-identity bump (agent-version change) re-pays
        discovery — the new tool set propagates with no staleness."""
        session = _make_mock_session(["tool_v3"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1
            # New version → fresh cache key → re-discovers.
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:4")

        assert session.list_tools.await_count == 2

    async def test_no_binding_id_never_caches(self, pool_runtime: McpSessionPool) -> None:
        """The API/test path (``binding_id=None``) is never cached — every call
        re-polls (preserving legacy uncached behaviour for the broker-less path)."""
        session = _make_mock_session(["t"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s")
            await discover_mcp_tools(URL, "v", {}, "s")

        assert session.list_tools.await_count == 2

    async def test_list_changed_notification_invalidates_cache(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: the MCP ``tools/list_changed`` notification invalidates
        the cache so the next discovery re-polls (no staleness regression)."""
        session = _make_mock_session(["t"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1

            # Simulate the server pushing a tools/list_changed notification by
            # invoking the same invalidation the registered handler performs.
            pool_runtime._invalidate_tools_for_pool_key((URL, "v", EMPTY_KEY))

            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")

        assert session.list_tools.await_count == 2


# ── per-server discovery timeout ────────────────────────────────────────────


class TestDiscoveryTimeout:
    async def test_slow_list_tools_times_out_and_marks_unhealthy(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: a hung ``list_tools()`` is bounded by the per-server
        timeout (does not hang forever) and the key is marked unhealthy."""

        async def _never_returns() -> Any:
            await asyncio.sleep(3600)

        session = _make_mock_session(["t"])
        session.list_tools = AsyncMock(side_effect=_never_returns)

        # Speed: patch the timeout to a tiny value so the test is fast.
        with (
            patch("aios.mcp.client._DISCOVERY_TIMEOUT_S", 0.05),
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
            pytest.raises(TimeoutError),
        ):
            await asyncio.wait_for(
                discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3"),
                timeout=5.0,
            )

        # The key is now in backoff so the next prelude can skip it fast.
        assert pool_runtime.is_unhealthy(URL, "v", EMPTY_KEY)

    async def test_default_discovery_timeout_is_finite_and_tight(self) -> None:
        """The discovery timeout exists and is well under the 960s step budget —
        the precondition that prevents a prelude stall (#1391)."""
        assert 0 < _DISCOVERY_TIMEOUT_S < 960


# ── circuit breaker bookkeeping ─────────────────────────────────────────────


class TestCircuitBreaker:
    def test_mark_unhealthy_then_healthy(self) -> None:
        pool = McpSessionPool()
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=100.0)
        pool.mark_unhealthy(URL, "v", EMPTY_KEY, backoff_s=60.0, now=100.0)
        # Inside the window → unhealthy.
        assert pool.is_unhealthy(URL, "v", EMPTY_KEY, now=130.0)
        # After the window → healthy again (and the entry is reaped).
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=200.0)
        # A successful discovery clears the breaker explicitly.
        pool.mark_unhealthy(URL, "v", EMPTY_KEY, backoff_s=60.0, now=300.0)
        pool.mark_healthy(URL, "v", EMPTY_KEY)
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=310.0)

    def test_cache_get_set_and_invalidate(self) -> None:
        pool = McpSessionPool()
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None
        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], "instr")
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") == ([{"x": 1}], "instr")
        # Different binding → independent entry.
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4") is None
        # list_changed drops every binding for the key.
        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4", [{"y": 2}], None)
        pool._invalidate_tools_for_pool_key((URL, "v", EMPTY_KEY))
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4") is None

    async def test_list_changed_handler_invalidates_only_on_tool_change(self) -> None:
        """The ``ClientSession`` message handler the pool registers (#1391)
        drops the cache on ``tools/list_changed`` but ignores other server
        notifications — so an unrelated notification can't thrash the cache."""
        import mcp.types as t

        pool = McpSessionPool()
        key = (URL, "v", EMPTY_KEY)
        handler = pool._make_list_changed_handler(key)

        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], None)
        await handler(t.ServerNotification(t.ToolListChangedNotification()))
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None

        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], None)
        await handler(
            t.ServerNotification(
                t.LoggingMessageNotification(
                    params=t.LoggingMessageNotificationParams(level="info", data="hi")
                )
            )
        )
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is not None

    def test_invalidate_by_vault(self) -> None:
        pool = McpSessionPool()
        pool.set_cached_tools(URL, "vlt_1", EMPTY_KEY, "agt_1:3", [{"a": 1}], None)
        pool.set_cached_tools(URL, "vlt_2", EMPTY_KEY, "agt_1:3", [{"b": 2}], None)
        pool.invalidate_tools_by_vault("vlt_1")
        assert pool.get_cached_tools(URL, "vlt_1", EMPTY_KEY, "agt_1:3") is None
        # Other vaults' entries survive.
        assert pool.get_cached_tools(URL, "vlt_2", EMPTY_KEY, "agt_1:3") is not None
