"""Unit tests for the MCP session pool.

All MCP SDK and httpx interactions are mocked. No network calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.mcp.pool import McpSessionPool, _headers_hash

# ── _headers_hash ──────────────────────────────────────────────────────────


class TestHeadersHash:
    def test_same_headers_same_hash(self) -> None:
        h = {"Authorization": "Bearer tok", "X-Extra": "v"}
        assert _headers_hash(h) == _headers_hash(h)

    def test_order_independent(self) -> None:
        h1 = {"A": "1", "B": "2"}
        h2 = {"B": "2", "A": "1"}
        assert _headers_hash(h1) == _headers_hash(h2)

    def test_different_values_different_hash(self) -> None:
        assert _headers_hash({"Authorization": "Bearer tok1"}) != _headers_hash(
            {"Authorization": "Bearer tok2"}
        )

    def test_empty_headers(self) -> None:
        assert _headers_hash({}) == _headers_hash({})


# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_session() -> AsyncMock:
    """Build an AsyncMock that looks like a ``ClientSession``."""
    session = AsyncMock()
    session.initialize = AsyncMock(return_value=MagicMock())
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))
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


def _session_ctx_seq(sessions: list[AsyncMock]) -> MagicMock:
    it: Iterator[AsyncMock] = iter(sessions)

    async def _enter(*_args: Any, **_kwargs: Any) -> AsyncMock:
        return next(it)

    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(side_effect=_enter)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


# ── McpSessionPool ─────────────────────────────────────────────────────────


class TestMcpSessionPool:
    async def test_pool_hit_reuses_session(self) -> None:
        """Two calls with same key → only one ``initialize()`` happens."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            headers = {"Authorization": "Bearer tok"}
            s1, _ = await pool.get_or_connect("https://m.example/", headers)
            s2, _ = await pool.get_or_connect("https://m.example/", headers)

        assert s1 is s2
        session.initialize.assert_awaited_once()

    async def test_pool_miss_different_headers(self) -> None:
        """Different auth headers → two separate sessions."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            r1, _ = await pool.get_or_connect("https://m.example/", {"Authorization": "Bearer t1"})
            r2, _ = await pool.get_or_connect("https://m.example/", {"Authorization": "Bearer t2"})

        assert r1 is not r2
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1

    async def test_evict_causes_reconnect(self) -> None:
        """After eviction, next ``get_or_connect`` opens a new session."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        headers = {"Authorization": "Bearer tok"}
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            r1, _ = await pool.get_or_connect("https://m.example/", headers)
            pool.evict("https://m.example/", headers)
            r2, _ = await pool.get_or_connect("https://m.example/", headers)

        assert r1 is not r2

    async def test_thundering_herd_single_initialize(self) -> None:
        """N concurrent ``get_or_connect`` calls → exactly one ``initialize()``."""
        session = _make_mock_session()

        init_count = 0

        async def slow_init() -> MagicMock:
            nonlocal init_count
            init_count += 1
            # Yield to let other concurrent callers pile up on the lock.
            await asyncio.sleep(0)
            return MagicMock()

        session.initialize = AsyncMock(side_effect=slow_init)

        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            headers = {"Authorization": "Bearer tok"}
            results = await asyncio.gather(
                *[pool.get_or_connect("https://m.example/", headers) for _ in range(8)]
            )

        assert init_count == 1
        first = results[0][0]
        for session_got, _ in results:
            assert session_got is first

    async def test_close_all_closes_entries(self) -> None:
        """``close_all`` tears down every pooled entry and clears the map."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            await pool.get_or_connect("https://a.example/", {"Authorization": "A"})
            await pool.get_or_connect("https://b.example/", {"Authorization": "B"})

            close_calls: list[str] = []
            for key, entry in pool._entries.items():
                url = key[0]
                orig = entry.close

                async def _tracked(u: str = url, o: Any = orig) -> None:
                    close_calls.append(u)
                    await o()

                entry.close = _tracked  # type: ignore[method-assign]

            await pool.close_all()

        assert len(close_calls) == 2
        assert len(pool._entries) == 0

    async def test_close_all_empty_pool(self) -> None:
        """``close_all`` on an empty pool is a no-op."""
        pool = McpSessionPool()
        await pool.close_all()
        assert len(pool._entries) == 0


# ── Pool integration via discover_mcp_tools / call_mcp_tool ────────────────


@pytest.fixture
def restore_runtime_pool() -> Iterator[None]:
    """Save + restore ``runtime.mcp_session_pool`` across each test."""
    saved = runtime.mcp_session_pool
    try:
        yield
    finally:
        runtime.mcp_session_pool = saved


class TestDiscoverMcpToolsWithPool:
    async def test_uses_pool_when_set(self, restore_runtime_pool: None) -> None:
        """Pool path: ``initialize`` called once even across two discovers."""
        from aios.mcp.client import discover_mcp_tools

        session = _make_mock_session()
        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools("https://m.example/", "srv", {})
            await discover_mcp_tools("https://m.example/", "srv", {})

        session.initialize.assert_awaited_once()

    async def test_evicts_and_retries_on_list_tools_failure(
        self, restore_runtime_pool: None
    ) -> None:
        """First ``list_tools`` raises → pool evicts, second succeeds."""
        from aios.mcp.client import discover_mcp_tools

        s_a, s_b = _make_mock_session(), _make_mock_session()
        s_a.list_tools = AsyncMock(side_effect=RuntimeError("broken pipe"))
        s_b.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            tools, _ = await discover_mcp_tools("https://m.example/", "srv", {})

        assert tools == []
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1

    async def test_falls_back_when_pool_is_none(self, restore_runtime_pool: None) -> None:
        """When the pool is None, fresh-connection path is used."""
        from aios.mcp.client import discover_mcp_tools

        runtime.mcp_session_pool = None
        session = _make_mock_session()
        with (
            patch("aios.mcp.client.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.client.ClientSession", _session_ctx(session)),
        ):
            tools, _ = await discover_mcp_tools("https://m.example/", "srv", {})

        assert tools == []
        session.initialize.assert_awaited_once()


class TestCallMcpToolWithPool:
    async def test_uses_pool_when_set(self, restore_runtime_pool: None) -> None:
        """Pool path: ``initialize`` called once even across two tool calls."""
        from aios.mcp.client import call_mcp_tool

        session = _make_mock_session()
        session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await call_mcp_tool("https://m.example/", {}, "do_thing", {})
            await call_mcp_tool("https://m.example/", {}, "do_thing", {})

        session.initialize.assert_awaited_once()

    async def test_evicts_and_retries_on_call_tool_failure(
        self, restore_runtime_pool: None
    ) -> None:
        """First ``call_tool`` raises → pool evicts, second succeeds."""
        from aios.mcp.client import call_mcp_tool

        s_a, s_b = _make_mock_session(), _make_mock_session()
        s_a.call_tool = AsyncMock(side_effect=RuntimeError("broken"))
        s_b.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            result = await call_mcp_tool("https://m.example/", {}, "do_thing", {})

        assert result == {"content": ""}
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1
