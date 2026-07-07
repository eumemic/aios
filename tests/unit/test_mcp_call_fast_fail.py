"""Unit tests for the MCP tool-call fast-fail path (services don't hang on a
403/5xx; the server's message is surfaced instead of a generic timeout)."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from aios.mcp.client import _call_tool_fast, _McpHttpError
from aios.mcp.pool import HttpErrorSink, _make_error_hook


class _FakeSession:
    def __init__(self, *, hang: bool) -> None:
        self._hang = hang
        self.result = object()

    async def call_tool(self, name: str, args: dict[str, Any], *, meta: Any = None) -> Any:
        if self._hang:
            await asyncio.Event().wait()  # never resolves
        return self.result


async def test_returns_result_on_success() -> None:
    sess = _FakeSession(hang=False)
    out = await _call_tool_fast(sess, "t", {}, None, HttpErrorSink(), 5.0)  # type: ignore[arg-type]
    assert out is sess.result


async def test_fast_fails_with_http_error_before_timeout() -> None:
    sess = _FakeSession(hang=True)
    sink = HttpErrorSink()

    async def fire() -> None:
        await asyncio.sleep(0.05)
        sink.record(403, "Calendar MCP API ... is disabled. Enable it by visiting ...")

    fired = asyncio.create_task(fire())
    started = time.monotonic()
    # A long timeout — the call must abort on the sink signal, not wait it out.
    with pytest.raises(_McpHttpError) as exc:
        await _call_tool_fast(sess, "t", {}, None, sink, 30.0)  # type: ignore[arg-type]
    assert time.monotonic() - started < 2.0
    assert exc.value.status == 403
    assert "is disabled" in exc.value.body
    await fired


async def test_times_out_without_a_signal() -> None:
    sess = _FakeSession(hang=True)
    with pytest.raises(TimeoutError):
        await _call_tool_fast(sess, "t", {}, None, HttpErrorSink(), 0.2)  # type: ignore[arg-type]


async def test_error_hook_records_4xx_only() -> None:
    sink = HttpErrorSink()
    hook = _make_error_hook(sink)

    import httpx

    await hook(httpx.Response(200, text="ok"))
    assert not sink.event.is_set()  # success responses are ignored

    await hook(httpx.Response(403, text='{"error":"nope"}'))
    assert sink.event.is_set()
    assert sink.status == 403
    assert "nope" in sink.body


# ── #1698 (c): a group-shaped failure at call-time → typed error dict, no raise


class _GroupSession:
    """A ClientSession whose ``call_tool`` raises a bare ``BaseExceptionGroup``
    (the incident's ``[HTTPError, CancelledError]`` shape — not an
    ``Exception``, so it slips past ``except Exception``)."""

    async def call_tool(self, name: str, args: Any, *, meta: Any = None) -> Any:
        raise BaseExceptionGroup(
            "unhandled errors in a TaskGroup",
            [httpx.HTTPError("401"), asyncio.CancelledError()],
        )


async def test_call_time_group_becomes_transport_error_dict(monkeypatch: Any) -> None:
    """A ``BaseExceptionGroup`` surfacing at call-time collapses into the typed
    ``{"code": "transport_error"}`` envelope — never an unhandled raise."""
    from aios.harness import runtime
    from aios.mcp import client as mcp_client
    from aios.mcp.client import call_mcp_tool
    from aios.mcp.pool import McpSessionPool

    pool = McpSessionPool()

    async def _fake_acquire(*_a: Any, **_k: Any) -> Any:
        entry = MagicMock()
        entry.session = _GroupSession()
        entry.error_sink = MagicMock()
        return entry

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(pool, "acquire", _fake_acquire)
    monkeypatch.setattr(pool, "discard", _noop)
    monkeypatch.setattr(pool, "release", _noop)
    monkeypatch.setattr(pool, "is_unhealthy", lambda *_a, **_k: False)
    monkeypatch.setattr(mcp_client, "_TOOL_CALL_TIMEOUT_S", 5.0)
    monkeypatch.setattr(runtime, "mcp_session_pool", pool)

    result = await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})
    assert result["code"] == "transport_error"
    assert "error" in result


async def test_call_time_fast_returns_when_breaker_open(monkeypatch: Any) -> None:
    """AC #4: with the breaker open, ``call_mcp_tool`` fast-returns the degraded
    error dict WITHOUT attempting a connect (``acquire`` not called)."""
    from aios.harness import runtime
    from aios.mcp.client import call_mcp_tool
    from aios.mcp.pool import McpSessionPool

    pool = McpSessionPool()
    acquired = False

    async def _acquire(*_a: Any, **_k: Any) -> Any:
        nonlocal acquired
        acquired = True
        raise AssertionError("acquire must not run while the breaker is open")

    monkeypatch.setattr(pool, "acquire", _acquire)
    monkeypatch.setattr(pool, "is_unhealthy", lambda *_a, **_k: True)
    monkeypatch.setattr(runtime, "mcp_session_pool", pool)

    result = await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})
    assert result["code"] == "transport_error"
    assert acquired is False
