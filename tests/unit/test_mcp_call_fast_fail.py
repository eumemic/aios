"""Unit tests for the MCP tool-call fast-fail path (services don't hang on a
403/5xx; the server's message is surfaced instead of a generic timeout)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

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
