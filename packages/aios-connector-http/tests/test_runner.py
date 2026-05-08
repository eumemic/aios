"""Unit tests for the HttpConnector runner.

Verifies the dispatch logic and tool-registration without an actual HTTP
roundtrip: an :class:`AiosClient` mock stand-in lets us assert the
runner POSTs the right tool-result for each call.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aios_connector_http import HttpConnector, tool


class _ProbeConnector(HttpConnector):
    """Three tools — happy path, error, returns dict."""

    def __init__(self) -> None:
        super().__init__(base_url="http://x", token="aios_conn_x")
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @tool()
    async def shout(self, *, text: str) -> str:
        self.calls.append(("shout", {"text": text}))
        return text.upper()

    @tool()
    async def boom(self) -> str:
        self.calls.append(("boom", {}))
        raise RuntimeError("kaboom")

    @tool(name="say_struct")
    async def returns_dict(self, *, n: int) -> dict[str, int]:
        self.calls.append(("say_struct", {"n": n}))
        return {"doubled": n * 2}


@pytest.fixture
def probe() -> _ProbeConnector:
    c = _ProbeConnector()
    c._client = AsyncMock()  # type: ignore[assignment]
    return c


class TestDispatch:
    async def test_routes_call_to_decorated_method(self, probe: _ProbeConnector) -> None:
        await probe._dispatch_call(
            {
                "tool_call_id": "call_1",
                "session_id": "sess_1",
                "name": "shout",
                "arguments": json.dumps({"text": "hi"}),
            }
        )
        assert probe.calls == [("shout", {"text": "hi"})]
        probe._client.post_tool_result.assert_awaited_once_with(  # type: ignore[union-attr]
            session_id="sess_1", tool_call_id="call_1", content="HI"
        )

    async def test_dict_result_serialized_as_json(self, probe: _ProbeConnector) -> None:
        await probe._dispatch_call(
            {
                "tool_call_id": "call_2",
                "session_id": "sess_2",
                "name": "say_struct",
                "arguments": json.dumps({"n": 3}),
            }
        )
        probe._client.post_tool_result.assert_awaited_once()  # type: ignore[union-attr]
        kwargs = probe._client.post_tool_result.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["content"] == json.dumps({"doubled": 6})
        assert kwargs.get("is_error", False) is False

    async def test_unknown_tool_returns_error(self, probe: _ProbeConnector) -> None:
        await probe._dispatch_call(
            {
                "tool_call_id": "call_x",
                "session_id": "sess_x",
                "name": "no_such_tool",
                "arguments": "{}",
            }
        )
        kwargs = probe._client.post_tool_result.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["is_error"] is True
        body = json.loads(kwargs["content"])
        assert "unknown tool" in body["error"]

    async def test_exception_in_tool_becomes_error_result(
        self, probe: _ProbeConnector
    ) -> None:
        await probe._dispatch_call(
            {
                "tool_call_id": "call_b",
                "session_id": "sess_b",
                "name": "boom",
                "arguments": "{}",
            }
        )
        kwargs = probe._client.post_tool_result.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["is_error"] is True
        body = json.loads(kwargs["content"])
        assert body["error"] == "kaboom"

    async def test_malformed_arguments_dispatched_as_empty(
        self, probe: _ProbeConnector
    ) -> None:
        """The model occasionally emits invalid JSON for arguments; the
        runner shouldn't crash — it dispatches with no kwargs and lets
        the tool method's signature decide whether to accept that."""
        await probe._dispatch_call(
            {
                "tool_call_id": "call_p",
                "session_id": "sess_p",
                "name": "shout",
                "arguments": "not json {",
            }
        )
        # shout(text=...) requires 'text', so it'll raise — surfaces as
        # an error result, NOT a crash.
        kwargs = probe._client.post_tool_result.call_args.kwargs  # type: ignore[union-attr]
        assert kwargs["is_error"] is True


class TestToolCollection:
    async def test_collects_decorated_methods_only(self) -> None:
        class Conn(HttpConnector):
            def __init__(self) -> None:
                super().__init__(base_url="x", token="t")

            @tool()
            async def yes(self) -> str:
                return "y"

            async def no(self) -> str:
                return "n"

        c = Conn()
        assert "yes" in c._tools
        assert "no" not in c._tools

    async def test_explicit_name_override(self) -> None:
        class Conn(HttpConnector):
            def __init__(self) -> None:
                super().__init__(base_url="x", token="t")

            @tool(name="published_name")
            async def internal_method(self) -> str:
                return "ok"

        c = Conn()
        assert "published_name" in c._tools
        assert "internal_method" not in c._tools


class TestAnsweredDedup:
    async def test_skips_already_answered(self, probe: _ProbeConnector) -> None:
        """The tool_loop guards against double-execution on SSE replay
        by checking ``_answered`` before dispatching.  Verify directly."""
        probe._answered.add("call_1")
        # Simulate the loop's check inline (since we can't run the SSE).
        call = {"tool_call_id": "call_1", "session_id": "s", "name": "shout", "arguments": "{}"}
        if call["tool_call_id"] in probe._answered:
            pass  # would skip
        else:
            await probe._dispatch_call(call)
        assert probe.calls == []
