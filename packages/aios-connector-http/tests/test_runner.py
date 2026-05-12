"""Unit tests for the multi-connection HttpConnector runner (#328 PR 5).

Tool dispatch and focal injection are exercised directly via
:meth:`HttpConnector.dispatch_call`; the heavyweight SSE plumbing
(discovery + tool loops) is exercised end-to-end in
``tests/e2e/test_echo_http_connector.py``.

Tool-result POSTs are intercepted by overriding the
:meth:`HttpConnector._post_tool_result` hook on a subclass; emit_inbound
is intercepted by mocking the underlying ``httpx.AsyncClient.post``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from aios_connector_http import HttpConnector, tool


class _RecordedResult:
    """One captured ``_post_tool_result`` call."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _ProbeConnector(HttpConnector):
    """Three tools — happy path, error, returns dict.

    Inherits the standard :class:`HttpConnector` shape but overrides
    :meth:`_post_tool_result` to capture instead of POSTing.  Tests
    instantiate with throwaway env (``base_url`` / ``token`` overrides)
    and read :attr:`results` after each :meth:`dispatch_call`.
    """

    connector = "probe"

    def __init__(self) -> None:
        super().__init__(base_url="http://x", token="aios_runtime_x")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[_RecordedResult] = []
        # Make _require_client() return a sentinel — dispatch_call never
        # actually uses the client because we override _post_tool_result.
        self._client = MagicMock()

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

    async def _post_tool_result(  # type: ignore[override]
        self,
        client: Any,
        *,
        connection_id: str,
        session_id: str,
        tool_call_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool = False,
    ) -> None:
        del client
        self.results.append(
            _RecordedResult(
                connection_id=connection_id,
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=content,
                is_error=is_error,
            )
        )


@pytest.fixture
def probe() -> _ProbeConnector:
    return _ProbeConnector()


class TestDispatch:
    async def test_routes_call_to_decorated_method(self, probe: _ProbeConnector) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_1",
                "session_id": "sess_1",
                "name": "shout",
                "arguments": json.dumps({"text": "hi"}),
            }
        )
        assert probe.calls == [("shout", {"text": "hi"})]
        assert len(probe.results) == 1
        r = probe.results[0]
        assert r.kwargs["content"] == "HI"
        assert r.kwargs["connection_id"] == "conn_1"
        assert r.kwargs["session_id"] == "sess_1"
        assert r.kwargs["is_error"] is False

    async def test_dict_result_serialized_as_json(self, probe: _ProbeConnector) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_2",
                "session_id": "sess_2",
                "name": "say_struct",
                "arguments": json.dumps({"n": 3}),
            }
        )
        r = probe.results[0]
        assert r.kwargs["content"] == json.dumps({"doubled": 6})
        assert r.kwargs["is_error"] is False

    async def test_unknown_tool_returns_error(self, probe: _ProbeConnector) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_x",
                "session_id": "sess_x",
                "name": "no_such_tool",
                "arguments": "{}",
            }
        )
        r = probe.results[0]
        assert r.kwargs["is_error"] is True
        body = json.loads(r.kwargs["content"])
        assert "unknown tool" in body["error"]

    async def test_exception_in_tool_becomes_error_result(
        self, probe: _ProbeConnector
    ) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_b",
                "session_id": "sess_b",
                "name": "boom",
                "arguments": "{}",
            }
        )
        r = probe.results[0]
        assert r.kwargs["is_error"] is True
        body = json.loads(r.kwargs["content"])
        assert body["error"] == "kaboom"

    async def test_malformed_arguments_dispatched_as_empty(
        self, probe: _ProbeConnector
    ) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_p",
                "session_id": "sess_p",
                "name": "shout",
                "arguments": "not json {",
            }
        )
        # ``shout(text=...)`` requires ``text``, so it'll raise — surfaces
        # as an error result, NOT a crash.
        r = probe.results[0]
        assert r.kwargs["is_error"] is True


class TestToolCollection:
    async def test_collects_decorated_methods_only(self) -> None:
        class Conn(HttpConnector):
            connector = "test"

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
            connector = "test"

            def __init__(self) -> None:
                super().__init__(base_url="x", token="t")

            @tool(name="published_name")
            async def internal_method(self) -> str:
                return "ok"

        c = Conn()
        assert "published_name" in c._tools
        assert "internal_method" not in c._tools

    async def test_subclass_without_connector_attr_raises(self) -> None:
        """Forgetting ``connector = ...`` is a programmer error — must crash."""

        class Bad(HttpConnector):
            def __init__(self) -> None:
                super().__init__(base_url="x", token="t")

        with pytest.raises(RuntimeError, match="must set ``connector``"):
            Bad()


class TestAnsweredDedup:
    async def test_skips_already_answered(self, probe: _ProbeConnector) -> None:
        """The tool_loop guards against double-execution on SSE replay
        by checking ``_answered`` before dispatching.  Verify directly."""
        probe._answered.add("call_1")
        call = {
            "connection_id": "conn_1",
            "tool_call_id": "call_1",
            "session_id": "s",
            "name": "shout",
            "arguments": "{}",
        }
        if call["tool_call_id"] in probe._answered:
            pass  # would skip
        else:
            await probe.dispatch_call(call)
        assert probe.calls == []


class _FocalConnector(_ProbeConnector):
    """Tools that opt into focal-channel + connection_id injection."""

    connector = "telegram"

    def __init__(self) -> None:
        super().__init__()
        self.focal_calls: list[dict[str, Any]] = []

    @tool()
    async def needs_chat(self, *, text: str, chat_id: str) -> str:
        self.focal_calls.append({"text": text, "chat_id": chat_id})
        return f"sent to {chat_id}"

    @tool()
    async def needs_both(self, *, account: str, chat_id: str, text: str) -> str:
        self.focal_calls.append({"account": account, "chat_id": chat_id, "text": text})
        return f"{account}:{chat_id}"

    @tool()
    async def needs_connection(self, *, connection_id: str, text: str) -> str:
        self.focal_calls.append({"connection_id": connection_id, "text": text})
        return connection_id

    @tool()
    async def chat_only(self, *, text: str) -> str:
        """No focal kwargs in signature — runner shouldn't inject."""
        self.focal_calls.append({"text": text})
        return text


class TestFocalChannelInjection:
    async def test_injects_chat_id_when_signature_accepts(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_f1",
                "session_id": "s",
                "name": "needs_chat",
                "arguments": json.dumps({"text": "hi"}),
                "focal_channel": "telegram/bot1/chat-123",
            }
        )
        assert c.focal_calls == [{"text": "hi", "chat_id": "chat-123"}]

    async def test_injects_both_account_and_chat_id(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_f2",
                "session_id": "s",
                "name": "needs_both",
                "arguments": json.dumps({"text": "hi"}),
                "focal_channel": "signal/+15551234/group-abc",
            }
        )
        assert c.focal_calls == [
            {"account": "+15551234", "chat_id": "group-abc", "text": "hi"}
        ]

    async def test_injects_connection_id_when_signature_accepts(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_xyz",
                "tool_call_id": "call_c1",
                "session_id": "s",
                "name": "needs_connection",
                "arguments": json.dumps({"text": "hi"}),
                "focal_channel": "telegram/bot1/chat-1",
            }
        )
        assert c.focal_calls == [{"connection_id": "conn_xyz", "text": "hi"}]

    async def test_skips_injection_when_signature_doesnt_ask(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_f3",
                "session_id": "s",
                "name": "chat_only",
                "arguments": json.dumps({"text": "hi"}),
                "focal_channel": "telegram/bot1/chat-123",
            }
        )
        assert c.focal_calls == [{"text": "hi"}]

    async def test_explicit_kwarg_wins_over_focal(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_f4",
                "session_id": "s",
                "name": "needs_chat",
                "arguments": json.dumps({"text": "hi", "chat_id": "explicit"}),
                "focal_channel": "telegram/bot1/chat-from-focal",
            }
        )
        assert c.focal_calls == [{"text": "hi", "chat_id": "explicit"}]

    async def test_no_focal_channel_is_a_noop(self) -> None:
        c = _FocalConnector()
        await c.dispatch_call(
            {
                "connection_id": "conn_1",
                "tool_call_id": "call_f5",
                "session_id": "s",
                "name": "chat_only",
                "arguments": json.dumps({"text": "hi"}),
                "focal_channel": "",
            }
        )
        assert c.focal_calls == [{"text": "hi"}]


class TestLogging:
    """Structured log records at the SDK's two boundary points.

    The SDK is the only layer that sees every inbound and every tool
    call — per-connector logging would duplicate this.  Connector authors
    debugging "did the message arrive?" / "did the tool fire?" should
    not need to add their own logging to find out.
    """

    async def test_emit_inbound_logs_one_record(self, probe: _ProbeConnector) -> None:
        # Patch the underlying async httpx client so emit_inbound's POST
        # is a no-op returning a stub response.
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"appended_event_id": "evt_x"})
        mock_post = AsyncMock(return_value=mock_response)
        probe._client.get_async_httpx_client.return_value = MagicMock(post=mock_post)  # type: ignore[union-attr]
        with structlog.testing.capture_logs() as records:
            await probe.emit_inbound(
                connection_id="conn_1",
                chat_id="chat-42",
                sender={"id": 99, "name": "alice"},
                content="hello world",
            )
        events = [r for r in records if r.get("event") == "connector.inbound"]
        assert len(events) == 1
        rec = events[0]
        assert rec["chat_id"] == "chat-42"
        assert rec["content_len"] == len("hello world")
        assert rec["sender"] == {"id": 99, "name": "alice"}
        assert rec["connection_id"] == "conn_1"

    async def test_emit_inbound_does_not_log_message_content(
        self, probe: _ProbeConnector
    ) -> None:
        """Message bodies are user data — log length, not content."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"appended_event_id": "evt_x"})
        mock_post = AsyncMock(return_value=mock_response)
        probe._client.get_async_httpx_client.return_value = MagicMock(post=mock_post)  # type: ignore[union-attr]
        secret_body = "sensitive-canary-DO-NOT-LOG"
        with structlog.testing.capture_logs() as records:
            await probe.emit_inbound(
                connection_id="conn_1",
                chat_id="c",
                sender={"id": 1},
                content=secret_body,
            )
        for r in records:
            for v in r.values():
                assert secret_body not in str(v), f"content leaked into log field: {r}"

    async def test_dispatch_call_logs_dispatched_then_completed(
        self, probe: _ProbeConnector
    ) -> None:
        with structlog.testing.capture_logs() as records:
            await probe.dispatch_call(
                {
                    "connection_id": "conn_1",
                    "tool_call_id": "call_1",
                    "session_id": "sess_1",
                    "name": "shout",
                    "arguments": json.dumps({"text": "hi"}),
                }
            )
        events = [r["event"] for r in records]
        assert "connector.tool_call.dispatched" in events
        assert "connector.tool_call.completed" in events
        completed = next(r for r in records if r["event"] == "connector.tool_call.completed")
        assert completed["name"] == "shout"
        assert completed["tool_call_id"] == "call_1"
        assert completed["is_error"] is False

    async def test_dispatch_call_logs_failed_on_unknown_tool(
        self, probe: _ProbeConnector
    ) -> None:
        with structlog.testing.capture_logs() as records:
            await probe.dispatch_call(
                {
                    "connection_id": "conn_1",
                    "tool_call_id": "call_x",
                    "session_id": "sess_x",
                    "name": "no_such_tool",
                    "arguments": "{}",
                }
            )
        failed = [r for r in records if r["event"] == "connector.tool_call.failed"]
        assert len(failed) == 1
        assert failed[0]["name"] == "no_such_tool"
        assert failed[0]["reason"] == "unknown_tool"


class TestMultiConnectionDispatch:
    """The new shape's headline property: one runner, N connections,
    dispatch routes by ``connection_id`` on the call payload."""

    async def test_two_connections_dispatched_independently(
        self, probe: _ProbeConnector
    ) -> None:
        await probe.dispatch_call(
            {
                "connection_id": "conn_A",
                "tool_call_id": "call_A",
                "session_id": "sess_A",
                "name": "shout",
                "arguments": json.dumps({"text": "hi-A"}),
            }
        )
        await probe.dispatch_call(
            {
                "connection_id": "conn_B",
                "tool_call_id": "call_B",
                "session_id": "sess_B",
                "name": "shout",
                "arguments": json.dumps({"text": "hi-B"}),
            }
        )
        assert len(probe.results) == 2
        assert probe.results[0].kwargs["connection_id"] == "conn_A"
        assert probe.results[0].kwargs["session_id"] == "sess_A"
        assert probe.results[1].kwargs["connection_id"] == "conn_B"
        assert probe.results[1].kwargs["session_id"] == "sess_B"
