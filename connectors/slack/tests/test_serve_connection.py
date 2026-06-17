"""Unit tests for the slice-1 Slack connection layer (no network).

Covers the four load-bearing decisions of design §3.3:

* secrets validation (``bot_token`` + ``app_token`` required),
* the 429 retry handler wired at Web-client construction,
* the fail-closed install-identity gate (INV-5), and
* the ack-first socket listener (correctness sev 82) that enqueues the
  raw event only *after* acking.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

import aios_slack.connector as connector_mod
from aios_slack.connector import SlackConnector, _SlackConnectionState

CONNECTION_ID = "conn_test"
TEAM_ID = "T0123456789"
BOT_USER_ID = "U0987654321"


def _make_web_client(monkeypatch: pytest.MonkeyPatch, *, team_id: str = TEAM_ID) -> MagicMock:
    """Patch ``AsyncWebClient`` so construction needs no token and no aiohttp.

    Returns the single instance the connector will build, with
    ``auth_test`` stubbed and a real ``retry_handlers`` list so the
    handler-append assertion is meaningful.
    """
    web = MagicMock(name="AsyncWebClient")
    web.retry_handlers = []
    web.auth_test = AsyncMock(return_value={"user_id": BOT_USER_ID, "team_id": team_id})
    monkeypatch.setattr(connector_mod, "AsyncWebClient", MagicMock(return_value=web))
    return web


def _make_socket_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``AsyncSocketModeClient`` with a real listener list + async stubs."""
    socket = MagicMock(name="AsyncSocketModeClient")
    socket.socket_mode_request_listeners = []
    socket.connect = AsyncMock()
    socket.disconnect = AsyncMock()
    socket.close = AsyncMock()
    socket.send_socket_mode_response = AsyncMock()
    monkeypatch.setattr(connector_mod, "AsyncSocketModeClient", MagicMock(return_value=socket))
    return socket


def _register_connection(c: SlackConnector, *, external_account_id: str = TEAM_ID) -> None:
    """Pre-populate the SDK per-connection slot the identity gate reads."""
    from aios_connector_http.runner import _ConnectionState

    c._connections[CONNECTION_ID] = _ConnectionState(
        connection_id=CONNECTION_ID,
        external_account_id=external_account_id,
    )


@pytest.fixture
def connector() -> SlackConnector:
    return SlackConnector()


def test_connector_type_is_slack(connector: SlackConnector) -> None:
    assert connector.connector == "slack"


async def test_missing_bot_token_raises(connector: SlackConnector) -> None:
    with pytest.raises(RuntimeError, match="bot_token"):
        await connector.serve_connection(CONNECTION_ID, {"app_token": "xapp-x"})


async def test_missing_app_token_raises(connector: SlackConnector) -> None:
    with pytest.raises(RuntimeError, match="app_token"):
        await connector.serve_connection(CONNECTION_ID, {"bot_token": "xoxb-x"})


async def test_rate_limit_retry_handler_wired_at_construction(
    connector: SlackConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = _make_web_client(monkeypatch)
    _make_socket_client(monkeypatch)
    _register_connection(connector)

    # Make auth.test mismatch so serve_connection returns fast (after the
    # client is built) without entering the TaskGroup — we only care that
    # the handler was appended at construction time.
    web.auth_test = AsyncMock(return_value={"user_id": BOT_USER_ID, "team_id": "TWRONG"})
    monkeypatch.setattr(connector, "emit_lifecycle", AsyncMock())

    await connector.serve_connection(CONNECTION_ID, {"bot_token": "xoxb-x", "app_token": "xapp-x"})

    assert any(isinstance(h, AsyncRateLimitErrorRetryHandler) for h in web.retry_handlers), (
        "rate-limit retry handler must be appended to the Web client at construction"
    )


async def test_identity_mismatch_fails_closed(
    connector: SlackConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_web_client(monkeypatch, team_id="TDIFFERENT")
    socket = _make_socket_client(monkeypatch)
    _register_connection(connector, external_account_id=TEAM_ID)

    emitted: list[dict[str, Any]] = []

    async def fake_emit_lifecycle(**kwargs: Any) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(connector, "emit_lifecycle", fake_emit_lifecycle)

    await connector.serve_connection(CONNECTION_ID, {"bot_token": "xoxb-x", "app_token": "xapp-x"})

    # Refused to serve: no live state, lifecycle emitted, socket never connected.
    assert CONNECTION_ID not in connector.state
    socket.connect.assert_not_called()
    assert len(emitted) == 1
    assert emitted[0]["event"] == "slack.install.identity_mismatch"
    assert emitted[0]["data"]["expected_team_id"] == TEAM_ID
    assert emitted[0]["data"]["actual_team_id"] == "TDIFFERENT"
    # The socket client is still closed in the finally.
    socket.disconnect.assert_awaited()


async def test_identity_match_serves_and_registers_listener(
    connector: SlackConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_web_client(monkeypatch)
    socket = _make_socket_client(monkeypatch)
    _register_connection(connector)

    # ``connect`` returns immediately; the serve task then blocks forever in
    # ``_run_socket`` waiting on an Event.  Cancel it shortly after start so
    # we can assert the steady-state wiring without a real socket.
    task = asyncio.ensure_future(
        connector.serve_connection(CONNECTION_ID, {"bot_token": "xoxb-x", "app_token": "xapp-x"})
    )
    await asyncio.sleep(0.05)
    try:
        assert CONNECTION_ID in connector.state
        state = connector.state[CONNECTION_ID]
        assert state.team_id == TEAM_ID
        assert state.bot_user_id == BOT_USER_ID
        assert len(socket.socket_mode_request_listeners) == 1
        socket.connect.assert_awaited()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # finally closed the socket.
    socket.disconnect.assert_awaited()


async def test_listener_acks_before_enqueue(
    connector: SlackConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket = _make_socket_client(monkeypatch)
    order: list[str] = []

    async def record_ack(response: SocketModeResponse) -> None:
        order.append(f"ack:{response.envelope_id}")

    socket.send_socket_mode_response = AsyncMock(side_effect=record_ack)

    state = _SlackConnectionState(
        web_client=MagicMock(),
        socket_client=socket,
        bot_user_id=BOT_USER_ID,
        team_id=TEAM_ID,
        inbound_queue=asyncio.Queue(),
    )
    connector.state[CONNECTION_ID] = state
    connector._register_listener(CONNECTION_ID, state)

    listener = socket.socket_mode_request_listeners[0]
    req = SocketModeRequest(
        type="events_api",
        envelope_id="env-1",
        payload={"event": {"type": "message", "text": "hi", "ts": "1700000000.000100"}},
    )
    await listener(socket, req)

    # Ack happened, exactly once, with the right envelope id.
    assert order == ["ack:env-1"]
    socket.send_socket_mode_response.assert_awaited_once()
    # The raw event was enqueued only after the ack.
    assert state.inbound_queue.qsize() == 1
    enqueued = state.inbound_queue.get_nowait()
    assert enqueued["type"] == "events_api"
    assert enqueued["envelope_id"] == "env-1"
    assert enqueued["payload"]["event"]["text"] == "hi"


async def test_drain_logs_without_emitting(
    connector: SlackConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slice A must NOT call emit_inbound — parsing/gating is slice B.
    emit = AsyncMock()
    monkeypatch.setattr(connector, "emit_inbound", emit)

    state = _SlackConnectionState(
        web_client=MagicMock(),
        socket_client=MagicMock(),
        bot_user_id=BOT_USER_ID,
        team_id=TEAM_ID,
        inbound_queue=asyncio.Queue(),
    )
    await state.inbound_queue.put({"type": "events_api", "envelope_id": "e", "payload": {}})

    drain = asyncio.ensure_future(connector._drain_queue(CONNECTION_ID, state))
    await asyncio.sleep(0.02)
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    emit.assert_not_called()
