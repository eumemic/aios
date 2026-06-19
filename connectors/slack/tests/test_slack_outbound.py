"""Unit coverage for the outbound ``slack_send`` / ``slack_react`` tools.

``AsyncWebClient`` is mocked end-to-end; tests assert the tool maps its
arguments onto the right Slack Web API call with the right kwargs, runs
the body through the markdown→mrkdwn pipeline + clamp, stamps the focal
channel on the result, and records bot-thread participation for the
mention-gate bypass.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector_http import HttpConnector

from aios_slack.connector import SlackConnector, _SlackConnectionState
from aios_slack.format import MESSAGE_MAX_CHARS
from aios_slack.parse import GateOutcome, gate

CONNECTION_ID = "conn_test"
TEAM_ID = "T0123456789"
BOT_USER_ID = "U0987654321"
CHANNEL_ID = "C0FOCAL000"


@pytest.fixture
def web() -> Any:
    """A mock ``AsyncWebClient`` with the outbound methods stubbed."""
    w = MagicMock(name="AsyncWebClient")
    w.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1700000000.000100"})
    w.chat_update = AsyncMock(
        return_value={"ok": True, "ts": "1700000000.000100", "channel": CHANNEL_ID}
    )
    w.chat_delete = AsyncMock(
        return_value={"ok": True, "ts": "1700000000.000100", "channel": CHANNEL_ID}
    )
    w.reactions_add = AsyncMock(return_value={"ok": True})
    w.reactions_remove = AsyncMock(return_value={"ok": True})
    return w


@pytest.fixture
def connector(web: Any) -> SlackConnector:
    """SlackConnector with one pre-registered connection wired to ``web``.

    The per-connection state is pre-populated so tool tests call the
    methods directly without running ``serve_connection`` first.
    """
    import asyncio as _asyncio

    c = SlackConnector()
    c.state[CONNECTION_ID] = _SlackConnectionState(
        web_client=web,
        socket_client=MagicMock(name="AsyncSocketModeClient"),
        bot_user_id=BOT_USER_ID,
        team_id=TEAM_ID,
        inbound_queue=_asyncio.Queue(),
    )
    return c


# ── registration ──────────────────────────────────────────────────────


def test_outbound_tools_registered(connector: SlackConnector) -> None:
    assert isinstance(connector, HttpConnector)
    assert {
        "slack_send",
        "slack_react",
        "slack_edit_message",
        "slack_delete_message",
    } <= set(connector._tools)


# ── slack_send ─────────────────────────────────────────────────────────


async def test_slack_send_top_level(connector: SlackConnector, web: Any) -> None:
    result = await connector.slack_send(
        text="hello", chat_id=CHANNEL_ID, connection_id=CONNECTION_ID
    )
    assert result == {
        "ts": "1700000000.000100",
        "channel": f"slack/{TEAM_ID}/{CHANNEL_ID}",
    }
    web.chat_postMessage.assert_awaited_once_with(channel=CHANNEL_ID, text="hello", mrkdwn=True)


async def test_slack_send_renders_markdown_to_mrkdwn(connector: SlackConnector, web: Any) -> None:
    await connector.slack_send(
        text="**bold** and [a](https://x.y)",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    kwargs = web.chat_postMessage.call_args.kwargs
    assert kwargs["text"] == "*bold* and <https://x.y|a>"


async def test_slack_send_clamps_overlong_body(connector: SlackConnector, web: Any) -> None:
    await connector.slack_send(
        text="x" * (MESSAGE_MAX_CHARS + 100),
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    sent = web.chat_postMessage.call_args.kwargs["text"]
    assert len(sent) == MESSAGE_MAX_CHARS
    assert sent.endswith("…")


async def test_slack_send_threads_when_thread_ts_set(connector: SlackConnector, web: Any) -> None:
    await connector.slack_send(
        text="in thread",
        thread_ts="1699999999.000001",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    kwargs = web.chat_postMessage.call_args.kwargs
    assert kwargs["thread_ts"] == "1699999999.000001"


async def test_slack_send_omits_thread_ts_when_none(connector: SlackConnector, web: Any) -> None:
    await connector.slack_send(text="top", chat_id=CHANNEL_ID, connection_id=CONNECTION_ID)
    assert "thread_ts" not in web.chat_postMessage.call_args.kwargs


async def test_slack_send_records_thread_participation_for_threaded_reply(
    connector: SlackConnector, web: Any
) -> None:
    """A threaded reply records the thread anchor so the mention-gate
    ``bot_thread_participant`` bypass fires for later replies (§3.6)."""
    await connector.slack_send(
        text="reply",
        thread_ts="1699999999.000001",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert "1699999999.000001" in connector.state[CONNECTION_ID].bot_thread_ts


async def test_slack_send_records_own_ts_as_thread_root_for_top_level(
    connector: SlackConnector, web: Any
) -> None:
    """A top-level send records its own ``ts`` as a thread root the bot is
    active in (so a human reply under it bypasses the mention-gate)."""
    await connector.slack_send(text="top", chat_id=CHANNEL_ID, connection_id=CONNECTION_ID)
    assert "1700000000.000100" in connector.state[CONNECTION_ID].bot_thread_ts


# ── slack_react ──────────────────────────────────────────────────────────


async def test_slack_react_add(connector: SlackConnector, web: Any) -> None:
    result = await connector.slack_react(
        message_ts="1700000000.000100",
        emoji="eyes",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert result == {"status": "ok"}
    web.reactions_add.assert_awaited_once_with(
        channel=CHANNEL_ID, timestamp="1700000000.000100", name="eyes"
    )
    web.reactions_remove.assert_not_awaited()


async def test_slack_react_strips_colons_and_normalizes(
    connector: SlackConnector, web: Any
) -> None:
    await connector.slack_react(
        message_ts="1700000000.000100",
        emoji=":White_Check_Mark:",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert web.reactions_add.call_args.kwargs["name"] == "white_check_mark"


async def test_slack_react_none_is_single_call_noop(connector: SlackConnector, web: Any) -> None:
    """``emoji=None`` does not add; it does not fire a nameless
    ``reactions.remove`` (which Slack rejects). One-call contract holds:
    zero add calls, zero remove calls, returns ok."""
    result = await connector.slack_react(
        message_ts="1700000000.000100",
        emoji=None,
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert result == {"status": "ok"}
    web.reactions_add.assert_not_awaited()
    web.reactions_remove.assert_not_awaited()


# ── slack_edit_message ───────────────────────────────────────────────────


async def test_slack_edit_message_calls_chat_update(connector: SlackConnector, web: Any) -> None:
    result = await connector.slack_edit_message(
        message_ts="1700000000.000100",
        text="edited",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert result == {
        "ts": "1700000000.000100",
        "channel": f"slack/{TEAM_ID}/{CHANNEL_ID}",
    }
    web.chat_update.assert_awaited_once_with(
        channel=CHANNEL_ID, ts="1700000000.000100", text="edited", mrkdwn=True
    )


async def test_slack_edit_message_renders_markdown_to_mrkdwn(
    connector: SlackConnector, web: Any
) -> None:
    await connector.slack_edit_message(
        message_ts="1700000000.000100",
        text="**bold** and [a](https://x.y)",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    kwargs = web.chat_update.call_args.kwargs
    assert kwargs["text"] == "*bold* and <https://x.y|a>"


async def test_slack_edit_message_clamps_overlong_body(connector: SlackConnector, web: Any) -> None:
    await connector.slack_edit_message(
        message_ts="1700000000.000100",
        text="x" * (MESSAGE_MAX_CHARS + 100),
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    sent = web.chat_update.call_args.kwargs["text"]
    assert len(sent) == MESSAGE_MAX_CHARS
    assert sent.endswith("…")


# ── slack_delete_message ─────────────────────────────────────────────────


async def test_slack_delete_message_calls_chat_delete(connector: SlackConnector, web: Any) -> None:
    result = await connector.slack_delete_message(
        message_ts="1700000000.000100",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    assert result == {"status": "ok"}
    web.chat_delete.assert_awaited_once_with(channel=CHANNEL_ID, ts="1700000000.000100")


# ── contract: an edit via slack_edit_message does not re-wake the session ──


async def test_edit_via_slack_edit_message_does_not_rewake_session(
    connector: SlackConnector, web: Any
) -> None:
    """A ``chat.update`` on the bot's own message arrives back as a
    ``message_changed`` event whose nested author is the bot. The slice-B
    nested self-filter MUST drop it before ``emit_inbound`` — editing the
    bot's own message must never wake the session it edited from (§3.5,
    §3.6 gate 1).

    The flow is exercised end to end: (1) the model edits via
    ``slack_edit_message``; (2) Slack echoes the resulting
    ``message_changed`` envelope back over the socket; (3)
    ``_handle_envelope`` runs the gates and must NOT emit.
    """
    # 1. The model edits its own message.
    edit_result = await connector.slack_edit_message(
        message_ts="1700000000.000100",
        text="corrected answer",
        chat_id=CHANNEL_ID,
        connection_id=CONNECTION_ID,
    )
    edited_ts = edit_result["ts"]

    # 2. Slack delivers the echo: a ``message_changed`` whose NESTED author
    #    (event.message.user / event.message.edited.user) is the bot itself.
    echo_event: dict[str, Any] = {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL_ID,
        "channel_type": "channel",
        "team": TEAM_ID,
        "message": {
            "type": "message",
            "user": BOT_USER_ID,
            "text": "corrected answer",
            "ts": edited_ts,
            "edited": {"user": BOT_USER_ID, "ts": "1700000999.000000"},
        },
        "previous_message": {
            "user": BOT_USER_ID,
            "text": "answer",
            "ts": edited_ts,
        },
    }
    echo_envelope: dict[str, Any] = {
        "type": "events_api",
        "envelope_id": "env-edit-echo",
        "payload": {"api_app_id": "A0APP00000", "event": echo_event},
    }

    # Sanity: the pure gate drops a bot-authored edit as a self-loop (not a
    # DIVERT_EDIT, which only a human edit reaches).
    state = connector.state[CONNECTION_ID]
    decision = gate(
        echo_event,
        bot_user_id=state.bot_user_id,
        team_id=state.team_id,
        api_app_id="A0APP00000",
    )
    assert decision.outcome is GateOutcome.DROP
    assert decision.reason == "self"

    # 3. Drive it through the connector's envelope handler and assert the
    #    session is never re-woken.
    emit = AsyncMock()
    connector.emit_inbound = emit  # type: ignore[method-assign]
    await connector._handle_envelope(CONNECTION_ID, state, echo_envelope)
    emit.assert_not_awaited()
