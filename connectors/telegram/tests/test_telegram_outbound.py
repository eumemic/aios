"""Unit coverage for the outbound action tools added in the openclaw-parity pass:
``telegram_typing``, ``telegram_edit_message``, ``telegram_delete_message``,
``telegram_react``, plus ``telegram_send`` with ``parse_mode="html"``.

PTB's ``Bot`` is mocked end-to-end; tests assert that the right
``send_*`` / ``edit_*`` / ``delete_*`` / ``set_message_reaction`` method
was called with the right arguments.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector.base import ToolDescriptor
from telegram import ReactionTypeEmoji
from telegram.constants import ChatAction

from aios_telegram.config import Settings
from aios_telegram.connector import TelegramConnector


@pytest.fixture
def connector() -> TelegramConnector:
    cfg = Settings(bot_token="0:test")
    c = TelegramConnector(cfg)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    bot.send_chat_action = AsyncMock(return_value=True)
    bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=99))
    bot.delete_message = AsyncMock(return_value=True)
    bot.set_message_reaction = AsyncMock(return_value=True)
    c._application = MagicMock()
    c._application.bot = bot
    return c


def _descriptor(connector: TelegramConnector, name: str) -> ToolDescriptor:
    descriptor = next(d for d in connector._tools if d.name == name)
    assert isinstance(descriptor, ToolDescriptor)
    return descriptor


def _stub_focal(connector: TelegramConnector, value: str | None) -> None:
    connector._focal_from_request_meta = lambda: value  # type: ignore[method-assign]


def _stub_session_id(connector: TelegramConnector, value: str | None) -> None:
    connector.current_session_id = lambda: value  # type: ignore[method-assign]


def _decode(content_list: list[Any]) -> dict[str, Any]:
    assert len(content_list) == 1
    payload = json.loads(content_list[0].text)
    assert isinstance(payload, dict)
    return payload


# ── telegram_typing ──────────────────────────────────────────────────


async def test_telegram_typing_default_action(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_typing")
    result = _decode(await connector._invoke_tool(descriptor, {}))
    assert result == {"status": "ok"}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.send_chat_action.assert_awaited_once_with(chat_id=123, action=ChatAction.TYPING)


async def test_telegram_typing_upload_photo_action(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_typing")
    await connector._invoke_tool(descriptor, {"action": "upload_photo"})
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.send_chat_action.assert_awaited_once_with(chat_id=123, action=ChatAction.UPLOAD_PHOTO)


# ── telegram_edit_message ────────────────────────────────────────────


async def test_telegram_edit_message_plain(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_edit_message")
    result = _decode(await connector._invoke_tool(descriptor, {"message_id": 99, "text": "fixed"}))
    assert result == {"message_id": 99}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=123, message_id=99, text="fixed", parse_mode=None
    )


async def test_telegram_edit_message_html_mode_converts_markdown(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_edit_message")
    await connector._invoke_tool(
        descriptor,
        {"message_id": 99, "text": "**bold**", "parse_mode": "html"},
    )
    bot = connector._application.bot  # type: ignore[union-attr]
    kwargs = bot.edit_message_text.call_args.kwargs
    assert kwargs["text"] == "<b>bold</b>"
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_edit_message_inline_returns_status_ok(
    connector: TelegramConnector,
) -> None:
    """Editing an inline-bot message returns True from PTB; we surface status only."""
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.edit_message_text = AsyncMock(return_value=True)
    descriptor = _descriptor(connector, "telegram_edit_message")
    result = _decode(await connector._invoke_tool(descriptor, {"message_id": 99, "text": "hi"}))
    assert result == {"status": "ok"}


# ── telegram_delete_message ──────────────────────────────────────────


async def test_telegram_delete_message(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_delete_message")
    result = _decode(await connector._invoke_tool(descriptor, {"message_id": 50}))
    assert result == {"status": "ok"}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=50)


# ── telegram_react ───────────────────────────────────────────────────


async def test_telegram_react_with_emoji(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_react")
    result = _decode(await connector._invoke_tool(descriptor, {"message_id": 50, "emoji": "👍"}))
    assert result == {"status": "ok"}
    bot = connector._application.bot  # type: ignore[union-attr]
    kwargs = bot.set_message_reaction.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["message_id"] == 50
    assert isinstance(kwargs["reaction"], list)
    assert len(kwargs["reaction"]) == 1
    assert isinstance(kwargs["reaction"][0], ReactionTypeEmoji)
    assert kwargs["reaction"][0].emoji == "👍"


async def test_telegram_react_clear(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_react")
    await connector._invoke_tool(descriptor, {"message_id": 50, "emoji": None})
    bot = connector._application.bot  # type: ignore[union-attr]
    kwargs = bot.set_message_reaction.call_args.kwargs
    assert kwargs["reaction"] is None


# ── telegram_send parse_mode ─────────────────────────────────────────


async def test_telegram_send_html_mode_converts_markdown(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_send")
    await connector._invoke_tool(descriptor, {"text": "**hi** _there_", "parse_mode": "html"})
    bot = connector._application.bot  # type: ignore[union-attr]
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == "<b>hi</b> <i>there</i>"
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_send_plain_mode_passes_through(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_send")
    await connector._invoke_tool(descriptor, {"text": "**not bold**"})
    bot = connector._application.bot  # type: ignore[union-attr]
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == "**not bold**"
    assert kwargs["parse_mode"] is None


async def test_telegram_send_invalid_parse_mode_raises(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _descriptor(connector, "telegram_send")
    with pytest.raises(ValueError, match="parse_mode must be"):
        await connector._invoke_tool(descriptor, {"text": "x", "parse_mode": "markdown"})
