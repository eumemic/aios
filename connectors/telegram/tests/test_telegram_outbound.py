"""Unit coverage for the outbound action tools — ``telegram_typing``,
``telegram_edit_message``, ``telegram_delete_message``, ``telegram_react`` —
plus ``telegram_send`` with ``parse_mode="html"``.

PTB's ``Bot`` is mocked end-to-end; tests assert that the right
``send_*`` / ``edit_*`` / ``delete_*`` / ``set_message_reaction`` method
was called with the right arguments.

Tools are called directly with explicit kwargs — the SDK's focal-channel
injection and SandboxPath resolution are SDK concerns covered in
``packages/aios-connector-http/tests``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import ReactionTypeEmoji
from telegram.constants import ChatAction

from aios_telegram.connector import TelegramConnector
from tests.conftest import CONNECTION_ID


@pytest.fixture
def bot() -> Any:
    """Mocked PTB Bot with stubbed send_* / edit_* / delete_* / react methods."""
    b = MagicMock()
    b.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    b.send_chat_action = AsyncMock(return_value=True)
    b.edit_message_text = AsyncMock(return_value=MagicMock(message_id=99))
    b.delete_message = AsyncMock(return_value=True)
    b.set_message_reaction = AsyncMock(return_value=True)
    return b


# ── telegram_typing ──────────────────────────────────────────────────


async def test_telegram_typing_default_action(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_typing(chat_id="123", connection_id=CONNECTION_ID)
    assert result == {"status": "ok"}
    bot.send_chat_action.assert_awaited_once_with(chat_id=123, action=ChatAction.TYPING)


async def test_telegram_typing_upload_photo_action(connector: TelegramConnector, bot: Any) -> None:
    await connector.telegram_typing(
        action="upload_photo", chat_id="123", connection_id=CONNECTION_ID
    )
    bot.send_chat_action.assert_awaited_once_with(chat_id=123, action=ChatAction.UPLOAD_PHOTO)


# ── telegram_edit_message ────────────────────────────────────────────


async def test_telegram_edit_message_plain(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_edit_message(
        message_id=99, text="fixed", chat_id="123", connection_id=CONNECTION_ID
    )
    assert result == {"message_id": 99}
    bot.edit_message_text.assert_awaited_once_with(
        chat_id=123, message_id=99, text="fixed", parse_mode=None
    )


async def test_telegram_edit_message_markdown_mode_converts(
    connector: TelegramConnector, bot: Any
) -> None:
    await connector.telegram_edit_message(
        message_id=99,
        text="**bold**",
        parse_mode="markdown",
        chat_id="123",
        connection_id=CONNECTION_ID,
    )
    kwargs = bot.edit_message_text.call_args.kwargs
    assert kwargs["text"] == "<b>bold</b>"
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_edit_message_html_mode_passes_through(
    connector: TelegramConnector, bot: Any
) -> None:
    """``parse_mode="html"`` ships raw HTML to Telegram (matching the
    Bot API's own ``parse_mode=HTML`` semantics).  No markdown
    conversion — what the agent wrote is what gets sent."""
    await connector.telegram_edit_message(
        message_id=99,
        text='<a href="https://example.com">link</a>',
        parse_mode="html",
        chat_id="123",
        connection_id=CONNECTION_ID,
    )
    kwargs = bot.edit_message_text.call_args.kwargs
    assert kwargs["text"] == '<a href="https://example.com">link</a>'
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_edit_message_inline_returns_status_ok(
    connector: TelegramConnector, bot: Any
) -> None:
    """Editing an inline-bot message returns True from PTB; we surface status only."""
    bot.edit_message_text = AsyncMock(return_value=True)
    result = await connector.telegram_edit_message(
        message_id=99, text="hi", chat_id="123", connection_id=CONNECTION_ID
    )
    assert result == {"status": "ok"}


# ── telegram_delete_message ──────────────────────────────────────────


async def test_telegram_delete_message(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_delete_message(
        message_id=50, chat_id="123", connection_id=CONNECTION_ID
    )
    assert result == {"status": "ok"}
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=50)


# ── telegram_react ───────────────────────────────────────────────────


async def test_telegram_react_with_emoji(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_react(
        message_id=50, emoji="👍", chat_id="123", connection_id=CONNECTION_ID
    )
    assert result == {"status": "ok"}
    kwargs = bot.set_message_reaction.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["message_id"] == 50
    assert isinstance(kwargs["reaction"], list)
    assert len(kwargs["reaction"]) == 1
    assert isinstance(kwargs["reaction"][0], ReactionTypeEmoji)
    assert kwargs["reaction"][0].emoji == "👍"


async def test_telegram_react_clear(connector: TelegramConnector, bot: Any) -> None:
    await connector.telegram_react(
        message_id=50, emoji=None, chat_id="123", connection_id=CONNECTION_ID
    )
    kwargs = bot.set_message_reaction.call_args.kwargs
    assert kwargs["reaction"] is None


# ── telegram_send parse_mode ─────────────────────────────────────────


async def test_telegram_send_markdown_mode_converts(connector: TelegramConnector, bot: Any) -> None:
    """``parse_mode="markdown"`` runs the input through
    :func:`markdown_to_telegram_html`.  This is the renamed form of
    what used to be ``parse_mode="html"`` before the smoke-#17 fix —
    the old name collided with Telegram Bot API semantics."""
    await connector.telegram_send(
        text="**hi** _there_", parse_mode="markdown", chat_id="123", connection_id=CONNECTION_ID
    )
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == "<b>hi</b> <i>there</i>"
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_send_html_mode_passes_through(
    connector: TelegramConnector, bot: Any
) -> None:
    """``parse_mode="html"`` matches Telegram's own ``parse_mode=HTML``
    semantics: the agent writes HTML directly, the connector forwards
    verbatim.  Critical for agents that want to use raw ``<a href>``
    tags or other constructs the markdown converter doesn't emit."""
    raw = '<b>bold</b> <a href="https://example.com">link</a>'
    await connector.telegram_send(
        text=raw, parse_mode="html", chat_id="123", connection_id=CONNECTION_ID
    )
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == raw  # untouched
    assert kwargs["parse_mode"] == "HTML"


async def test_telegram_send_plain_mode_passes_through(
    connector: TelegramConnector, bot: Any
) -> None:
    await connector.telegram_send(text="**not bold**", chat_id="123", connection_id=CONNECTION_ID)
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == "**not bold**"
    assert kwargs["parse_mode"] is None
