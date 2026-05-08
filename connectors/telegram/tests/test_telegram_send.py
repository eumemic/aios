"""Unit coverage for ``telegram_send``'s ``attachments`` parameter,
type-by-extension routing, and the caption-on-first quirk in
``send_media_group``.

PTB's ``Bot`` is mocked end-to-end; tests assert that the right
``send_*`` method was called with the right ``chat_id`` /
``caption`` / ``media`` arguments.

Tools take already-resolved host paths (``Path`` objects) — the SDK's
``SandboxPath`` resolution happens at dispatch time and is exercised by
the SDK's own tests.  Connector-side tests focus on PTB routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector_http import HttpConnector

from aios_telegram.config import Settings
from aios_telegram.connector import (
    TelegramConnector,
    _build_media_group,
    _classify,
)


def test_classify_extensions() -> None:
    assert _classify(Path("/x/cat.jpg")) == "photo"
    assert _classify(Path("/x/CAT.JPEG")) == "photo"
    assert _classify(Path("/x/clip.mp4")) == "video"
    assert _classify(Path("/x/voice.ogg")) == "voice"
    assert _classify(Path("/x/song.mp3")) == "audio"
    assert _classify(Path("/x/report.pdf")) == "document"
    assert _classify(Path("/x/no-ext")) == "document"


def test_build_media_group_caption_on_first_only() -> None:
    items = _build_media_group(
        [Path("/host/a.jpg"), Path("/host/b.jpg"), Path("/host/c.jpg")],
        caption="three pics",
    )
    assert items[0].caption == "three pics"
    assert items[1].caption is None
    assert items[2].caption is None


def test_build_media_group_no_caption() -> None:
    items = _build_media_group([Path("/host/a.jpg"), Path("/host/b.jpg")], caption=None)
    assert all(item.caption is None for item in items)


def test_build_media_group_voice_demoted_to_document() -> None:
    """Voice can't ride in a media group; falls back to document."""
    from telegram import InputMediaDocument

    items = _build_media_group([Path("/host/v.ogg")], caption=None)
    assert isinstance(items[0], InputMediaDocument)


def test_telegram_connector_subclasses_http_connector() -> None:
    """Sanity: the SDK base picks up our @tool methods as a tool registry."""
    cfg = Settings(bot_token="0:test")
    c = TelegramConnector(cfg)
    assert isinstance(c, HttpConnector)
    expected = {
        "telegram_send",
        "telegram_typing",
        "telegram_edit_message",
        "telegram_delete_message",
        "telegram_react",
    }
    assert expected <= set(c._tools)


@pytest.fixture
def bot() -> Any:
    b = MagicMock()
    b.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    b.send_photo = AsyncMock(return_value=MagicMock(message_id=43))
    b.send_voice = AsyncMock(return_value=MagicMock(message_id=44))
    b.send_video = AsyncMock(return_value=MagicMock(message_id=45))
    b.send_audio = AsyncMock(return_value=MagicMock(message_id=46))
    b.send_document = AsyncMock(return_value=MagicMock(message_id=47))
    b.send_media_group = AsyncMock(
        return_value=[
            MagicMock(message_id=100),
            MagicMock(message_id=101),
        ]
    )
    return b


# End-to-end tests calling the tool method directly with resolved paths.


async def test_telegram_send_text_only(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_send(text="hello", chat_id="123")
    assert result == {"message_id": 42}
    bot.send_message.assert_awaited_once_with(chat_id=123, text="hello", parse_mode=None)


async def test_telegram_send_single_photo_routes_to_send_photo(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    photo = tmp_path / "cat.jpg"
    photo.write_bytes(b"x")

    result = await connector.telegram_send(text="look", attachments=[photo], chat_id="123")

    assert result == {"message_id": 43}
    bot.send_photo.assert_awaited_once()
    kwargs = bot.send_photo.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["photo"] == photo
    assert kwargs["caption"] == "look"


async def test_telegram_send_single_voice_routes_to_send_voice(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    voice = tmp_path / "v.ogg"
    voice.write_bytes(b"x")
    await connector.telegram_send(text="", attachments=[voice], chat_id="123")
    bot.send_voice.assert_awaited_once()


async def test_telegram_send_single_document_routes_to_send_document(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"x")
    await connector.telegram_send(text="", attachments=[doc], chat_id="123")
    bot.send_document.assert_awaited_once()


async def test_telegram_send_multi_media_uses_send_media_group(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.jpg"
    b_path = tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b_path.write_bytes(b"x")

    result = await connector.telegram_send(
        text="two pics", attachments=[a, b_path], chat_id="123"
    )

    assert result == {"message_ids": [100, 101]}
    bot.send_media_group.assert_awaited_once()
    bot.send_photo.assert_not_awaited()
    media = bot.send_media_group.call_args.kwargs["media"]
    assert len(media) == 2
    assert media[0].caption == "two pics"
    assert media[1].caption is None


async def test_telegram_send_non_int_chat_id_raises(
    connector: TelegramConnector,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        await connector.telegram_send(text="hi", chat_id="not-a-number")


# Integration check: feed the SDK dispatch path with a sandbox path
# string and verify the connector receives a resolved Path.


async def test_telegram_send_dispatch_resolves_sandbox_path(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip via dispatch_call: the SDK should hand the connector a
    real :class:`Path` for every ``SandboxPath`` argument so PTB's
    upload code can ``open()`` it directly."""
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir()
    (ws / "cat.jpg").write_bytes(b"x")
    connector._client = AsyncMock()

    await connector.dispatch_call(
        {
            "tool_call_id": "c1",
            "session_id": "sess-1",
            "name": "telegram_send",
            "arguments": json.dumps(
                {"text": "look", "attachments": ["/workspace/cat.jpg"]}
            ),
            "focal_channel": "telegram/0/123",
        }
    )

    photo_arg: Any = bot.send_photo.call_args.kwargs["photo"]
    assert isinstance(photo_arg, Path)
    assert photo_arg == ws / "cat.jpg"
