"""Unit coverage for ``telegram_send``'s ``attachments`` parameter,
type-by-extension routing, and the caption-on-first quirk in
``send_media_group``.

The PTB ``Bot`` is mocked end-to-end; tests assert that the right
``send_*`` method was called with the right ``chat_id`` /
``caption`` / ``media`` arguments.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios_telegram.config import Settings
from aios_telegram.connector import (
    TelegramConnector,
    _build_media_group,
    _classify,
)


def test_classify_extensions() -> None:
    assert _classify("/x/cat.jpg") == "photo"
    assert _classify("/x/CAT.JPEG") == "photo"
    assert _classify("/x/clip.mp4") == "video"
    assert _classify("/x/voice.ogg") == "voice"
    assert _classify("/x/song.mp3") == "audio"
    assert _classify("/x/report.pdf") == "document"
    assert _classify("/x/no-ext") == "document"


def test_build_media_group_caption_on_first_only() -> None:
    items = _build_media_group(
        ["/host/a.jpg", "/host/b.jpg", "/host/c.jpg"],
        caption="three pics",
    )
    assert items[0].caption == "three pics"
    assert items[1].caption is None
    assert items[2].caption is None


def test_build_media_group_no_caption() -> None:
    items = _build_media_group(["/host/a.jpg", "/host/b.jpg"], caption=None)
    assert all(item.caption is None for item in items)


def test_build_media_group_voice_demoted_to_document() -> None:
    """Voice can't ride in a media group; falls back to document."""
    from telegram import InputMediaDocument

    items = _build_media_group(["/host/v.ogg"], caption=None)
    assert isinstance(items[0], InputMediaDocument)


@pytest.fixture
def connector() -> TelegramConnector:
    cfg = Settings(bot_token="0:test")
    c = TelegramConnector(cfg)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    bot.send_photo = AsyncMock(return_value=MagicMock(message_id=43))
    bot.send_voice = AsyncMock(return_value=MagicMock(message_id=44))
    bot.send_video = AsyncMock(return_value=MagicMock(message_id=45))
    bot.send_audio = AsyncMock(return_value=MagicMock(message_id=46))
    bot.send_document = AsyncMock(return_value=MagicMock(message_id=47))
    bot.send_media_group = AsyncMock(
        return_value=[
            MagicMock(message_id=100),
            MagicMock(message_id=101),
        ]
    )
    c._application = MagicMock()
    c._application.bot = bot
    return c


def _patch_session_id(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr(TelegramConnector, "current_session_id", lambda self: value)


async def test_telegram_send_text_only(
    connector: TelegramConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_session_id(monkeypatch, None)
    result = await connector.telegram_send("hello", chat_id="123")
    assert result == {"message_id": 42}
    connector._application.bot.send_message.assert_awaited_once_with(  # type: ignore[union-attr]
        chat_id=123, text="hello"
    )


async def test_telegram_send_single_photo_routes_to_send_photo(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    result = await connector.telegram_send(
        "look", attachments=["/workspace/cat.jpg"], chat_id="123"
    )

    assert result == {"message_id": 43}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.send_photo.assert_awaited_once()
    kwargs = bot.send_photo.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["photo"] == str(ws / "cat.jpg")
    assert kwargs["caption"] == "look"


async def test_telegram_send_single_voice_routes_to_send_voice(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "v.ogg").write_bytes(b"x")

    await connector.telegram_send("", attachments=["/workspace/v.ogg"], chat_id="123")

    connector._application.bot.send_voice.assert_awaited_once()  # type: ignore[union-attr]


async def test_telegram_send_single_document_routes_to_send_document(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "report.pdf").write_bytes(b"x")

    await connector.telegram_send("", attachments=["/workspace/report.pdf"], chat_id="123")

    connector._application.bot.send_document.assert_awaited_once()  # type: ignore[union-attr]


async def test_telegram_send_multi_media_uses_send_media_group(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (ws / name).write_bytes(b"x")

    result = await connector.telegram_send(
        "two pics",
        attachments=["/workspace/a.jpg", "/workspace/b.jpg"],
        chat_id="123",
    )

    assert result == {"message_ids": [100, 101]}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.send_media_group.assert_awaited_once()
    bot.send_photo.assert_not_awaited()
    media = bot.send_media_group.call_args.kwargs["media"]
    assert len(media) == 2
    assert media[0].caption == "two pics"
    assert media[1].caption is None


async def test_telegram_send_attachment_traversal_rejected(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    with pytest.raises(ValueError, match="could not be resolved"):
        await connector.telegram_send("", attachments=["/workspace/../escape.jpg"], chat_id="123")


async def test_telegram_send_attachment_disallowed_root_raises(
    connector: TelegramConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector.telegram_send("", attachments=["/etc/passwd"], chat_id="123")


async def test_telegram_send_attachment_without_session_id_raises(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await connector.telegram_send("", attachments=["/workspace/x.jpg"], chat_id="123")


async def test_telegram_send_non_int_chat_id_raises(
    connector: TelegramConnector,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        await connector.telegram_send("hi", chat_id="not-a-number")
