"""Unit coverage for ``telegram_send``'s ``attachments`` parameter,
type-by-extension routing, and the caption-on-first quirk in
``send_media_group``.

The PTB ``Bot`` is mocked end-to-end; tests assert that the right
``send_*`` method was called with the right ``chat_id`` /
``caption`` / ``media`` arguments.

End-to-end tests go through the SDK's ``_invoke_tool`` dispatch
wrapper — that's the layer where ``SandboxPath`` resolution happens,
so connector-side tests must exercise it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aios_connector.base import ToolDescriptor

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


def _telegram_send_descriptor(connector: TelegramConnector) -> ToolDescriptor:
    descriptor = next(d for d in connector._tools if d.name == "telegram_send")
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


# Descriptor-level checks: the schema reaches the model with the right
# shape and the dispatch wrapper knows which arg to resolve.


def test_telegram_send_descriptor_records_sandbox_param(
    connector: TelegramConnector,
) -> None:
    descriptor = _telegram_send_descriptor(connector)
    assert descriptor.sandbox_params == {"attachments": "list"}


def test_telegram_send_schema_publishes_string_array_with_description(
    connector: TelegramConnector,
) -> None:
    descriptor = _telegram_send_descriptor(connector)
    attachments_schema = descriptor.input_schema["properties"]["attachments"]
    assert attachments_schema["type"] == "array"
    assert attachments_schema["items"]["type"] == "string"
    assert "/workspace/" in attachments_schema["items"]["description"]


# End-to-end tests via _invoke_tool (the dispatch wrapper resolves
# SandboxPath args before the tool body runs).


async def test_telegram_send_text_only(connector: TelegramConnector) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    descriptor = _telegram_send_descriptor(connector)
    result = _decode(await connector._invoke_tool(descriptor, {"text": "hello"}))
    assert result == {"message_id": 42}
    connector._application.bot.send_message.assert_awaited_once_with(  # type: ignore[union-attr]
        chat_id=123, text="hello"
    )


async def test_telegram_send_single_photo_routes_to_send_photo(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    descriptor = _telegram_send_descriptor(connector)
    result = _decode(
        await connector._invoke_tool(
            descriptor,
            {"text": "look", "attachments": ["/workspace/cat.jpg"]},
        )
    )

    assert result == {"message_id": 43}
    bot = connector._application.bot  # type: ignore[union-attr]
    bot.send_photo.assert_awaited_once()
    kwargs = bot.send_photo.call_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["photo"] == ws / "cat.jpg"
    assert kwargs["caption"] == "look"


async def test_telegram_send_single_voice_routes_to_send_voice(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "v.ogg").write_bytes(b"x")

    descriptor = _telegram_send_descriptor(connector)
    await connector._invoke_tool(descriptor, {"text": "", "attachments": ["/workspace/v.ogg"]})

    connector._application.bot.send_voice.assert_awaited_once()  # type: ignore[union-attr]


async def test_telegram_send_single_document_routes_to_send_document(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "report.pdf").write_bytes(b"x")

    descriptor = _telegram_send_descriptor(connector)
    await connector._invoke_tool(descriptor, {"text": "", "attachments": ["/workspace/report.pdf"]})

    connector._application.bot.send_document.assert_awaited_once()  # type: ignore[union-attr]


async def test_telegram_send_multi_media_uses_send_media_group(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (ws / name).write_bytes(b"x")

    descriptor = _telegram_send_descriptor(connector)
    result = _decode(
        await connector._invoke_tool(
            descriptor,
            {
                "text": "two pics",
                "attachments": ["/workspace/a.jpg", "/workspace/b.jpg"],
            },
        )
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
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    descriptor = _telegram_send_descriptor(connector)
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(
            descriptor,
            {"text": "", "attachments": ["/workspace/../escape.jpg"]},
        )


async def test_telegram_send_attachment_disallowed_root_raises(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, "sess-1")
    descriptor = _telegram_send_descriptor(connector)
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(descriptor, {"text": "", "attachments": ["/etc/passwd"]})


async def test_telegram_send_attachment_without_session_id_raises(
    connector: TelegramConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "0/123")
    _stub_session_id(connector, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    descriptor = _telegram_send_descriptor(connector)
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await connector._invoke_tool(descriptor, {"text": "", "attachments": ["/workspace/x.jpg"]})


async def test_telegram_send_non_int_chat_id_raises(
    connector: TelegramConnector,
) -> None:
    _stub_focal(connector, "0/not-a-number")
    descriptor = _telegram_send_descriptor(connector)
    with pytest.raises(ValueError, match="must be an integer"):
        await connector._invoke_tool(descriptor, {"text": "hi"})
