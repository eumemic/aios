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
from telegram import InputFile

from aios_telegram.connector import (
    TelegramConnector,
    _build_media_group,
    _classify,
)
from tests.conftest import BOT_ID, CONNECTION_ID


def _sent(message_id: int, chat_type: str = "private") -> MagicMock:
    """Build a mock PTB ``Message`` return value for a send_* call.

    ``telegram_send`` derives the result's ``chat_type`` from the
    returned message's ``.chat.type`` via the same ``_chat_kind`` helper
    inbound uses, so the mock must carry a realistic Telegram chat-type
    string ("private"/"group"/"supergroup"/"channel").
    """
    m = MagicMock(message_id=message_id)
    m.chat.type = chat_type
    return m


def test_classify_extensions() -> None:
    assert _classify(Path("/x/cat.jpg")) == "photo"
    assert _classify(Path("/x/CAT.JPEG")) == "photo"
    assert _classify(Path("/x/clip.mp4")) == "video"
    assert _classify(Path("/x/voice.ogg")) == "voice"
    assert _classify(Path("/x/song.mp3")) == "audio"
    assert _classify(Path("/x/report.pdf")) == "document"
    assert _classify(Path("/x/no-ext")) == "document"
    # .gif routes to send_animation, not send_photo: Telegram's Bot API
    # treats a GIF passed to sendPhoto as a static image and only
    # renders the first frame.  sendAnimation is the first-class
    # animated-image surface — clients play it inline.
    assert _classify(Path("/x/zoom.gif")) == "animation"


def test_build_media_group_caption_on_first_only(tmp_path: Path) -> None:
    paths = [tmp_path / f"{name}.jpg" for name in ("a", "b", "c")]
    for p in paths:
        p.write_bytes(b"x")
    items = _build_media_group(paths, caption="three pics")
    assert items[0].caption == "three pics"
    assert items[1].caption is None
    assert items[2].caption is None


def test_build_media_group_no_caption(tmp_path: Path) -> None:
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for p in paths:
        p.write_bytes(b"x")
    items = _build_media_group(paths, caption=None)
    assert all(item.caption is None for item in items)


def test_build_media_group_voice_demoted_to_document(tmp_path: Path) -> None:
    """Voice can't ride in a media group; falls back to document."""
    from telegram import InputMediaDocument

    voice = tmp_path / "v.ogg"
    voice.write_bytes(b"x")
    items = _build_media_group([voice], caption=None)
    assert isinstance(items[0], InputMediaDocument)


def test_telegram_connector_subclasses_http_connector() -> None:
    """Sanity: the SDK base picks up our @tool methods as a tool registry."""
    c = TelegramConnector()
    assert isinstance(c, HttpConnector)
    expected = {
        "telegram_send",
        "telegram_typing",
        "telegram_edit_message",
        "telegram_delete_message",
        "telegram_react",
    }
    assert expected <= set(c._tools)


def test_message_tools_are_fire_and_forget_but_typing_is_not() -> None:
    """Categorization guard (#1121 review): ``fire_and_forget`` marks a
    *terminal* delivery confirmation the model needn't react to, so the runtime
    skips the re-wake (closing the duplicate-send loop). ``telegram_typing`` is
    a *precursor* the model calls before slow work — it MUST stay a normal
    (waking) tool, or a typing-only turn settles idle and never sends.
    """
    tools = TelegramConnector()._tools
    assert tools["telegram_send"].fire_and_forget is True
    assert tools["telegram_react"].fire_and_forget is True
    assert tools["telegram_typing"].fire_and_forget is False


@pytest.fixture
def bot() -> Any:
    b = MagicMock()
    b.send_message = AsyncMock(return_value=_sent(42))
    b.send_photo = AsyncMock(return_value=_sent(43))
    b.send_voice = AsyncMock(return_value=_sent(44))
    b.send_video = AsyncMock(return_value=_sent(45))
    b.send_audio = AsyncMock(return_value=_sent(46))
    b.send_document = AsyncMock(return_value=_sent(47))
    b.send_media_group = AsyncMock(return_value=[_sent(100), _sent(101)])
    return b


# End-to-end tests calling the tool method directly with resolved paths.


async def test_telegram_send_text_only(connector: TelegramConnector, bot: Any) -> None:
    result = await connector.telegram_send(text="hello", chat_id="123", connection_id=CONNECTION_ID)
    # Result stamps the resolved focal channel + chat_type.  ``chat_type``
    # is derived from the chat the send API returned (``.chat.type ==
    # "private"``) via the same ``_chat_kind`` helper inbound uses → "dm".
    assert result == {
        "message_id": 42,
        "channel": f"telegram/{BOT_ID}/123",
        "chat_type": "dm",
    }
    bot.send_message.assert_awaited_once_with(chat_id=123, text="hello", parse_mode=None)


async def test_telegram_send_group_result_chat_type_group(
    connector: TelegramConnector, bot: Any
) -> None:
    """A legacy ``group`` chat reports ``chat_type == "group"`` — derived
    from the returned message's ``.chat.type``, matching inbound's
    ``_chat_kind`` mapping."""
    bot.send_message = AsyncMock(return_value=_sent(42, chat_type="group"))
    result = await connector.telegram_send(
        text="hello group", chat_id="-987654321", connection_id=CONNECTION_ID
    )
    assert result["channel"] == f"telegram/{BOT_ID}/-987654321"
    assert result["chat_type"] == "group"


async def test_telegram_send_supergroup_result_chat_type_supergroup(
    connector: TelegramConnector, bot: Any
) -> None:
    """Regression for #943: a supergroup (chat_id with the ``-100…``
    prefix) MUST report ``chat_type == "supergroup"`` outbound, exactly
    as inbound stamps it.  The old ``_chat_type_from_chat_id`` guessed
    from the chat_id sign and flattened every negative id to "group",
    so a supergroup recorded ``"supergroup"`` inbound but ``"group"``
    outbound — breaking inbound↔outbound correlation.  Deriving from the
    returned ``.chat.type`` via ``_chat_kind`` makes the two byte-identical.
    """
    bot.send_message = AsyncMock(return_value=_sent(42, chat_type="supergroup"))
    result = await connector.telegram_send(
        text="hello supergroup", chat_id="-1001234567890", connection_id=CONNECTION_ID
    )
    assert result["channel"] == f"telegram/{BOT_ID}/-1001234567890"
    assert result["chat_type"] == "supergroup"


async def test_telegram_send_reply_to_message_id_threads_through(
    connector: TelegramConnector, bot: Any
) -> None:
    """``reply_to_message_id`` flows into PTB so the message renders as
    a native Telegram reply quoting the parent (the client UI shows the
    quoted message inline above your text).
    """
    await connector.telegram_send(
        text="me too",
        chat_id="123",
        reply_to_message_id=99,
        connection_id=CONNECTION_ID,
    )
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["reply_to_message_id"] == 99


async def test_telegram_send_omits_reply_when_none(connector: TelegramConnector, bot: Any) -> None:
    """The default ``None`` is *not* forwarded to PTB — the kwarg is
    omitted entirely so the call surface stays clean and PTB picks
    its own default.
    """
    await connector.telegram_send(text="standalone", chat_id="123", connection_id=CONNECTION_ID)
    kwargs = bot.send_message.call_args.kwargs
    assert "reply_to_message_id" not in kwargs


async def test_telegram_send_single_photo_routes_to_send_photo(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    photo = tmp_path / "cat.jpg"
    photo.write_bytes(b"x")

    result = await connector.telegram_send(
        text="look", attachments=[photo], chat_id="123", connection_id=CONNECTION_ID
    )

    assert result == {
        "message_id": 43,
        "channel": f"telegram/{BOT_ID}/123",
        "chat_type": "dm",
    }
    bot.send_photo.assert_awaited_once()
    kwargs = bot.send_photo.call_args.kwargs
    assert kwargs["chat_id"] == 123
    # Wrapped in ``InputFile`` with the source filename so PTB's
    # multipart writer sets the right Content-Type from the
    # extension.  Raw bytes would land as application/octet-stream
    # and Telegram would render the attachment as a downloadable
    # blob instead of an inline image / animation / etc.
    photo_arg = kwargs["photo"]
    assert isinstance(photo_arg, InputFile)
    assert photo_arg.input_file_content == photo.read_bytes()
    assert photo_arg.filename == "cat.jpg"
    assert kwargs["caption"] == "look"


async def test_telegram_send_single_voice_routes_to_send_voice(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    voice = tmp_path / "v.ogg"
    voice.write_bytes(b"x")
    await connector.telegram_send(
        text="", attachments=[voice], chat_id="123", connection_id=CONNECTION_ID
    )
    bot.send_voice.assert_awaited_once()


async def test_telegram_send_wraps_bytes_in_input_file_with_filename(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    """Pin the full upload-shape contract:

    1. **Never a Path** — python-telegram-bot's HTTPX layer JSON-
       serializes the body; ``pathlib.Path`` raises
       ``TypeError('Object of type PosixPath is not JSON
       serializable')``.  Surfaced live in PR 8 smoke as an outbound
       image crash.
    2. **Never raw bytes** — bytes survive JSON, but PTB defaults the
       ``InputFile`` filename to ``"application.octet-stream"`` and
       Telegram renders the attachment as a downloadable blob
       (``Content-Type: application/octet-stream``) instead of an
       inline image / animation / audio.  Caught live on the
       follow-on smoke (Tom: "don't send an octet-stream") after the
       ``.gif`` extension already routed correctly to
       ``send_animation`` — without the filename hint, Telegram
       still couldn't determine the type.

    Right shape: ``InputFile(bytes, filename=host_path.name)`` so
    PTB's multipart writer sets ``Content-Type`` from the extension.
    """
    photo = tmp_path / "img.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\nbytes-canary")
    await connector.telegram_send(
        text="", attachments=[photo], chat_id="123", connection_id=CONNECTION_ID
    )
    photo_arg = bot.send_photo.call_args.kwargs["photo"]
    assert isinstance(photo_arg, InputFile), (
        f"PTB media kwarg should be InputFile so the filename drives "
        f"Content-Type, got {type(photo_arg).__name__}: {photo_arg!r}"
    )
    assert photo_arg.input_file_content == b"\x89PNG\r\n\x1a\nbytes-canary"
    assert photo_arg.filename == "img.png"


async def test_telegram_send_single_document_routes_to_send_document(
    connector: TelegramConnector,
    bot: Any,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"x")
    await connector.telegram_send(
        text="", attachments=[doc], chat_id="123", connection_id=CONNECTION_ID
    )
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
        text="two pics",
        attachments=[a, b_path],
        chat_id="123",
        connection_id=CONNECTION_ID,
    )

    assert result == {
        "message_ids": [100, 101],
        "channel": f"telegram/{BOT_ID}/123",
        "chat_type": "dm",
    }
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
        await connector.telegram_send(
            text="hi", chat_id="not-a-number", connection_id=CONNECTION_ID
        )


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
    ws = (tmp_path / "acc-1" / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")
    connector._client = AsyncMock()

    async def _noop_result(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(connector, "_post_tool_result", _noop_result)

    await connector.dispatch_call(
        {
            "connection_id": CONNECTION_ID,
            "tool_call_id": "c1",
            "session_id": "sess-1",
            "name": "telegram_send",
            "arguments": json.dumps({"text": "look", "attachments": ["/workspace/cat.jpg"]}),
            "focal_channel": "telegram/0/123",
            "workspace_path": str(ws),
        }
    )

    # The SDK resolves the ``SandboxPath`` to a real ``Path`` for the
    # tool method; we then read it up-front (see ``_read_for_upload``)
    # so PTB gets bytes rather than a ``PosixPath`` it can't JSON-encode.
    photo_arg: Any = bot.send_photo.call_args.kwargs["photo"]
    assert isinstance(photo_arg, InputFile)
    assert photo_arg.input_file_content == (ws / "cat.jpg").read_bytes()
