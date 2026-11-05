"""#2551: undelivered Telegram inbound attachments reach the agent as markers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from telegram.error import TelegramError

from aios_telegram.connector import TelegramConnector, _TelegramConnectionState
from aios_telegram.parse import Attachment, InboundMessage

pytestmark = pytest.mark.asyncio

DOSSIER = "J.RA.5.102244 Dosier Macellinus.pdf"
DOSSIER_MARKER = (
    "[attachment not delivered: 'J.RA.5.102244 Dosier Macellinus.pdf', "
    "application/pdf, 7.8 MiB > 5 MiB limit]"
)


@pytest.fixture
def isolated_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _state(sizes: dict[str, int | bytes]) -> _TelegramConnectionState:
    """Fake state whose download writes ``sizes[file_id]`` (int → sparse
    file of that size; bytes → literal content).  A missing id makes
    ``get_file`` raise a swallowed :class:`TelegramError`."""
    state = MagicMock(spec=_TelegramConnectionState)
    state.application = MagicMock()
    state.bot_id = 42

    async def get_file(file_id: str) -> Any:
        if file_id not in sizes:
            raise TelegramError("File is too big")
        payload = sizes[file_id]

        async def download_to_drive(custom_path: Path) -> Path:
            if isinstance(payload, bytes):
                Path(custom_path).write_bytes(payload)  # noqa: ASYNC240
            else:
                with open(custom_path, "wb") as f:  # noqa: ASYNC230
                    f.truncate(payload)
            return Path(custom_path)

        fake = MagicMock()
        fake.download_to_drive = AsyncMock(side_effect=download_to_drive)
        return fake

    state.application.bot.get_file = AsyncMock(side_effect=get_file)
    return state


def _connector() -> TelegramConnector:
    c = TelegramConnector.__new__(TelegramConnector)
    c.state = {}
    return c


async def _collect(
    state: _TelegramConnectionState, atts: tuple[Attachment, ...]
) -> tuple[list[tuple[str, bytes, str]] | None, list[str], list[dict[str, Any]]]:
    with structlog.testing.capture_logs() as logs:
        tuples, markers = await _connector()._collect_attachments(state, atts)
    return tuples, markers, logs


def _reasons(logs: list[dict[str, Any]]) -> list[str]:
    return [e["reason"] for e in logs if e["event"] == "telegram.inbound.attachment_not_delivered"]


async def test_over_cap_marker_and_no_tempfile(isolated_tmp: Path) -> None:
    att = Attachment(file_id="big", content_type="application/pdf", filename=DOSSIER)
    tuples, markers, logs = await _collect(_state({"big": 8153223}), (att,))
    assert tuples is None
    assert markers == [DOSSIER_MARKER]
    assert any(e["event"] == "telegram.inbound.attachment_rejected" for e in logs)
    assert _reasons(logs) == ["over_cap"]
    assert list(isolated_tmp.glob("aios-telegram-*")) == []  # noqa: ASYNC240


async def test_under_cap_unchanged(isolated_tmp: Path) -> None:
    att = Attachment(file_id="s", content_type="application/pdf", filename="s.pdf", file_size=3)
    tuples, markers, _ = await _collect(_state({"s": b"abc"}), (att,))
    assert tuples == [("s.pdf", b"abc", "application/pdf")]
    assert markers == []


async def test_mixed(isolated_tmp: Path) -> None:
    atts = (
        Attachment(file_id="big", content_type="application/pdf", filename=DOSSIER),
        Attachment(file_id="s", content_type="text/plain", filename="s.txt"),
    )
    tuples, markers, _ = await _collect(_state({"big": 8153223, "s": b"ok"}), atts)
    assert tuples == [("s.txt", b"ok", "text/plain")]
    assert markers == [DOSSIER_MARKER]


async def test_zero_byte_with_declared_size_is_download_failed(isolated_tmp: Path) -> None:
    att = Attachment(
        file_id="z", content_type="application/pdf", filename="z.pdf", file_size=163815482
    )
    tuples, markers, logs = await _collect(_state({"z": b""}), (att,))
    assert tuples is None
    assert markers == [
        "[attachment not delivered: 'z.pdf', application/pdf, "
        "156.2 MiB > 5 MiB limit: download failed]"
    ]
    assert _reasons(logs) == ["download_failed"]
    assert list(isolated_tmp.glob("aios-telegram-*")) == []  # noqa: ASYNC240


async def test_zero_byte_without_declared_size_keeps_todays_behaviour(
    isolated_tmp: Path,
) -> None:
    att = Attachment(file_id="z", content_type="application/pdf", filename="z.pdf")
    tuples, markers, _ = await _collect(_state({"z": b""}), (att,))
    assert tuples == [("z.pdf", b"", "application/pdf")]
    assert markers == []


async def test_swallowed_download_failure_becomes_marker(isolated_tmp: Path) -> None:
    att = Attachment(
        file_id="missing", content_type="application/pdf", filename="m.pdf", file_size=34811382
    )
    tuples, markers, logs = await _collect(_state({}), (att,))
    assert tuples is None
    assert markers == [
        "[attachment not delivered: 'm.pdf', application/pdf, "
        "33.2 MiB > 5 MiB limit: download failed]"
    ]
    assert _reasons(logs) == ["download_failed"]
    assert list(isolated_tmp.glob("aios-telegram-*")) == []  # noqa: ASYNC240


async def test_reraised_failure_still_propagates(isolated_tmp: Path) -> None:
    state = _state({})
    state.application.bot.get_file = AsyncMock(side_effect=RuntimeError("boom"))
    att = Attachment(file_id="x", content_type="application/pdf", filename="x.pdf")
    with pytest.raises(RuntimeError):
        await _connector()._collect_attachments(state, (att,))


async def test_generic_rejection(isolated_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios_connector_http import AttachmentError
    from aios_connector_http.sandbox import Attachment as SDKAttachment

    def reject(self: SDKAttachment) -> dict[str, Any]:
        raise AttachmentError("nope")

    monkeypatch.setattr(SDKAttachment, "as_params", reject)
    att = Attachment(file_id="s", content_type="application/pdf", filename="s.pdf")
    tuples, markers, logs = await _collect(_state({"s": b"abc"}), (att,))
    assert tuples is None
    assert markers == ["[attachment not delivered: 's.pdf', application/pdf, 3 B: rejected]"]
    assert _reasons(logs) == ["rejected"]


async def test_stat_oserror_does_not_escape(
    isolated_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_stat = Path.stat
    calls = {"n": 0}

    def flaky(self: Path, *a: Any, **k: Any) -> Any:
        if "aios-telegram-" in str(self) and self.suffix == ".pdf" and calls["n"] == 0:
            calls["n"] += 1
            raise PermissionError("denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky)
    atts = (
        Attachment(file_id="a", content_type="application/pdf", filename="a.pdf", file_size=3),
        Attachment(file_id="b", content_type="text/plain", filename="b.txt"),
    )
    tuples, markers, _ = await _collect(_state({"a": b"abc", "b": b"ok"}), atts)
    assert tuples == [("a.pdf", b"abc", "application/pdf"), ("b.txt", b"ok", "text/plain")]
    assert markers == []


def _msg(text: str, atts: tuple[Attachment, ...]) -> InboundMessage:
    return InboundMessage(
        chat_id=-100,
        chat_kind="group",
        chat_name="Metals",
        message_id=7,
        sender_id=1,
        sender_name="David",
        text=text,
        timestamp_ms=1700000000000,
        attachments=atts,
        mentions=(),
        reply=None,
    )


async def _emit(text: str, atts: tuple[Attachment, ...], sizes: dict[str, Any]) -> Any:
    c = _connector()
    emit = AsyncMock(return_value={"ok": True})
    c.emit_inbound = emit  # type: ignore[method-assign]
    c.focal_channel = MagicMock(return_value="telegram/42/-100")  # type: ignore[method-assign]
    await c._emit_message("conn", _state(sizes), _msg(text, atts))
    emit.assert_awaited_once()
    return emit.await_args.kwargs


async def test_emit_marker_only_message(isolated_tmp: Path) -> None:
    att = Attachment(file_id="big", content_type="application/pdf", filename=DOSSIER)
    kwargs = await _emit("", (att,), {"big": 8153223})
    assert kwargs["content"] == DOSSIER_MARKER
    assert kwargs["attachments"] is None
    assert kwargs["event_id"] == "telegram--100-7"


async def test_emit_text_plus_over_cap(isolated_tmp: Path) -> None:
    att = Attachment(file_id="big", content_type="application/pdf", filename=DOSSIER)
    kwargs = await _emit("here", (att,), {"big": 8153223})
    assert kwargs["content"] == "here\n" + DOSSIER_MARKER


async def test_event_id_independent_of_markers(isolated_tmp: Path) -> None:
    big = Attachment(file_id="big", content_type="application/pdf", filename=DOSSIER)
    small = Attachment(file_id="big", content_type="application/pdf", filename="s.pdf")
    with_marker = await _emit("x", (big,), {"big": 8153223})
    without = await _emit("x", (small,), {"big": b"abc"})
    assert with_marker["event_id"] == without["event_id"]
