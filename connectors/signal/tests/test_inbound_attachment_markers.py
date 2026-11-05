"""#2551: undelivered inbound attachments reach the agent as markers.

Over-cap files and failed signal-cli downloads (0-byte file, envelope
``size`` > 0) used to vanish into a log line, leaving the agent with an
empty event.  Each must now surface as an ``[attachment not delivered:
...]`` line in the inbound content, never counted as an attachment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog

from aios_signal import connector as connector_mod
from aios_signal.connector import SignalConnector, _SignalConnectionState
from aios_signal.parse import parse_envelope
from tests.conftest import ALICE_UUID, BOT_UUID, CONNECTION_ID, FIXTURES_DIR, PHONE

ISSUE_NAME = "Online FSOT Practice Test.pdf"
ISSUE_MARKER = (
    "[attachment not delivered: 'Online FSOT Practice Test.pdf', "
    "application/pdf, 7.1 MiB > 5 MiB limit]"
)


def _state() -> _SignalConnectionState:
    return _SignalConnectionState(phone=PHONE, bot_uuid=BOT_UUID, contact_names={}, groups=[])


def _sparse(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def _att(path: Path, name: str, ctype: str, size: Any) -> dict[str, Any]:
    return {
        "contentType": ctype,
        "filename": name,
        "id": path.name,
        "size": size,
        "file": str(path),
    }


def _envelope(attachments: list[dict[str, Any]], text: str | None = None) -> dict[str, Any]:
    dm: dict[str, Any] = {"timestamp": 1700000000000, "attachments": attachments}
    if text is not None:
        dm["message"] = text
    return {
        "sourceUuid": ALICE_UUID,
        "sourceName": "Alice",
        "timestamp": 1700000000000,
        "dataMessage": dm,
    }


async def _run(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, envelope: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    emit = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(connector, "emit_inbound", emit)
    monkeypatch.setattr(connector, "_send_read_receipt", AsyncMock())
    with structlog.testing.capture_logs() as logs:
        await connector._handle_envelope(CONNECTION_ID, _state(), envelope)
    emit.assert_awaited_once()
    return emit.await_args.kwargs, logs


def _count(kwargs: dict[str, Any]) -> int:
    return len(kwargs["attachments"] or [])


def _not_delivered(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == "signal.inbound.attachment_not_delivered"]


async def test_over_cap_becomes_marker(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    big = _sparse(tmp_path / "att" / "big.pdf", 7441443)
    kwargs, logs = await _run(
        connector, monkeypatch, _envelope([_att(big, ISSUE_NAME, "application/pdf", 7441443)])
    )
    assert kwargs["content"] == ISSUE_MARKER
    assert _count(kwargs) == 0
    assert any(e["event"] == "signal.inbound.attachment_rejected" for e in logs)
    [nd] = _not_delivered(logs)
    assert nd["reason"] == "over_cap"


async def test_zero_byte_failed_download_is_marker_not_delivery(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Seat red test: 0-byte host file, envelope size 163815482."""
    assert connector_mod.SIGNAL_ZERO_BYTE_DETECTION is False
    empty = _sparse(tmp_path / "att" / "basil.pdf", 0)
    kwargs, logs = await _run(
        connector,
        monkeypatch,
        _envelope([_att(empty, "basil.pdf", "application/pdf", 163815482)]),
    )
    assert kwargs["content"] == (
        "[attachment not delivered: 'basil.pdf', application/pdf, "
        "156.2 MiB > 5 MiB limit: download failed]"
    )
    assert kwargs["attachments"] is None
    [nd] = _not_delivered(logs)
    assert nd["reason"] == "download_failed"


async def test_zero_byte_with_string_size_and_garbage_size(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = _sparse(tmp_path / "att" / "a.pdf", 0)
    kwargs, _ = await _run(
        connector, monkeypatch, _envelope([_att(empty, "a.pdf", "application/pdf", "163815482")])
    )
    assert kwargs["content"].endswith("156.2 MiB > 5 MiB limit: download failed]")
    # Unknown declared size: R2 cannot fire, the file goes through today's path.
    kwargs, _ = await _run(
        connector, monkeypatch, _envelope([_att(empty, "a.pdf", "application/pdf", "abc")])
    )
    assert kwargs["content"] == ""
    assert _count(kwargs) == 1


async def test_under_cap_unchanged(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small = tmp_path / "att" / "s.pdf"
    small.parent.mkdir(parents=True)
    small.write_bytes(b"%PDF-1.4 tiny")
    kwargs, logs = await _run(
        connector,
        monkeypatch,
        _envelope([_att(small, "s.pdf", "application/pdf", 13)], text="hello"),
    )
    assert kwargs["content"] == "hello"
    assert kwargs["attachments"] == [("s.pdf", b"%PDF-1.4 tiny", "application/pdf")]
    assert _not_delivered(logs) == []


async def test_mixed_and_text_first(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    big = _sparse(tmp_path / "att" / "big.pdf", 7441443)
    small = tmp_path / "att" / "s.txt"
    small.write_bytes(b"ok")
    empty = _sparse(tmp_path / "att" / "e.pdf", 0)
    kwargs, _ = await _run(
        connector,
        monkeypatch,
        _envelope(
            [
                _att(big, ISSUE_NAME, "application/pdf", 7441443),
                _att(small, "s.txt", "text/plain", 2),
                _att(empty, "e.pdf", "application/pdf", 1000),
            ],
            text="see files",
        ),
    )
    assert kwargs["content"] == (
        "see files\n"
        + ISSUE_MARKER
        + "\n[attachment not delivered: 'e.pdf', application/pdf, 1000 B: download failed]"
    )
    assert kwargs["attachments"] == [("s.txt", b"ok", "text/plain")]


async def test_text_plus_over_cap(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    big = _sparse(tmp_path / "att" / "big.pdf", 7441443)
    kwargs, _ = await _run(
        connector,
        monkeypatch,
        _envelope([_att(big, ISSUE_NAME, "application/pdf", 7441443)], text="here"),
    )
    assert kwargs["content"] == "here\n" + ISSUE_MARKER


async def test_partial_file_delivered_with_flag_false(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    part = tmp_path / "att" / "p.bin"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"0123456789")
    kwargs, _ = await _run(
        connector,
        monkeypatch,
        _envelope([_att(part, "p.bin", "application/octet-stream", 1000)]),
    )
    assert kwargs["content"] == ""
    assert _count(kwargs) == 1


async def test_generic_rejection_marker(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-size AttachmentError (here: not a regular file) → ``rejected``."""
    d = tmp_path / "att" / "dir.pdf"
    d.mkdir(parents=True)
    kwargs, logs = await _run(
        connector, monkeypatch, _envelope([_att(d, "dir.pdf", "application/pdf", 100)])
    )
    assert kwargs["content"].startswith("[attachment not delivered: 'dir.pdf', application/pdf")
    assert kwargs["content"].endswith(": rejected]")
    [nd] = _not_delivered(logs)
    assert nd["reason"] == "rejected"


async def test_stat_oserror_does_not_escape(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guarded pre-stat raises (file never downloaded); the message is
    still emitted, the sibling attachment is delivered, nothing escapes."""
    missing = tmp_path / "att" / "gone.pdf"
    ok = tmp_path / "att" / "ok.txt"
    ok.parent.mkdir(parents=True)
    ok.write_bytes(b"ok")
    kwargs, _ = await _run(
        connector,
        monkeypatch,
        _envelope(
            [_att(missing, "gone.pdf", "application/pdf", 5), _att(ok, "ok.txt", "text/plain", 2)]
        ),
    )
    assert kwargs["attachments"] == [("ok.txt", b"ok", "text/plain")]
    assert (
        kwargs["content"]
        == "[attachment not delivered: 'gone.pdf', application/pdf, 5 B: rejected]"
    )


async def test_r2_leaves_file_like_over_cap_path(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither the over-cap nor the R2 path deletes the signal-cli file,
    and reprocessing the same envelope still does not deliver it."""
    big = _sparse(tmp_path / "att" / "big.pdf", 7441443)
    empty = _sparse(tmp_path / "att" / "e.pdf", 0)
    env = _envelope(
        [
            _att(big, "big.pdf", "application/pdf", 7441443),
            _att(empty, "e.pdf", "application/pdf", 163815482),
        ]
    )
    for _ in range(2):
        kwargs, _ = await _run(connector, monkeypatch, env)
        assert kwargs["attachments"] is None
        assert len(kwargs["content"].splitlines()) == 2
    assert big.exists() and empty.exists()


async def test_marker_only_message_is_emitted(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    big = _sparse(tmp_path / "att" / "big.pdf", 7441443)
    kwargs, _ = await _run(
        connector, monkeypatch, _envelope([_att(big, ISSUE_NAME, "application/pdf", 7441443)])
    )
    assert kwargs["content"]
    assert kwargs["event_id"] == f"signal-{ALICE_UUID}-1700000000000"


def test_parse_carries_coerced_size() -> None:
    env = json.loads((FIXTURES_DIR / "attachment_only.json").read_text())
    msg = parse_envelope(env, bot_account_uuid=BOT_UUID)
    assert msg is not None
    assert msg.attachments[0].size == 102400
    env["dataMessage"]["attachments"][0]["size"] = "garbage"
    msg = parse_envelope(env, bot_account_uuid=BOT_UUID)
    assert msg is not None and msg.attachments[0].size is None
