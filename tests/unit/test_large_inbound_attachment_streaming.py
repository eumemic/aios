"""aios#2563 acceptance: a 9 MiB PDF reaches ``_attachments/<sess>/`` byte-identical.

Drives the real connector code (Signal ``_collect_attachments`` /
Telegram ``_collect_attachments``) → the real SDK ``emit_inbound`` → an
in-process ASGI app whose handler takes ``list[UploadFile]`` exactly as
``api/routers/connectors.py`` does and hands them to the real
:func:`stage_inbound_attachments`.  Only auth / DB / session routing are
elided.  Also asserts (via a spy on httpx ``files=``) that the attachment
payload is a streamed file object, never a ``bytes`` blob, and that a
6 MiB PNG is still dropped with a marker naming the 5 MiB image cap.
"""

from __future__ import annotations

import hashlib
import random
import tempfile
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aios_signal.config import Settings as SignalSettings
from aios_signal.connector import SignalConnector, _SignalConnectionState
from aios_telegram.connector import TelegramConnector, _TelegramConnectionState
from aios_telegram.parse import Attachment as TgAttachment
from aios_telegram.parse import InboundMessage as TgInboundMessage
from fastapi import FastAPI, File, Form, UploadFile

from aios.config import get_settings
from aios.services.attachment_staging import InboundAttachment, stage_inbound_attachments

MiB = 1024 * 1024
SESSION_ID = "sess_2563"
ALICE = "11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOT = "99999999-8888-7777-6666-555555555555"


@pytest.fixture(autouse=True)
def _default_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIOS_CONNECTOR_MAX_ATTACHMENT_BYTES", raising=False)


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setattr(get_settings(), "workspace_root", root)
    return root


def _pdf_fixture(path: Path, size: int = 9 * MiB) -> str:
    """Deterministic ``%PDF-`` + seeded bytes; returns its sha256."""
    rng = random.Random(2563)
    body = b"%PDF-1.7\n" + rng.randbytes(size - 9)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _png_fixture(path: Path, size: int = 6 * MiB) -> None:
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.truncate(size)


def _staging_app(connector_name: str, staged: list[dict[str, Any]]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/connectors/runtime/inbound")
    async def inbound(
        connection_id: Annotated[str, Form()],
        event_id: Annotated[str, Form()],
        chat_id: Annotated[str, Form()],
        content: Annotated[str, Form()] = "",
        attachments: Annotated[list[UploadFile] | None, File()] = None,
    ) -> dict[str, Any]:
        records, _ = await stage_inbound_attachments(
            session_id=SESSION_ID,
            connector_name=connector_name,
            event_id=event_id,
            attachments=[
                InboundAttachment(
                    stream=u,
                    filename=u.filename or "",
                    content_type=u.content_type or "application/octet-stream",
                )
                for u in (attachments or [])
            ],
        )
        staged.append({"content": content, "records": records})
        return {"appended_event_id": "evt_1"}

    return app


class _SpyClient(httpx.AsyncClient):
    """Records the type of every attachment payload handed to ``files=``."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.payload_types: list[type] = []

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        for field, value in kwargs.get("files") or []:
            if field == "attachments":
                self.payload_types.append(type(value[1]))
        return await super().post(url, **kwargs)


def _wire(connector: Any, app: FastAPI) -> _SpyClient:
    http = _SpyClient(transport=httpx.ASGITransport(app=app), base_url="http://aios")
    client = MagicMock()
    client.get_async_httpx_client.return_value = http
    connector._client = client
    return http


def _staged_file(workspace_root: Path, connector_name: str, record: dict[str, Any]) -> Path:
    name = str(record["in_sandbox_path"]).rsplit("/", 1)[1]
    return workspace_root / "_attachments" / SESSION_ID / connector_name / name


# ── Signal ───────────────────────────────────────────────────────────


def _signal_connector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SignalConnector:
    # HttpConnector reads these at __init__; never contacted (ASGI transport).
    monkeypatch.setenv("AIOS_URL", "http://aios")
    monkeypatch.setenv("AIOS_RUNTIME_TOKEN", "aios_runtime_test")
    c = SignalConnector(SignalSettings(config_dir=tmp_path / "cfg", cli_bin="/bin/true"))
    c._send_read_receipt = AsyncMock()  # type: ignore[method-assign]
    return c


def _signal_envelope(atts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sourceUuid": ALICE,
        "sourceName": "David",
        "timestamp": 1700000000000,
        "dataMessage": {"timestamp": 1700000000000, "attachments": atts},
    }


def _signal_state() -> _SignalConnectionState:
    return _SignalConnectionState(phone="+15550001", bot_uuid=BOT, contact_names={}, groups=[])


async def test_signal_9_mib_pdf_staged_byte_identical(
    tmp_path: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "signal-data" / "attachments" / "abc.pdf"
    pdf.parent.mkdir(parents=True)
    sha = _pdf_fixture(pdf)
    staged: list[dict[str, Any]] = []
    connector = _signal_connector(tmp_path, monkeypatch)
    http = _wire(connector, _staging_app("signal", staged))
    env = _signal_envelope(
        [
            {
                "contentType": "application/pdf",
                "filename": "Kam-exercises.pdf",
                "id": "abc.pdf",
                "size": 9 * MiB,
                "file": str(pdf),
            }
        ]
    )
    await connector._handle_envelope("conn_1", _signal_state(), env)
    await http.aclose()

    [call] = staged
    assert call["content"] == ""  # no drop marker
    [record] = call["records"]
    assert record["size"] == 9 * MiB
    out = _staged_file(workspace_root, "signal", record)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == sha
    # Streamed as a file object, never a whole-file ``bytes`` blob.
    assert http.payload_types and all(t is not bytes for t in http.payload_types)
    assert all(hasattr(t, "read") for t in http.payload_types)
    # signal-cli's original is left in place (retention is signal-cli's).
    assert pdf.exists()


async def test_signal_handles_closed_even_when_emit_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "a.pdf"
    _pdf_fixture(pdf, 1 * MiB)
    connector = _signal_connector(tmp_path, monkeypatch)
    seen: list[Any] = []

    async def boom(**kw: Any) -> Any:
        seen.extend(h for _, h, _ in kw["attachments"])
        raise RuntimeError("POST failed mid-stream")

    monkeypatch.setattr(connector, "emit_inbound", boom)
    env = _signal_envelope(
        [{"contentType": "application/pdf", "filename": "a.pdf", "size": 1 * MiB, "file": str(pdf)}]
    )
    with pytest.raises(RuntimeError):
        await connector._handle_envelope("conn_1", _signal_state(), env)
    assert seen and all(h.closed for h in seen)


async def test_signal_6_mib_png_dropped_with_5_mib_marker(
    tmp_path: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    png = tmp_path / "pic.png"
    _png_fixture(png)
    staged: list[dict[str, Any]] = []
    connector = _signal_connector(tmp_path, monkeypatch)
    http = _wire(connector, _staging_app("signal", staged))
    env = _signal_envelope(
        [{"contentType": "image/png", "filename": "pic.png", "size": 6 * MiB, "file": str(png)}]
    )
    await connector._handle_envelope("conn_1", _signal_state(), env)
    await http.aclose()
    [call] = staged
    assert call["content"] == (
        "[attachment not delivered: 'pic.png', image/png, 6 MiB > 5 MiB limit]"
    )
    assert call["records"] == []


async def test_signal_above_signal_cli_ceiling_gets_marker(
    tmp_path: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """signal-cli's ``Response exceeds max size!`` leaves a 0-byte file with
    the envelope's declared size: the #2551 download-failed marker fires."""
    empty = tmp_path / "basil.pdf"
    empty.write_bytes(b"")
    staged: list[dict[str, Any]] = []
    connector = _signal_connector(tmp_path, monkeypatch)
    http = _wire(connector, _staging_app("signal", staged))
    env = _signal_envelope(
        [
            {
                "contentType": "application/pdf",
                "filename": "basil.pdf",
                "size": 163815482,
                "file": str(empty),
            }
        ]
    )
    await connector._handle_envelope("conn_1", _signal_state(), env)
    await http.aclose()
    [call] = staged
    assert call["content"] == (
        "[attachment not delivered: 'basil.pdf', application/pdf, "
        "156.2 MiB > 50 MiB limit: download failed]"
    )
    assert call["records"] == []


# ── Telegram ─────────────────────────────────────────────────────────


def _tg_state(payload_path: Path) -> _TelegramConnectionState:
    state = MagicMock(spec=_TelegramConnectionState)
    state.application = MagicMock()
    state.bot_id = 42

    async def download_to_drive(custom_path: Path) -> Path:
        Path(custom_path).write_bytes(payload_path.read_bytes())  # noqa: ASYNC240
        return Path(custom_path)

    fake = MagicMock()
    fake.download_to_drive = AsyncMock(side_effect=download_to_drive)
    state.application.bot.get_file = AsyncMock(return_value=fake)
    return state


def _tg_connector() -> TelegramConnector:
    c = TelegramConnector.__new__(TelegramConnector)
    c.state = {}
    c.focal_channel = MagicMock(return_value="telegram/42/-100")  # type: ignore[method-assign]
    return c


def _tg_msg(att: TgAttachment) -> TgInboundMessage:
    return TgInboundMessage(
        chat_id=-100,
        chat_kind="group",
        chat_name="Metals",
        message_id=7,
        sender_id=1,
        sender_name="David",
        text="",
        timestamp_ms=1700000000000,
        attachments=(att,),
        mentions=(),
        reply=None,
    )


@pytest.fixture
def isolated_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "tg-tmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


async def test_telegram_9_mib_pdf_staged_byte_identical(
    tmp_path: Path, workspace_root: Path, isolated_tmp: Path
) -> None:
    src = tmp_path / "src.pdf"
    sha = _pdf_fixture(src)
    staged: list[dict[str, Any]] = []
    connector = _tg_connector()
    http = _wire(connector, _staging_app("telegram", staged))
    att = TgAttachment(
        file_id="f1", content_type="application/pdf", filename="paper.pdf", file_size=9 * MiB
    )
    await connector._emit_message("conn_1", _tg_state(src), _tg_msg(att))
    await http.aclose()

    [call] = staged
    assert call["content"] == ""
    [record] = call["records"]
    out = _staged_file(workspace_root, "telegram", record)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == sha
    assert http.payload_types and all(t is not bytes for t in http.payload_types)
    assert list(isolated_tmp.glob("aios-telegram-*")) == []  # noqa: ASYNC240


@pytest.mark.parametrize("raises", [False, True])
async def test_telegram_handles_closed_and_no_tempfile(
    tmp_path: Path, isolated_tmp: Path, raises: bool
) -> None:
    src = tmp_path / "src.pdf"
    _pdf_fixture(src, 1 * MiB)
    connector = _tg_connector()
    seen: list[Any] = []

    async def emit(**kw: Any) -> Any:
        seen.extend(h for _, h, _ in kw["attachments"])
        assert all(not h.closed for h in seen)  # still open during the POST
        if raises:
            raise RuntimeError("POST failed mid-stream")
        return {"ok": True}

    connector.emit_inbound = emit  # type: ignore[method-assign]
    att = TgAttachment(file_id="f1", content_type="application/pdf", filename="p.pdf")
    if raises:
        with pytest.raises(RuntimeError):
            await connector._emit_message("conn_1", _tg_state(src), _tg_msg(att))
    else:
        await connector._emit_message("conn_1", _tg_state(src), _tg_msg(att))
    assert seen and all(h.closed for h in seen)
    assert list(isolated_tmp.glob("aios-telegram-*")) == []  # noqa: ASYNC240


async def test_telegram_6_mib_png_dropped_with_5_mib_marker(
    tmp_path: Path, workspace_root: Path, isolated_tmp: Path
) -> None:
    src = tmp_path / "pic.png"
    _png_fixture(src)
    staged: list[dict[str, Any]] = []
    connector = _tg_connector()
    http = _wire(connector, _staging_app("telegram", staged))
    att = TgAttachment(file_id="f1", content_type="image/png", filename="pic.png")
    await connector._emit_message("conn_1", _tg_state(src), _tg_msg(att))
    await http.aclose()
    [call] = staged
    assert call["content"] == (
        "[attachment not delivered: 'pic.png', image/png, 6 MiB > 5 MiB limit]"
    )
    assert call["records"] == []
