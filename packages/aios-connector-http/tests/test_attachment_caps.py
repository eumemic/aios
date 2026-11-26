"""Per-type, runtime-configurable inbound attachment caps + streamed upload (aios#2563).

Documents (every non-``image/*`` type) default to 50 MiB and are
overridable per connector process via ``AIOS_CONNECTOR_MAX_ATTACHMENT_BYTES``;
images keep 5 MiB.  ``emit_inbound`` accepts an open file handle and
streams it rather than holding the whole file in memory.
"""

from __future__ import annotations

import hashlib
import logging
import random
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from aios_connector_http import (
    ATTACHMENT_CAP_ENV,
    MAX_ATTACHMENT_BYTES,
    MAX_IMAGE_ATTACHMENT_BYTES,
    HttpConnector,
    document_attachment_cap,
    effective_attachment_cap,
    sandbox,
)
from aios_connector_http.sandbox import (
    Attachment,
    AttachmentError,
    drop_attachment,
    is_over_cap,
    rejection_marker,
)

MiB = 1024 * 1024


@pytest.fixture(autouse=True)
def _no_cap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ATTACHMENT_CAP_ENV, raising=False)


def _sparse(path: Path, size: int, header: bytes = b"") -> Path:
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(size)
    return path


# ── caps ──────────────────────────────────────────────────────────────


def test_defaults() -> None:
    assert ATTACHMENT_CAP_ENV == "AIOS_CONNECTOR_MAX_ATTACHMENT_BYTES"
    assert MAX_ATTACHMENT_BYTES == 50 * MiB
    assert MAX_IMAGE_ATTACHMENT_BYTES == 5 * MiB
    assert document_attachment_cap() == 50 * MiB


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("application/pdf", 50 * MiB),
        ("video/mp4", 50 * MiB),
        (None, 50 * MiB),
        ("image/png", 5 * MiB),
        ("IMAGE/JPEG", 5 * MiB),
        ("image/webp; charset=binary", 5 * MiB),
        (" Image/Gif ", 5 * MiB),
        ("application/pdf; name=x", 50 * MiB),
    ],
)
def test_effective_cap_per_type(content_type: str | None, expected: int) -> None:
    assert effective_attachment_cap(content_type) == expected


def test_env_override_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set AFTER import: the cap must still follow.
    monkeypatch.setenv(ATTACHMENT_CAP_ENV, str(64 * MiB))
    assert document_attachment_cap() == 64 * MiB
    assert effective_attachment_cap("application/pdf") == 64 * MiB
    assert effective_attachment_cap("image/png") == 5 * MiB  # image cap is fixed
    assert is_over_cap(60 * MiB, "application/pdf") is False
    assert is_over_cap(65 * MiB, "application/pdf") is True
    assert rejection_marker("x.pdf", "application/pdf", 65 * MiB).endswith(
        "65 MiB > 64 MiB limit]"
    )


def test_lowered_doc_cap_also_lowers_image_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ATTACHMENT_CAP_ENV, str(1 * MiB))
    assert effective_attachment_cap("image/png") == 1 * MiB
    monkeypatch.delenv(ATTACHMENT_CAP_ENV)
    monkeypatch.setattr(sandbox, "MAX_ATTACHMENT_BYTES", 2 * MiB)
    assert effective_attachment_cap("image/png") == 2 * MiB
    assert effective_attachment_cap("application/pdf") == 2 * MiB


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "1.5", "  "])
def test_invalid_env_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    monkeypatch.setenv(ATTACHMENT_CAP_ENV, bad)
    with caplog.at_level(logging.WARNING, logger="aios_connector_http.sandbox"):
        assert document_attachment_cap() == 50 * MiB
    if bad.strip():
        assert any(r.getMessage() == "attachment.cap.invalid_env" for r in caplog.records)


def test_is_over_cap_default_is_document_cap() -> None:
    assert is_over_cap(9 * MiB) is False
    assert is_over_cap(50 * MiB) is False
    assert is_over_cap(50 * MiB + 1) is True
    assert is_over_cap(6 * MiB, "image/png") is True
    assert is_over_cap(5 * MiB, "image/png") is False


# ── markers report the per-type limit (seat directive: pinned) ────────


def test_marker_pdf_over_doc_cap_reports_50_mib() -> None:
    assert rejection_marker("big.pdf", "application/pdf", 60 * MiB) == (
        "[attachment not delivered: 'big.pdf', application/pdf, 60 MiB > 50 MiB limit]"
    )


def test_marker_png_over_image_cap_reports_5_mib() -> None:
    markers: list[str] = []
    drop_attachment(
        markers,
        logging.getLogger("t.caps"),
        "signal",
        name="pic.png",
        content_type="image/png",
        size=6 * MiB,
        reason="over_cap",
    )
    assert markers == ["[attachment not delivered: 'pic.png', image/png, 6 MiB > 5 MiB limit]"]


def test_9_mib_pdf_is_not_over_cap() -> None:
    assert rejection_marker("x.pdf", "application/pdf", 9 * MiB).endswith("9 MiB: rejected]")


# ── as_params ────────────────────────────────────────────────────────


def test_as_params_9_mib_pdf_passes(tmp_path: Path) -> None:
    p = _sparse(tmp_path / "paper.pdf", 9 * MiB, b"%PDF-1.7\n")
    params = Attachment(
        host_path=str(p), filename="paper.pdf", content_type="application/pdf"
    ).as_params()
    assert params["size"] == 9 * MiB


def test_as_params_6_mib_png_rejected_naming_5_mib(tmp_path: Path) -> None:
    p = _sparse(tmp_path / "pic.png", 6 * MiB, b"\x89PNG\r\n\x1a\n")
    with pytest.raises(AttachmentError) as exc:
        Attachment(host_path=str(p), filename="pic.png", content_type="image/png").as_params()
    assert str(exc.value) == (
        f"attachment 'pic.png' is {6 * MiB} bytes; SDK cap is {5 * MiB} bytes (5 MiB)."
    )


def test_as_params_message_follows_env_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ATTACHMENT_CAP_ENV, str(2 * MiB))
    p = _sparse(tmp_path / "a.pdf", 3 * MiB)
    with pytest.raises(AttachmentError, match=r"SDK cap is 2097152 bytes \(2 MiB\)"):
        Attachment(host_path=str(p), filename="a.pdf", content_type="application/pdf").as_params()


def test_as_params_doc_over_50_mib_rejected(tmp_path: Path) -> None:
    p = _sparse(tmp_path / "huge.pdf", 50 * MiB + 1)
    with pytest.raises(AttachmentError, match=r"\(50 MiB\)"):
        Attachment(
            host_path=str(p), filename="huge.pdf", content_type="application/pdf"
        ).as_params()


# ── streaming through emit_inbound ───────────────────────────────────


class _HashingTransport(httpx.AsyncBaseTransport):
    """Consume the request body chunk by chunk, hashing; never store it."""

    def __init__(self) -> None:
        self.sha = hashlib.sha256()
        self.total = 0
        self.max_chunk = 0
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        stream = request.stream
        assert isinstance(stream, httpx.AsyncByteStream)
        async for chunk in stream:
            self.sha.update(chunk)
            self.total += len(chunk)
            self.max_chunk = max(self.max_chunk, len(chunk))
        return httpx.Response(200, json={"appended_event_id": "evt_1"})


class _Conn(HttpConnector):
    connector = "probe"

    def __init__(self, http: httpx.AsyncClient) -> None:
        super().__init__(base_url="http://x", token="aios_runtime_x")
        client = MagicMock()
        client.get_async_httpx_client.return_value = http
        self._client = client


def _write_seeded(path: Path, size: int) -> str:
    rng = random.Random(2563)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        f.write(b"%PDF-1.7\n")
        h.update(b"%PDF-1.7\n")
        remaining = size - 9
        while remaining > 0:
            n = min(remaining, MiB)
            chunk = rng.randbytes(n)
            f.write(chunk)
            h.update(chunk)
            remaining -= n
    return h.hexdigest()


async def _emit_with_handle(path: Path) -> tuple[_HashingTransport, int]:
    transport = _HashingTransport()
    async with httpx.AsyncClient(base_url="http://x", transport=transport) as http:
        conn = _Conn(http)
        with open(path, "rb") as fh:
            tracemalloc.start()
            try:
                result = await conn.emit_inbound(
                    connection_id="conn_1",
                    chat_id="c",
                    sender={"id": 1},
                    content="",
                    attachments=[("paper.pdf", fh, "application/pdf")],
                    event_id="evt-stream",
                )
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert not fh.closed  # caller owns the handle
    assert result == {"appended_event_id": "evt_1"}
    return transport, peak


async def test_emit_inbound_streams_50_mib_file_handle(tmp_path: Path) -> None:
    size = 50 * MiB
    path = tmp_path / "paper.pdf"
    file_sha = _write_seeded(path, size)
    transport, peak = await _emit_with_handle(path)
    assert transport.requests == 1
    # The multipart body is the file plus small text parts + boundaries.
    assert size < transport.total < size + 4096
    assert transport.max_chunk <= 1 * MiB
    assert peak < 8 * MiB, f"emit_inbound buffered the file: tracemalloc peak {peak}"
    assert len(file_sha) == 64  # byte-exactness: see the round-trip test below


async def test_emit_inbound_file_handle_body_byte_identical(tmp_path: Path) -> None:
    """The file part inside the streamed multipart body equals the file."""
    size = 3 * MiB + 17
    path = tmp_path / "paper.pdf"
    file_sha = _write_seeded(path, size)
    captured: list[bytes] = []

    class _Capture(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = b"".join([c async for c in request.stream])  # type: ignore[union-attr]
            captured.append(body)
            ctype = request.headers["content-type"]
            captured.append(ctype.split("boundary=", 1)[1].encode())
            return httpx.Response(200, json={})

    async with httpx.AsyncClient(base_url="http://x", transport=_Capture()) as http:
        with open(path, "rb") as fh:
            await _Conn(http).emit_inbound(
                connection_id="conn_1",
                chat_id="c",
                sender={"id": 1},
                content="hi",
                attachments=[("paper.pdf", fh, "application/pdf")],
            )
    body, boundary = captured
    part = next(p for p in body.split(b"--" + boundary) if b'filename="paper.pdf"' in p)
    payload = part.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
    assert len(payload) == size
    assert hashlib.sha256(payload).hexdigest() == file_sha


async def test_emit_inbound_bytes_payload_still_works() -> None:
    transport = _HashingTransport()
    async with httpx.AsyncClient(base_url="http://x", transport=transport) as http:
        result = await _Conn(http).emit_inbound(
            connection_id="conn_1",
            chat_id="c",
            sender={"id": 1},
            content="",
            attachments=[("a.txt", b"hello", "text/plain")],
        )
    assert result == {"appended_event_id": "evt_1"}
    assert transport.total > len(b"hello")

