"""The request server answers every connection with exactly one envelope — a
valid request round-trips, and an over-long line (no newline within the read
limit) still yields an ``invalid_request`` document rather than a dropped
connection."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile

import pytest
from aios_browser_driver import server
from aios_browser_driver.browser_protocol import BrowserRequest, BrowserResponse
from aios_browser_driver.dispatch import dispatch


class _StubHost:
    boot = "01SERVERBOOTULIDULIDULID"
    epoch = 0

    async def handle(self, request: BrowserRequest, *, deadline: float) -> BrowserResponse:
        return BrowserResponse(ok=True, boot=self.boot, epoch=self.epoch, url="https://ok.test")


async def _start(monkeypatch: pytest.MonkeyPatch) -> tuple[str, asyncio.Task[None], str]:
    # A short dir — an AF_UNIX path over ~104 chars is rejected (pytest's
    # tmp_path is too long on macOS).
    tmpdir = tempfile.mkdtemp(prefix="aiosdrv-")
    sock = os.path.join(tmpdir, "d.sock")
    monkeypatch.setenv("AIOS_BROWSER_DRIVER_SOCK", sock)
    ready = asyncio.Event()
    host = _StubHost()
    task = asyncio.create_task(server.serve(lambda raw: dispatch(raw, host), ready=ready))
    await asyncio.wait_for(ready.wait(), timeout=2)
    return sock, task, tmpdir


async def _stop(task: asyncio.Task[None], tmpdir: str) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    shutil.rmtree(tmpdir, ignore_errors=True)


async def _round_trip(sock: str, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_unix_connection(sock)

    async def _write() -> None:
        with contextlib.suppress(OSError):
            writer.write(payload)
            await writer.drain()
            writer.write_eof()

    write_task = asyncio.create_task(_write())
    try:
        return await asyncio.wait_for(reader.readline(), timeout=2)
    finally:
        await write_task
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def test_valid_request_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    sock, task, tmpdir = await _start(monkeypatch)
    try:
        line = await _round_trip(sock, json.dumps({"op": "status"}).encode() + b"\n")
        resp = BrowserResponse.model_validate_json(line)
        assert resp.ok and resp.boot == "01SERVERBOOTULIDULIDULID"
    finally:
        await _stop(task, tmpdir)


async def test_oversized_line_is_invalid_request_not_a_dropped_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_MAX_REQUEST_BYTES", 256)
    sock, task, tmpdir = await _start(monkeypatch)
    try:
        # 1 KiB with no newline blows the 256-byte read limit — the client must
        # get an envelope back, not EOF (which the worker reads as a transport
        # failure instead of a malformed request).
        line = await _round_trip(sock, b"x" * 1024)
        resp = BrowserResponse.model_validate_json(line)
        assert resp.error is not None and resp.error.code == "invalid_request"
    finally:
        await _stop(task, tmpdir)
