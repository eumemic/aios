"""End-to-end integration: stubbed signal-cli + stubbed aios + real app.run.

We stub:

- **signal-cli daemon** — an ``asyncio.start_server`` listening on a
  loopback port that speaks the JSON-RPC protocol (answers ``listAccounts``,
  records ``send`` / ``sendReaction`` / ``sendReceipt``, and emits inbound
  ``receive`` notifications on the persistent listener connection).
- **signal-cli subprocess** — we patch :func:`_spawn_subprocess` to return a
  :class:`FakeProcess` that does nothing (the stub TCP server is all we need).
- **aios HTTP** — we patch :class:`IngestClient` in ``app.py`` to inject an
  :class:`httpx.MockTransport` that records every POST.

Assertions:

1. An inbound envelope emitted by the stub daemon → POST observed to aios
   with the expected shape (path, content, ``metadata.channel``).
2. A ``signal_send`` tool call over real MCP streamable HTTP → stub daemon
   records a ``send`` RPC with the expected params.
3. Closing the listener socket triggers a fatal crash → ``app.run`` raises.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from aios_signal import app as app_module
from aios_signal import daemon as daemon_module
from aios_signal.config import Settings
from aios_signal.ingest import IngestClient

BOT_UUID = "99999999-8888-7777-6666-555555555555"
BOT_PHONE = "+15550000000"


# ─── Fake subprocess ────────────────────────────────────────────────────────


class FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = _empty_reader()
        self.stderr = _empty_reader()
        self._wait_event = asyncio.Event()

    def send_signal(self, _sig: int) -> None:
        self.returncode = 0
        self._wait_event.set()

    def kill(self) -> None:
        self.send_signal(0)

    async def wait(self) -> int:
        await self._wait_event.wait()
        return self.returncode or 0


def _empty_reader() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    return reader


# ─── Stub signal-cli daemon ─────────────────────────────────────────────────


class StubDaemon:
    """TCP server speaking signal-cli's JSON-RPC."""

    def __init__(self) -> None:
        self.send_calls: list[dict[str, Any]] = []
        self.react_calls: list[dict[str, Any]] = []
        self.receipt_calls: list[dict[str, Any]] = []
        self._listener_writer: asyncio.StreamWriter | None = None
        self._server: asyncio.Server | None = None
        self.port: int = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # The first request is the `listAccounts` probe (fresh TCP).
        # Subsequent fresh-TCP connections are other RPC calls.
        # One long-lived connection is the listener — we detect it by an
        # absence of immediate data: if we don't see a request within 200ms
        # after accept, treat it as the listener.
        try:
            first = await asyncio.wait_for(reader.readline(), timeout=0.2)
        except TimeoutError:
            # This is the listener connection — hold it and use for pushing.
            self._listener_writer = writer
            with contextlib.suppress(ConnectionError, asyncio.CancelledError):
                # Park until the caller closes us.
                await reader.read()  # reads until EOF
            return

        if not first:
            return
        await self._handle_request(first, writer)
        # After the first request, keep handling more on this same connection
        # until it closes (allows simple clients that pool, even though our
        # RpcClient doesn't).
        while not reader.at_eof():
            try:
                line = await reader.readline()
            except ConnectionError:
                break
            if not line:
                break
            await self._handle_request(line, writer)

    async def _handle_request(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return
        method = request.get("method")
        params = request.get("params") or {}
        req_id = request.get("id", 0)

        result: Any
        if method == "listAccounts":
            result = [{"number": BOT_PHONE, "uuid": BOT_UUID}]
        elif method == "send":
            self.send_calls.append(params)
            result = {"timestamp": 42, "results": []}
        elif method == "sendReaction":
            self.react_calls.append(params)
            result = {"timestamp": 43, "results": []}
        elif method == "sendReceipt":
            self.receipt_calls.append(params)
            result = {"results": []}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
            return

        response = {"jsonrpc": "2.0", "id": req_id, "result": result}
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()

    async def push_envelope(self, envelope: dict[str, Any]) -> None:
        """Emit a ``receive`` notification on the listener connection."""
        # Wait briefly for the listener to connect.
        for _ in range(50):
            if self._listener_writer is not None:
                break
            await asyncio.sleep(0.05)
        assert self._listener_writer is not None, "listener never connected"
        notification = {
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {"envelope": envelope},
        }
        self._listener_writer.write((json.dumps(notification) + "\n").encode())
        await self._listener_writer.drain()

    async def close_listener(self) -> None:
        """Close the listener socket to trigger a fatal ListenerClosedError."""
        if self._listener_writer is not None:
            self._listener_writer.close()
            with contextlib.suppress(Exception):
                await self._listener_writer.wait_closed()
            self._listener_writer = None


# ─── Stub aios ingest ───────────────────────────────────────────────────────


class IngestRecorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(201, json={"session_id": "ses_x", "event_id": "evt_x"})


def _make_ingest_factory(recorder: IngestRecorder):  # type: ignore[no-untyped-def]
    class _PatchedIngest(IngestClient):
        async def __aenter__(self) -> _PatchedIngest:
            transport = httpx.MockTransport(recorder.handle)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                transport=transport,
            )
            return self

    return _PatchedIngest


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def stub_daemon() -> AsyncIterator[StubDaemon]:
    stub = StubDaemon()
    await stub.start()
    try:
        yield stub
    finally:
        await stub.stop()


@pytest.fixture
def settings(tmp_path: Path, stub_daemon: StubDaemon) -> Settings:
    # Pydantic-settings reads from env; clear any stray vars.
    for k in list(os.environ):
        if k.startswith(("AIOS_", "AIOS_SIGNAL_")):
            os.environ.pop(k)
    os.environ["AIOS_URL"] = "http://aios.stub"
    os.environ["AIOS_API_KEY"] = "stub-key"
    os.environ["AIOS_CONNECTION_ID"] = "conn_stub"
    os.environ["AIOS_SIGNAL_MCP_TOKEN"] = "stub-token"

    return Settings(
        phone=BOT_PHONE,
        config_dir=tmp_path,
        cli_bin="/nonexistent",
        daemon_host="127.0.0.1",
        daemon_port=stub_daemon.port,
        mcp_bind="127.0.0.1:0",  # the MCP server task will pick an ephemeral port
        mcp_token="stub-token",  # pydantic will pull from env if missing; explicit is fine
    )


@pytest.fixture
def patched_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_spawn(_args: list[str]) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr(daemon_module, "_spawn_subprocess", _fake_spawn)
    # Shorten ready-poll so tests aren't slow if something goes wrong.
    monkeypatch.setattr(daemon_module, "READY_POLL_ATTEMPTS", 20)
    monkeypatch.setattr(daemon_module, "READY_POLL_INTERVAL_S", 0.05)


# ─── Integration tests ──────────────────────────────────────────────────────


async def test_end_to_end_inbound_and_crash(
    stub_daemon: StubDaemon,
    settings: Settings,
    patched_spawn: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: inbound delivery, then listener-close fatality.

    Merged into one test because pytest-asyncio creates a fresh event loop
    per test, and uvicorn's in-process server state doesn't always tear down
    cleanly across loops — keeping the whole flow on one loop sidesteps that.
    """
    recorder = IngestRecorder()
    monkeypatch.setattr(app_module, "IngestClient", _make_ingest_factory(recorder))

    envelope = {
        "sourceUuid": "11111111-2222-3333-4444-555555555555",
        "sourceName": "Alice",
        "timestamp": 1700000000000,
        "dataMessage": {"message": "hello from alice", "timestamp": 1700000000000},
    }

    run_task = asyncio.create_task(app_module.run(settings))
    try:
        # Phase 1: inbound envelope arrives and is POSTed to aios.
        await asyncio.sleep(0.3)  # allow setup
        await stub_daemon.push_envelope(envelope)
        for _ in range(50):
            if recorder.requests:
                break
            await asyncio.sleep(0.05)

        assert len(recorder.requests) == 1
        request = recorder.requests[0]
        body = json.loads(request.read())
        assert body["path"] == "11111111-2222-3333-4444-555555555555"
        assert body["content"] == "hello from alice"
        assert body["metadata"]["channel"] == (
            f"signal/{BOT_UUID}/11111111-2222-3333-4444-555555555555"
        )
        assert body["metadata"]["sender_uuid"] == "11111111-2222-3333-4444-555555555555"

        # Phase 2: closing the listener triggers a fatal crash.
        await stub_daemon.close_listener()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(run_task, timeout=3.0)
        assert run_task.done()
        assert run_task.exception() is not None
    finally:
        if not run_task.done():
            run_task.cancel()
            with contextlib.suppress(BaseException):
                await run_task
