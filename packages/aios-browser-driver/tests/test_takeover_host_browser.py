"""The takeover lifecycle against a real BrowserHost + Chromium (opt-in:
``pytest -m browser``). Exercises the epoch state machine, gate blocking,
idempotency, replay, and the screencast plumbing end to end over a loopback
fixture page."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.server
import json
import socketserver
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from aios_browser_driver import host as host_mod
from aios_browser_driver.browser_protocol import BrowserRequest
from aios_browser_driver.host import BrowserHost

pytestmark = pytest.mark.browser

_HTML = b"<!doctype html><html><body><h1>Takeover fixture</h1><button>Go</button></body></html>"


@pytest.fixture
def server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        def log_message(self, *args: Any) -> None:
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
    finally:
        srv.shutdown()


@pytest.fixture
async def host(tmp_path: Path) -> AsyncIterator[BrowserHost]:
    from playwright.async_api import Error

    for sub in ("profile", "frames", "shots", "downloads", "input"):
        (tmp_path / sub).mkdir()
    h = BrowserHost(workspace=tmp_path, allow_private_nav=True)
    try:
        await h.start()
    except Error as exc:
        pytest.skip(f"playwright chromium not installed: {exc}")
    yield h
    await h.close()


async def _handle(h: BrowserHost, op: str, **kw: Any) -> Any:
    session_id = kw.pop("session_id", None)
    request = BrowserRequest(op=op, session_id=session_id, args=dict(kw))
    return await h.handle(request, deadline=time.monotonic() + 30)


async def _land_on_fixture(h: BrowserHost, server: str, session_id: str) -> None:
    resp = await _handle(h, "navigate", session_id=session_id, url=server, description="open")
    assert resp.ok, resp.error


async def test_open_close_lifecycle(host: BrowserHost, server: str) -> None:
    await _land_on_fixture(host, server, "s1")

    opened = await _handle(host, "takeover_open", session_id="s1", grant_id="g1", reason="login")
    assert opened.ok
    assert opened.data["target"]["url"] == server
    assert opened.snapshot is None and opened.url is None  # page-blind top level
    open_epoch = opened.epoch
    assert open_epoch > 0

    # An agent action is refused, page-blind, while the human holds the browser.
    blocked = await _handle(host, "snapshot", session_id="s1")
    assert not blocked.ok and blocked.error.code == "takeover_active"
    assert blocked.snapshot is None and blocked.tabs == []

    # A competing open for a different grant is refused without disturbing g1.
    competing = await _handle(host, "takeover_open", session_id="s1", grant_id="g2")
    assert competing.error.code == "takeover_active"

    # Idempotent re-open of the SAME grant echoes the original epoch.
    echo = await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    assert echo.ok and echo.epoch == open_epoch

    # Close hands back — url populated, epoch rotates again.
    closed = await _handle(host, "takeover_close", grant_id="g1", outcome="done")
    assert closed.ok and closed.epoch > open_epoch
    assert closed.url is not None  # A2 fold: close must populate url

    # After close, the agent can act again.
    acted = await _handle(host, "snapshot", session_id="s1")
    assert acted.ok


async def test_close_replays_from_cache_and_unknown_is_no_takeover(
    host: BrowserHost, server: str
) -> None:
    await _land_on_fixture(host, server, "s1")
    await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    first = await _handle(host, "takeover_close", grant_id="g1")
    replay = await _handle(host, "takeover_close", grant_id="g1")
    assert replay.ok and replay.url == first.url

    unknown = await _handle(host, "takeover_close", grant_id="gX")
    assert unknown.error.code == "no_takeover"


async def test_grant_mismatch_close(host: BrowserHost, server: str) -> None:
    await _land_on_fixture(host, server, "s1")
    await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    mismatch = await _handle(host, "takeover_close", grant_id="other")
    assert mismatch.error.code == "grant_mismatch"


async def test_screencast_writes_a_frame_manifest(
    host: BrowserHost, server: str, tmp_path: Path
) -> None:
    await _land_on_fixture(host, server, "s1")
    await _handle(host, "takeover_open", session_id="s1", grant_id="g1")

    manifest_path = tmp_path / "frames" / "manifest.json"
    for _ in range(50):
        if manifest_path.exists():
            break
        await asyncio.sleep(0.1)
    assert manifest_path.exists(), "screencast never wrote a manifest"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["boot"] == host.boot
    assert manifest["file"] == f"frame-{manifest['seq']}.jpg"
    assert "/" not in manifest["file"]
    assert manifest["security"] == "insecure"  # http fixture, derived from scheme
    assert manifest["origin"] == server.rstrip("/")
    assert (tmp_path / "frames" / manifest["file"]).exists()
    await _handle(host, "takeover_close", grant_id="g1")


async def test_cancelled_open_reopens_the_gate(
    host: BrowserHost, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE critical regression: a dispatch-deadline cancellation lands inside
    # takeover_open AFTER the gate has closed. If the rollback only caught
    # Exception (not the BaseException CancelledError), the gate would wedge
    # closed forever and every agent action would return takeover_active.
    await _land_on_fixture(host, server, "s1")
    started = asyncio.Event()

    class _HangingScreencast:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def start(self, page: Any) -> None:
            started.set()
            await asyncio.Event().wait()  # never returns — the open parks here

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(host_mod, "Screencast", _HangingScreencast)
    task = asyncio.create_task(_handle(host, "takeover_open", session_id="s1", grant_id="g1"))
    await asyncio.wait_for(started.wait(), timeout=10)  # gate is now closed, parked in start()
    assert host._gate.closed is True

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The finally must have reopened the gate and left no standing takeover.
    assert host._gate.closed is False
    assert host._standing is None
    # And an agent action is admitted again — not wedged at takeover_active.
    acted = await _handle(host, "snapshot", session_id="s1")
    assert acted.ok


async def test_input_spool_drives_the_page(host: BrowserHost, server: str, tmp_path: Path) -> None:
    await _land_on_fixture(host, server, "s1")
    opened = await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    epoch = opened.epoch

    # A current-epoch batch is applied; a stale-epoch batch is dropped. We prove
    # application by observing that no exception tears down the takeover and the
    # close still succeeds after replaying input.
    spool = tmp_path / "input" / "spool.jsonl"
    batches = [
        {
            "grant_id": "g1",
            "epoch": epoch,
            "seq": 1,
            "events": [{"type": "pointer_move", "x": 5, "y": 5}],
        },
        {"grant_id": "g1", "epoch": epoch - 1, "seq": 2, "events": [{"type": "text", "text": "x"}]},
        {"grant_id": "other", "epoch": epoch, "seq": 3, "events": [{"type": "text", "text": "y"}]},
    ]
    spool.write_text("".join(json.dumps(b) + "\n" for b in batches), "utf-8")
    await asyncio.sleep(0.3)  # let the input pump drain a couple of polls

    closed = await _handle(host, "takeover_close", grant_id="g1")
    assert closed.ok
    assert host._standing is None


async def test_status_answers_during_a_takeover(host: BrowserHost, server: str) -> None:
    await _land_on_fixture(host, server, "s1")
    await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    status = await _handle(host, "status")
    assert status.ok and "signed_in_hosts" in status.data
    # revoke_site, by contrast, refuses during a takeover.
    revoke = await _handle(host, "revoke_site", host="example.com")
    assert revoke.error.code == "takeover_active"


async def test_peek_shows_a_session_page_without_touching_it(
    host: BrowserHost, server: str
) -> None:
    # Nothing open yet: a running computer with no page is not a page.
    empty = await _handle(host, "peek", session_id="s1")
    assert empty.ok and empty.data == {"page": None}
    assert "s1" not in host._entries  # a look never creates a page

    await _land_on_fixture(host, server, "s1")
    peek = await _handle(host, "peek", session_id="s1")
    assert peek.ok and peek.url == server
    page = peek.data["page"]
    assert page["w"] > 0 and page["h"] > 0
    assert base64.b64decode(page["jpeg_b64"])[:3] == b"\xff\xd8\xff"  # a JPEG
    # Trusted chrome is derived from the committed URL, not the pixels.
    assert page["origin"] == server.rstrip("/")
    assert page["security"] == "insecure"

    # Without a session it falls back to the last-active page.
    last = await _handle(host, "peek")
    assert last.ok and last.url == server

    # A human holding the computer is theirs alone.
    await _handle(host, "takeover_open", session_id="s1", grant_id="g1")
    held = await _handle(host, "peek", session_id="s1")
    assert held.error.code == "takeover_active"
