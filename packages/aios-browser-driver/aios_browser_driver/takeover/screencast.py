"""The takeover screencast pump.

Subscribes to CDP ``Page.startScreencast`` and writes each throttled frame to
``frames/frame-<seq>.jpg`` plus an atomically-renamed ``frames/manifest.json``
the API tails. The invariants (red-team folds #9/#10/#14):

* Every frame is ACKED (Chromium flow-controls on the ack); throttling skips
  WRITES, never acks, so frames keep flowing while disk churn stays bounded.
* The manifest is the publication barrier: it is renamed over its old copy only
  after ``frame-<seq>.jpg`` is fully written, so a reader (the API) that follows
  the manifest never sees a partial frame or a manifest pointing at a
  half-written file. The previous frame is then unlinked, so the frames dir
  holds ~one frame rather than growing unbounded (a reader mid-open of the
  just-unlinked frame gets ENOENT, which the API already tolerates).
* ``manifest["file"]`` is a frames-dir-relative BASENAME.
* ``seq`` is monotonic per BOOT (the API ends the stream on a boot change and
  advances on ``seq >``), so the counter is owned by the host and survives
  across takeovers within one boot.
* ``origin``/``security`` come from the committed main-frame URL (never pixels):
  origin is the URL's scheme+host, security is derived from the scheme
  (``https`` ⇒ secure). Chromium no longer emits ``Security.securityStateChanged``.
* The reaper can delete the frames dir mid-write; every write recreates it and
  tolerates ``ENOENT``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, CDPSession, Page

log = logging.getLogger("aios_browser_driver.takeover.screencast")

_PERSIST_INTERVAL_S = 0.2
_JPEG_QUALITY = 70
_MAX_W = 1280
_MAX_H = 800


class Screencast:
    def __init__(
        self,
        context: BrowserContext,
        frames_dir: Path,
        *,
        boot: str,
        epoch: int,
        next_seq: Callable[[], int],
    ) -> None:
        self._context = context
        self._frames_dir = frames_dir
        self._boot = boot
        self._epoch = epoch
        self._next_seq = next_seq
        self._cdp: CDPSession | None = None
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], int]] = asyncio.Queue()
        self._pump_task: asyncio.Task[None] | None = None
        self._origin: str | None = None
        self._security: str | None = None
        # Full committed main-frame URL, for the viewer's URL bar. Same
        # provenance as origin/security: the driver's view of the committed
        # navigation, never the pixels.
        self._url: str | None = None
        self._last_persist = 0.0
        self._last_frame: str | None = None

    async def start(self, page: Page) -> None:
        # A fresh queue and persist clock per (re)start, so a retarget never
        # persists a stale old-page frame under the new page's origin.
        self._queue = asyncio.Queue()
        self._last_persist = 0.0
        self._origin, self._security = _chrome_of(page.url)
        self._url = page.url or None
        cdp = await self._context.new_cdp_session(page)
        self._cdp = cdp
        await cdp.send("Page.enable")
        cdp.on("Page.screencastFrame", self._on_frame)
        cdp.on("Page.frameNavigated", self._on_nav)
        await cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": _JPEG_QUALITY, "maxWidth": _MAX_W, "maxHeight": _MAX_H},
        )
        self._pump_task = asyncio.get_running_loop().create_task(self._pump())

    async def retarget(self, page: Page) -> None:
        """Follow the active page — full stop on the old, start on the new; the
        boot-scoped seq counter continues, so the viewer sees no restart."""
        await self.stop()
        await self.start(page)

    async def stop(self) -> None:
        """Stop the screencast and JOIN the pump before returning — the caller
        rotates the epoch only once no more frames can be written."""
        if self._cdp is not None:
            with contextlib.suppress(Exception):
                await self._cdp.send("Page.stopScreencast")
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
            self._pump_task = None
        if self._cdp is not None:
            with contextlib.suppress(Exception):
                await self._cdp.detach()
            self._cdp = None

    # ── CDP event handlers (sync — enqueue / record only) ─────────────────

    def _on_frame(self, params: dict[str, Any]) -> None:
        data = params.get("data")
        session_id = params.get("sessionId")  # an int in the CDP wire format
        if isinstance(data, str) and isinstance(session_id, int):
            self._queue.put_nowait((data, params.get("metadata") or {}, session_id))

    def _on_nav(self, params: dict[str, Any]) -> None:
        frame = params.get("frame") or {}
        if not frame.get("parentId"):  # main frame only
            self._origin, self._security = _chrome_of(frame.get("url") or "")
            self._url = frame.get("url") or None

    # ── the pump ──────────────────────────────────────────────────────────

    async def _pump(self) -> None:
        while True:
            data, metadata, session_id = await self._queue.get()
            if self._cdp is not None:
                with contextlib.suppress(Exception):
                    await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
            now = time.monotonic()
            if now - self._last_persist < _PERSIST_INTERVAL_S:
                continue  # throttle WRITES, never acks
            self._last_persist = now
            try:
                self._persist(data, metadata)
            except OSError as exc:
                log.warning("frame persist failed: %s", exc)

    def _persist(self, data_b64: str, metadata: dict[str, Any]) -> None:
        seq = self._next_seq()
        jpeg = base64.b64decode(data_b64)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        name = f"frame-{seq}.jpg"
        # The frame file is written under a unique per-seq name, so it need not
        # be atomic — the manifest rename below is the publication barrier and
        # only names this frame once it is complete.
        (self._frames_dir / name).write_bytes(jpeg)
        manifest = {
            "seq": seq,
            "file": name,  # frames-dir-relative basename
            "ts_ms": int(time.time() * 1000),
            "epoch": self._epoch,
            "boot": self._boot,
            "origin": self._origin,
            "security": self._security,
            "url": self._url,
            "w": int(metadata.get("deviceWidth") or _MAX_W),
            "h": int(metadata.get("deviceHeight") or _MAX_H),
        }
        manifest_tmp = self._frames_dir / ".manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest), "utf-8")
        os.replace(manifest_tmp, self._frames_dir / "manifest.json")
        # The manifest now names the new frame; drop the previous one so the dir
        # stays O(1) instead of accumulating a frame every 200ms.
        if self._last_frame is not None and self._last_frame != name:
            with contextlib.suppress(OSError):
                (self._frames_dir / self._last_frame).unlink()
        self._last_frame = name


def _chrome_of(url: str) -> tuple[str | None, str | None]:
    """The trusted-chrome (origin, security) for a URL, both from the committed
    URL alone — origin is scheme+host, security is ``secure`` for https,
    ``insecure`` for http, ``None`` for anything else (about:blank, data:)."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None, None
    security = (
        "secure" if parts.scheme == "https" else "insecure" if parts.scheme == "http" else None
    )
    return f"{parts.scheme}://{parts.netloc}", security
