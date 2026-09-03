"""The Chromium-backed browser host.

One persistent context over the plane profile, one page per agent session,
popups auto-followed. ``boot`` identifies the Chromium-context generation: it
rotates on EVERY (re)launch — daemon start and in-container relaunch alike —
because both lose page state, and boot is the wire signal for that loss.

Restart truth (``driver_restarted``) is a per-session ledger at the plane
root (``/workspace/.aios/sessions.json`` — outside the five subdirs, so
``clear_state``'s subdir wipe and the reaper never touch it): each session
records the boot it last held a page under, and a first touch under a
different boot reports the loss. The ledger advances only when an ``ok``
response actually carries the flag, so a cancelled or ``ok: false`` first
touch does not silently spend the once-per-restart signal. A single page
dying without Chromium dying (renderer crash) sets the entry's ``page_lost``
flag instead — same report, no boot rotation.

Death handling is unified through :meth:`_trigger_relaunch`. Chromium death
fires the context ``close`` event → relaunch under a fresh boot (page state is
gone; in-flight ops fail ``browser_crashed``). Death of playwright's own driver
subprocess fires NO context event, so a ``TargetClosedError`` surfacing from
any operation ALSO triggers a relaunch — which then fails to launch (the
connection is gone) and resolves :meth:`failed`, crashing the daemon visibly. A
crash-looping container beats a live one that answers nothing.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ulid import ULID

from aios_browser_driver import actions
from aios_browser_driver.browser_protocol import (
    DOWNLOADS_DIR,
    DRIVER_TAKEOVER_IDLE_TIMEOUT_S,
    DRIVER_TAKEOVER_UNCLAIMED_TIMEOUT_S,
    FRAMES_DIR,
    INPUT_SPOOL,
    PROFILE_DIR,
    TAKEOVER_HEARTBEAT_MARKER,
    BrowserError,
    BrowserRequest,
    BrowserResponse,
    BrowserTab,
)
from aios_browser_driver.errors import ActionError, error_response, first_line, require_str
from aios_browser_driver.hosts import normalize_host, signed_in_hosts
from aios_browser_driver.snapshot.snapshot import take_snapshot
from aios_browser_driver.takeover.injector import InputInjector
from aios_browser_driver.takeover.screencast import Screencast, chrome_of
from aios_browser_driver.takeover.spool import SpoolTailer
from aios_browser_driver.takeover.state import AdmissionGate, ReplayCache, Takeover, now

if TYPE_CHECKING:
    from playwright.async_api import (
        BrowserContext,
        CDPSession,
        Dialog,
        Download,
        FileChooser,
        Page,
        Playwright,
    )

log = logging.getLogger("aios_browser_driver.host")

WORKSPACE = Path("/workspace")
_LEDGER_RELPATH = Path(".aios/sessions.json")
_PROFILE_SUBDIR = PurePosixPath(PROFILE_DIR).name
_DOWNLOADS_SUBDIR = PurePosixPath(DOWNLOADS_DIR).name
_FRAMES_SUBDIR = PurePosixPath(FRAMES_DIR).name
_SPOOL_REL = PurePosixPath(INPUT_SPOOL).relative_to("/workspace")
_HEARTBEAT_REL = PurePosixPath(TAKEOVER_HEARTBEAT_MARKER)

# Takeover watchdog cadence and the absolute lifetime cap (a backstop above the
# TTL/idle timeouts — a takeover cannot hold the browser forever).
_WATCHDOG_TICK_S = 15.0
_ABSOLUTE_CAP_S = 3600.0
_INPUT_POLL_S = 0.05
# Keep the admission drain strictly inside the exec/dispatch deadline.
_DRAIN_MARGIN_S = 2.0
# Cap the handback screenshot so the whole capture stays inside dispatch's
# deadline (a hung page degrades to a partial handback, never a cancel).
_HANDBACK_SHOT_TIMEOUT_MS = 8000.0
# Background pages must keep rendering (a session's page is "backgrounded"
# whenever another session's is frontmost); the disk cache is capped and
# /dev/shm avoided for container-friendliness. NO --no-sandbox — Chromium's
# own sandbox is the primary boundary and launch must FAIL if it cannot hold.
# (Absence from this list is not enough: playwright INJECTS --no-sandbox
# unless ``chromium_sandbox=True`` is passed at launch — see ``_launch``.)
_LAUNCH_ARGS = [
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disk-cache-size=268435456",
    "--disable-dev-shm-usage",
]
_MAX_PAGES_PER_SESSION = 4

# Chromium's process-singleton residue in a persistent profile. The registry
# keys browser containers one-per-account, so this daemon's Chromium is the
# ONLY legitimate holder of this profile — a lock present before launch is
# always the residue of an unclean death (idle-reap SIGKILL, OOM, host
# reboot), never a live peer. Chromium itself cannot tell: the lock names the
# dead container's hostname, so it refuses the profile as "in use by another
# computer" and every future container for the account crash-loops (wedged
# prod, 2026-09-01: takeover open → "no reply from driver" forever).
_SINGLETON_RESIDUE = ("SingletonLock", "SingletonCookie", "SingletonSocket")


def clear_stale_singleton_locks(profile_dir: Path) -> None:
    """Remove Chromium's singleton lock residue before launch."""
    for name in _SINGLETON_RESIDUE:
        path = profile_dir / name
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
            log.warning("removed stale profile lock %s", name)


@dataclass
class PageEntry:
    """One session's slot in the page registry: its page(s), snapshot
    bookkeeping, and the dialog notes the next snapshot will carry. The
    per-session lock lives on the host (keyed by session id) so it survives
    entry recreation — the mutual-exclusion invariant must not reset when a
    page is lost and rebuilt."""

    session_id: str
    pages: list[Page]
    generation: int = 0
    issued: int = 0
    dialogs: list[str] = field(default_factory=list)
    page_lost: bool = False

    def open_pages(self) -> list[Page]:
        self.pages = [p for p in self.pages if not p.is_closed()]
        return self.pages

    @property
    def active_page(self) -> Page | None:
        """The newest open page — popups are auto-followed."""
        pages = self.open_pages()
        return pages[-1] if pages else None

    def drain_dialogs(self) -> list[str]:
        drained = self.dialogs[:]
        self.dialogs.clear()
        return drained


class BrowserHost:
    """Implements the dispatch :class:`~aios_browser_driver.dispatch.Host`
    protocol over a persistent Chromium context."""

    def __init__(self, *, workspace: Path = WORKSPACE, allow_private_nav: bool = False) -> None:
        self.boot = str(ULID())
        self.epoch = 0
        self.workspace = workspace
        self.allow_private_nav = allow_private_nav
        self._entries: dict[str, PageEntry] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._ledger = _load_ledger(self._ledger_path)
        self._last_session: str | None = None
        self._pw: Playwright | None = None
        self._context: BrowserContext | None = None
        self._ready = asyncio.Event()
        self._closing = False
        self._failure: asyncio.Future[None] | None = None
        self._relaunch_task: asyncio.Task[None] | None = None
        # Takeover machinery (jarbot#106 §5.6): the gate serializes agent
        # actions against the human's control; one takeover stands at a time,
        # serialized by the transition lock; frame seq is per-boot.
        self._gate = AdmissionGate()
        self._transition_lock = asyncio.Lock()
        self._standing: Takeover | None = None
        self._replay = ReplayCache()
        self._frame_seq = 0
        self._watchdog_task: asyncio.Task[None] | None = None

    @property
    def _ledger_path(self) -> Path:
        return self.workspace / _LEDGER_RELPATH

    @property
    def _frames_dir(self) -> Path:
        return self.workspace / _FRAMES_SUBDIR

    @property
    def _spool_path(self) -> Path:
        return self.workspace / _SPOOL_REL

    @property
    def _heartbeat_marker(self) -> Path:
        return self.workspace / _HEARTBEAT_REL

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch Chromium. Raises on failure — the daemon must crash visibly
        rather than serve a browser that never came up."""
        self._failure = asyncio.get_running_loop().create_future()
        await self._launch()

    async def failed(self) -> None:
        """Resolves (with the exception) if the browser dies and cannot be
        relaunched. Pends forever otherwise — the daemon races it against
        shutdown."""
        assert self._failure is not None, "failed() before start()"
        await self._failure

    async def close(self) -> None:
        """Best-effort graceful shutdown (SIGTERM path): closing the context
        flushes the profile to disk before the container is destroyed."""
        self._closing = True
        if self._relaunch_task is not None:
            self._relaunch_task.cancel()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
        # Tear a standing takeover down FIRST — its screencast/input tasks must
        # be joined before the context closes, or they run against a dying
        # context and can wedge the shutdown.
        standing = self._standing
        if standing is not None:
            if standing.input_task is not None:
                standing.input_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await standing.input_task
            with contextlib.suppress(Exception):
                await standing.screencast.stop()
            self._standing = None
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
        if self._pw is not None:
            with contextlib.suppress(Exception):
                await self._pw.stop()

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

        clear_stale_singleton_locks(self.workspace / _PROFILE_SUBDIR)
        if self._pw is None:
            self._pw = await async_playwright().start()
        context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.workspace / _PROFILE_SUBDIR),
            channel="chromium",
            headless=True,
            # Playwright DEFAULTS the Chromium sandbox OFF (it injects
            # --no-sandbox); this opt-in is what actually keeps the primary
            # boundary up. The browser image's seccomp profile re-permits the
            # unprivileged USER/PID/NET namespaces the zygote needs, and the
            # image contract e2e asserts renderers really run namespaced
            # (and that launch FAILS under the stricter sandbox profile).
            chromium_sandbox=True,
            viewport={"width": actions.VIEWPORT_WIDTH, "height": actions.VIEWPORT_HEIGHT},
            accept_downloads=True,
            downloads_path=str(self.workspace / _DOWNLOADS_SUBDIR),
            args=_LAUNCH_ARGS,
        )
        context.on("close", self._on_context_close)
        self._context = context
        self._ready.set()
        log.info("chromium up (boot=%s)", self.boot)

    def _on_context_close(self, _context: BrowserContext) -> None:
        # Chromium died under us (this event does NOT fire for driver-process
        # death — that path reaches _trigger_relaunch via a TargetClosedError).
        self._trigger_relaunch()

    def _trigger_relaunch(self) -> None:
        if self._closing or (self._relaunch_task is not None and not self._relaunch_task.done()):
            return
        self._ready.clear()
        self._relaunch_task = asyncio.get_running_loop().create_task(self._relaunch())

    async def _relaunch(self) -> None:
        log.warning("browser died (boot=%s); relaunching", self.boot)
        self.boot = str(ULID())
        self._frame_seq = 0  # seq is per-boot
        # A standing takeover is defunct (its page and CDP died with Chromium):
        # force-close it so the worker's later close replays an honest
        # browser_crashed handback rather than hanging on a dead grant.
        async with self._transition_lock:
            if self._standing is not None:
                await self._finalize(self._standing, "browser_crashed")
        async with self._registry_lock:
            self._entries.clear()
        self._last_session = None
        try:
            await self._launch()
        except Exception as exc:
            # A launch failure here means the playwright driver itself is gone
            # (Chromium alone would relaunch fine) — unrecoverable in-process.
            log.error("browser relaunch failed: %s", exc)
            if self._failure is not None and not self._failure.done():
                self._failure.set_exception(exc)

    # ── dispatch entry point ──────────────────────────────────────────────

    async def handle(self, request: BrowserRequest, *, deadline: float) -> BrowserResponse:
        from playwright._impl._errors import TargetClosedError

        await self._ready.wait()
        # Stamp the envelope from boot/epoch read at entry, so a relaunch
        # racing this request can't pair a new boot with old page data.
        boot, epoch = self.boot, self.epoch
        try:
            if request.op == "takeover_open":
                return await self._takeover_open(request, deadline, boot)
            if request.op == "takeover_close":
                return await self._takeover_close(request, boot, epoch)
            if request.op == "status":
                return await self._status(boot, epoch)
            if request.op == "peek":
                return await self._peek(request, boot, epoch)
            if request.op == "revoke_site":
                if self._standing is not None:
                    # Never clear cookies under the human — the product polls
                    # status during takeovers, but revoke waits for handback.
                    return error_response(
                        boot, epoch, "takeover_active", "cannot revoke a site during a takeover"
                    )
                return await self._revoke_site(request.args, boot, epoch)
            if not request.session_id:
                raise ActionError("invalid_request", f"{request.op} requires a session_id")
            async with self._session_lock(request.session_id):
                if not self._gate.admit():
                    # A human holds the browser. Page-blind by construction
                    # (error_response carries no snapshot/url/title/tabs) so the
                    # human's live page never leaks into agent context.
                    return error_response(
                        boot, epoch, "takeover_active", "a human has taken over the computer"
                    )
                try:
                    entry, restarted = await self._ensure_entry(request.session_id)
                    self._last_session = request.session_id
                    return await self._run_action(entry, request, restarted, deadline, boot, epoch)
                finally:
                    self._gate.release()
        except ActionError as exc:
            # The currency's total sink for the control path + pre-action
            # checks; an action's own ActionError is enveloped WITH the
            # observation inside _run_action, so it never reaches here.
            return error_response(boot, epoch, exc.code, exc.message)
        except TargetClosedError:
            self._trigger_relaunch()
            return error_response(boot, epoch, "browser_crashed", "the browser went away")

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    # ── the page registry ─────────────────────────────────────────────────

    def _require_context(self) -> BrowserContext:
        assert self._context is not None, "context gone while ready was set"
        return self._context

    async def _ensure_entry(self, session_id: str) -> tuple[PageEntry, bool]:
        """Get-or-create the session's entry; report a page-state loss.

        Runs under the session lock, so no concurrent request for the same
        session can be mid-action while this recreates its pages."""
        async with self._registry_lock:
            restarted = False
            entry = self._entries.get(session_id)
            if entry is not None and entry.page_lost:
                # The page's renderer died without Chromium dying: same loss,
                # same report, no boot rotation.
                restarted = True
                for page in entry.open_pages():
                    with contextlib.suppress(Exception):
                        await page.close()
                entry = None
            if entry is not None and entry.active_page is None:
                # The page closed itself (window.close()) — recreate silently:
                # nothing was lost that the model didn't do itself.
                entry = None
            if entry is None:
                self._entries.pop(session_id, None)
                prior_boot = self._ledger.get(session_id)
                if prior_boot is not None and prior_boot != self.boot:
                    restarted = True
                page = await self._require_context().new_page()
                entry = PageEntry(session_id=session_id, pages=[page])
                self._attach_page(entry, page)
                self._entries[session_id] = entry
            return entry, restarted

    def _attach_page(self, entry: PageEntry, page: Page) -> None:
        async def on_dialog(dialog: Dialog) -> None:
            entry.dialogs.append(f'{dialog.type} — "{dialog.message[:200]}"')
            with contextlib.suppress(Exception):
                await dialog.dismiss()

        async def on_filechooser(chooser: FileChooser) -> None:
            entry.dialogs.append("file chooser — dismissed (file upload is not supported)")
            with contextlib.suppress(Exception):
                await chooser.set_files([])

        async def on_download(download: Download) -> None:
            # Persist to the plane with a real name; playwright otherwise stores
            # the artifact under a GUID and DELETES it on context close.
            await self._save_download(download)

        def on_crash(_page: Page) -> None:
            entry.page_lost = True

        async def on_popup(popup: Page) -> None:
            self._adopt_popup(entry, popup)

        page.on("dialog", on_dialog)
        page.on("filechooser", on_filechooser)
        page.on("download", on_download)
        page.on("crash", on_crash)
        page.on("popup", on_popup)

    async def _save_download(self, download: Download) -> None:
        dest_dir = self.workspace / _DOWNLOADS_SUBDIR
        suggested = download.suggested_filename or "download"
        dest = dest_dir / f"{ULID()!s}-{suggested}"
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            await download.save_as(dest)
        except Exception as exc:
            log.warning("download save failed (%s): %s", suggested, exc)

    def _adopt_popup(self, entry: PageEntry, popup: Page) -> None:
        """A popup becomes the session's active page (newest = active)."""
        entry.pages.append(popup)
        self._attach_page(entry, popup)
        open_pages = entry.open_pages()
        while len(open_pages) > _MAX_PAGES_PER_SESSION:
            oldest = open_pages.pop(0)
            task = asyncio.get_running_loop().create_task(oldest.close())
            task.add_done_callback(lambda t: t.exception())  # close is best-effort
        # If a takeover stands for this session, the human just opened a popup —
        # move the screencast AND injector to it together (they must always
        # target the same page).
        standing = self._standing
        if standing is not None and standing.session_id == entry.session_id:
            retarget = asyncio.get_running_loop().create_task(
                self._retarget_takeover(entry.session_id)
            )
            retarget.add_done_callback(lambda t: t.exception())

    # ── the action envelope ───────────────────────────────────────────────

    async def _run_action(
        self,
        entry: PageEntry,
        request: BrowserRequest,
        restarted: bool,
        deadline: float,
        boot: str,
        epoch: int,
    ) -> BrowserResponse:
        from playwright._impl._errors import TargetClosedError
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        started = time.monotonic()
        page = entry.active_page
        assert page is not None, "_ensure_entry returned an entry with no page"
        error: BrowserError | None = None
        shot_path: str | None = None
        try:
            shot_path = await actions.run(
                entry,
                page,
                request.op,
                request.args,
                deadline=deadline,
                allow_private_nav=self.allow_private_nav,
                workspace=self.workspace,
            )
        except ActionError as exc:
            if exc.guardrail:
                log.info("guardrail refusal (session=%s op=%s)", entry.session_id, request.op)
            error = BrowserError(code=exc.code, message=exc.message)
        except TargetClosedError:
            self._trigger_relaunch()
            error = BrowserError(code="browser_crashed", message="the browser page went away")
        except PlaywrightTimeoutError as exc:
            # Playwright's own timeout on an element op is its actionability
            # wait expiring (hidden/covered/disabled); the overall deadline is
            # dispatch's to enforce.
            error = BrowserError(code="not_interactable", message=first_line(exc))
        except PlaywrightError as exc:
            error = BrowserError(code="internal", message=first_line(exc))

        # The post-action snapshot — also on ok:false, so the model can
        # self-correct (stale ref → re-target) without a re-observe call.
        snapshot: str | None = None
        truncated = False
        page = entry.active_page or page  # the action may have opened a popup
        if error is None or error.code != "browser_crashed":
            try:
                snapshot, truncated = await take_snapshot(page, entry)
            except Exception as exc:
                if error is None:
                    # The action succeeded but the page cannot be observed —
                    # that is a failure of the action's contract, not garnish.
                    if isinstance(exc, TargetClosedError):
                        self._trigger_relaunch()
                        error = BrowserError(code="browser_crashed", message=first_line(exc))
                    else:
                        error = BrowserError(code="internal", message=first_line(exc))

        url, title, tabs = await self._observe(entry, page)
        # Deliver-then-consume: advance the ledger only when this ok response
        # actually carries the restart flag, so a cancelled/failed first touch
        # leaves the once-per-restart signal for the next response.
        if error is None and self._ledger.get(entry.session_id) != boot:
            self._ledger[entry.session_id] = boot
            _save_ledger(self._ledger_path, self._ledger)
        return BrowserResponse(
            ok=error is None,
            boot=boot,
            epoch=epoch,
            url=url,
            title=title,
            tabs=tabs,
            snapshot=snapshot,
            snapshot_truncated=truncated,
            duration_ms=int((time.monotonic() - started) * 1000),
            shot_path=shot_path,
            driver_restarted=restarted,
            error=error,
        )

    async def _observe(
        self, entry: PageEntry, page: Page
    ) -> tuple[str | None, str | None, list[BrowserTab]]:
        """url/title/tabs, every field coerced — one null field would fail the
        whole response's validation worker-side."""
        url: str | None = None
        active_title: str | None = None
        with contextlib.suppress(Exception):
            url = page.url
        with contextlib.suppress(Exception):
            active_title = await page.title()
        tabs: list[BrowserTab] = []
        for index, open_page in enumerate(entry.open_pages()):
            is_active = open_page is page
            tab_title = active_title if is_active else None  # reuse the active read
            if tab_title is None:
                with contextlib.suppress(Exception):
                    tab_title = await open_page.title()
            tabs.append(
                BrowserTab(
                    index=index,
                    url=open_page.url or "",
                    title=tab_title or "",
                    active=is_active,
                )
            )
        return url, active_title, tabs

    # ── control ops ───────────────────────────────────────────────────────

    async def _status(self, boot: str, epoch: int) -> BrowserResponse:
        """Whole-browser state. Page-blind beyond url/title of the current
        page (protocol pin) — no snapshot, no tabs."""
        url: str | None = None
        title: str | None = None
        entry = self._entries.get(self._last_session) if self._last_session else None
        page = entry.active_page if entry else None
        if page is not None:
            with contextlib.suppress(Exception):
                url = page.url
            with contextlib.suppress(Exception):
                title = await page.title()
        return BrowserResponse(
            ok=True,
            boot=boot,
            epoch=epoch,
            url=url,
            title=title,
            data={"signed_in_hosts": await self._signed_in_hosts()},
        )

    async def _peek(self, request: BrowserRequest, boot: str, epoch: int) -> BrowserResponse:
        """A read-only look at a page for the product's live view: one JPEG
        of the current viewport plus the trusted chrome, taken from the
        session's page when ``session_id`` is given, else the last-active
        page. Never creates a page (``data.page`` is ``None`` when there is
        nothing to look at) and never admits through the gate — it neither
        moves the page nor competes with an action. Refused during a
        takeover: the human's live page is theirs alone."""
        if self._standing is not None:
            return error_response(
                boot, epoch, "takeover_active", "a human has taken over the computer"
            )
        session_id = request.session_id or self._last_session
        entry = self._entries.get(session_id) if session_id else None
        page = entry.active_page if entry else None
        if page is None:
            return BrowserResponse(ok=True, boot=boot, epoch=epoch, data={"page": None})
        url = page.url
        origin, security = chrome_of(url)
        jpeg = await page.screenshot(type="jpeg", quality=55, scale="css")
        viewport = page.viewport_size or {"width": 0, "height": 0}
        return BrowserResponse(
            ok=True,
            boot=boot,
            epoch=epoch,
            url=url,
            title=await page.title(),
            data={
                "page": {
                    "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                    "w": viewport["width"],
                    "h": viewport["height"],
                    "origin": origin,
                    "security": security,
                }
            },
        )

    async def _revoke_site(self, args: dict[str, Any], boot: str, epoch: int) -> BrowserResponse:
        host = normalize_host(require_str(args, "host"))
        context = self._require_context()
        await context.clear_cookies(domain=host)
        await context.clear_cookies(domain=f".{host}")
        await self._clear_origin_storage(context, host)
        return BrowserResponse(
            ok=True,
            boot=boot,
            epoch=epoch,
            data={"signed_in_hosts": await self._signed_in_hosts()},
        )

    async def _signed_in_hosts(self) -> list[str]:
        cookies = await self._require_context().cookies()
        return signed_in_hosts(cookies, now=time.time())

    async def _clear_origin_storage(self, context: BrowserContext, host: str) -> None:
        """CDP ``Storage.clearDataForOrigin`` for both schemes — cookies are
        cleared context-wide above; this removes local/session storage,
        IndexedDB, service workers, and cache storage. One temporary page (a
        rare, owner-driven op) keeps the path single."""
        page = await context.new_page()
        cdp: CDPSession | None = None
        try:
            cdp = await context.new_cdp_session(page)
            for origin in (f"https://{host}", f"http://{host}"):
                await cdp.send(
                    "Storage.clearDataForOrigin", {"origin": origin, "storageTypes": "all"}
                )
        finally:
            if cdp is not None:
                with contextlib.suppress(Exception):
                    await cdp.detach()
            with contextlib.suppress(Exception):
                await page.close()

    # ── takeover ──────────────────────────────────────────────────────────

    def _next_frame_seq(self) -> int:
        self._frame_seq += 1
        return self._frame_seq

    async def _takeover_open(
        self, request: BrowserRequest, deadline: float, boot: str
    ) -> BrowserResponse:
        grant_id = str(request.args.get("grant_id") or "")
        session_id = request.session_id or ""
        if not grant_id or not session_id:
            return error_response(
                boot, self.epoch, "invalid_request", "takeover_open needs grant_id and a session_id"
            )
        async with self._transition_lock:
            standing = self._standing
            if standing is not None:
                if standing.grant_id == grant_id:
                    # Idempotent redrive: a pure echo of the ORIGINAL epoch —
                    # no drain, no rotate, no tailer re-arm. Page-blind.
                    return self._open_result(standing.target, boot, standing.epoch)
                return error_response(
                    boot, self.epoch, "takeover_active", "a takeover is already in progress"
                )
            return await self._open_fresh(grant_id, session_id, deadline, boot)

    async def _open_fresh(
        self, grant_id: str, session_id: str, deadline: float, boot: str
    ) -> BrowserResponse:
        from playwright._impl._errors import TargetClosedError

        assert self._transition_lock.locked(), "_open_fresh must hold the transition lock"
        # Ensure a page exists to take over (a fresh session gets about:blank).
        entry, _ = await self._ensure_entry(session_id)
        _unlink_manifest(self._frames_dir)  # kill the stale-frame flash / cross-takeover leak
        drain_s = max(1.0, (deadline - time.monotonic()) - _DRAIN_MARGIN_S)
        # From close_and_drain onward the gate is CLOSED. Every exit that does
        # not leave a takeover standing MUST reopen it — including a
        # CancelledError from dispatch's deadline (a BaseException the
        # `except Exception` below never sees). The `stood` flag + finally is
        # the single reopen point; the reopen is synchronous so it runs during
        # cancellation unwinding.
        stood = False
        screencast: Screencast | None = None
        try:
            if not await self._gate.close_and_drain(drain_s):
                return error_response(
                    boot, self.epoch, "action_timeout", "timed out draining in-flight actions"
                )
            # Re-read the active page AFTER the drain — it is stable now (no
            # agent action can be mid-navigation), and a drained action may have
            # left a popup frontmost.
            page = entry.active_page
            if page is None:
                return error_response(boot, self.epoch, "browser_crashed", "no page to take over")
            target = {"url": page.url, "title": await _safe_title(page)}
            signed_open = await self._signed_in_hosts()
            self.epoch += 1
            new_epoch = self.epoch
            screencast = Screencast(
                self._require_context(),
                self._frames_dir,
                boot=boot,
                epoch=new_epoch,
                next_seq=self._next_frame_seq,
            )
            await screencast.start(page)
            takeover = Takeover(
                grant_id=grant_id,
                session_id=session_id,
                epoch=new_epoch,
                opened_at=now(),
                screencast=screencast,
                injector=InputInjector(page, allow_private=self.allow_private_nav),
                target=target,
                signed_in_at_open=signed_open,
            )
            takeover.input_task = asyncio.get_running_loop().create_task(self._input_pump(takeover))
            self._standing = takeover
            stood = True  # no await between here and return — the flag is exact
            self._ensure_watchdog()
            log.info(
                "takeover open (grant=%s session=%s epoch=%d)", grant_id, session_id, new_epoch
            )
            return self._open_result(target, boot, new_epoch)
        except Exception as exc:
            # A non-cancellation failure: stop a partially-started screencast and
            # rotate again to drop any raced input. The finally reopens the gate.
            if screencast is not None:
                with contextlib.suppress(Exception):
                    await screencast.stop()
            self.epoch += 1
            if isinstance(exc, TargetClosedError):
                self._trigger_relaunch()
                return error_response(
                    boot, self.epoch, "browser_crashed", "the browser went away during open"
                )
            log.exception("takeover open failed")
            return error_response(boot, self.epoch, "internal", first_line(exc))
        finally:
            if not stood:
                self._gate.reopen()

    async def _takeover_close(
        self, request: BrowserRequest, boot: str, epoch: int
    ) -> BrowserResponse:
        grant_id = str(request.args.get("grant_id") or "")
        # outcome is OPAQUE: recorded for the log, never validated (the reaper
        # mints "expired"; a validating driver would reject reaper handbacks).
        outcome = str(request.args.get("outcome") or "done")
        async with self._transition_lock:
            standing = self._standing
            if standing is not None and standing.grant_id == grant_id:
                handback = await self._finalize(standing, outcome)
                return self._close_result(handback, boot, self.epoch)
            if standing is not None:
                return error_response(
                    boot, epoch, "grant_mismatch", "a different takeover is in progress"
                )
            cached = self._replay.get(grant_id)
            if cached is not None:
                return self._close_result(cached, boot, epoch)
            return error_response(boot, epoch, "no_takeover", f"no open takeover {grant_id}")

    async def _finalize(self, takeover: Takeover, outcome: str) -> dict[str, Any]:
        """Terminal-move a takeover: stop input+screencast (both JOINED), rotate
        the epoch, capture the handback, reopen admission, cache for replay.
        Called only under the transition lock (close, watchdog, relaunch).

        The gate reopen + standing-clear run in a finally, so a dispatch-deadline
        cancellation mid-handback (a BaseException) cannot leave the agent locked
        out with a takeover that no longer stands. A cancelled close caches no
        handback (the redrive then gets no_takeover) — but the agent is freed
        immediately rather than waiting out the idle watchdog."""
        assert self._transition_lock.locked(), "_finalize must hold the transition lock"
        try:
            if takeover.input_task is not None:
                takeover.input_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await takeover.input_task
            await takeover.screencast.stop()  # joins the pump — no frame lands after
            self.epoch += 1  # close-side rotation drops input that raced the close
            handback = await self._capture_handback(takeover)
            self._replay.put(takeover.grant_id, handback)
            log.info(
                "takeover closed (grant=%s outcome=%s epoch=%d)",
                takeover.grant_id,
                outcome,
                self.epoch,
            )
            return handback
        finally:
            self._gate.reopen()
            if self._standing is takeover:
                self._standing = None

    async def _capture_handback(self, takeover: Takeover) -> dict[str, Any]:
        entry = self._entries.get(takeover.session_id)
        page = entry.active_page if entry else None
        if entry is None or page is None:
            return {"snapshot": None, "shot_path": None, "signed_in_hosts": [], "url": None}
        shot_path: str | None = None
        with contextlib.suppress(Exception):
            # Bounded so the whole handback capture stays well inside dispatch's
            # deadline — a hung page yields a partial handback, not a cancel.
            shot_path = await actions.capture_shot(
                page, self.workspace, prefix="handback", timeout_ms=_HANDBACK_SHOT_TIMEOUT_MS
            )
        snapshot: str | None = None
        with contextlib.suppress(Exception):
            snapshot, _ = await take_snapshot(page, entry)
        url: str | None = None
        with contextlib.suppress(Exception):
            url = page.url
        signed_now = await self._signed_in_hosts()
        delta = sorted(set(signed_now) - set(takeover.signed_in_at_open))
        return {"snapshot": snapshot, "shot_path": shot_path, "signed_in_hosts": delta, "url": url}

    async def _input_pump(self, takeover: Takeover) -> None:
        tailer = SpoolTailer(self._spool_path)
        tailer.arm()  # from EOF — input written before the open is never replayed
        try:
            while True:
                for batch in tailer.poll():
                    if not self._accept_batch(takeover, batch):
                        continue
                    at = (batch.get("ts_ms") or 0) / 1000.0 or now()
                    for event in batch.get("events") or []:
                        if isinstance(event, dict):
                            with contextlib.suppress(Exception):
                                await takeover.injector.apply(event, at=at)
                    takeover.last_input = now()
                    takeover.last_seq = batch["seq"]  # _accept_batch verified int
                await asyncio.sleep(_INPUT_POLL_S)
        finally:
            tailer.close()

    def _accept_batch(self, takeover: Takeover, batch: dict[str, Any]) -> bool:
        # The driver is the enforcement authority — drop anything not for the
        # standing takeover at the current epoch beyond the last-applied seq.
        # (The isinstance guard tolerates a malformed spool line the JSON parse
        # let through — a non-int seq would else raise on the comparison.)
        if batch.get("grant_id") != takeover.grant_id or batch.get("epoch") != takeover.epoch:
            return False
        seq = batch.get("seq")
        return isinstance(seq, int) and seq > takeover.last_seq

    def _ensure_watchdog(self) -> None:
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.get_running_loop().create_task(self._watchdog())

    async def _watchdog(self) -> None:
        """Idle/unclaimed/absolute-cap auto-close. Ticks while a takeover
        stands, then exits (a later open restarts it)."""
        while not self._closing:
            await asyncio.sleep(_WATCHDOG_TICK_S)
            async with self._transition_lock:
                standing = self._standing
                if standing is None:
                    return
                marker_mtime = _mtime(self._heartbeat_marker)
                elapsed = now() - standing.opened_at
                idle = now() - standing.liveness(marker_mtime)
                unclaimed = (
                    standing.is_unclaimed(marker_mtime)
                    and elapsed >= DRIVER_TAKEOVER_UNCLAIMED_TIMEOUT_S
                )
                if (
                    unclaimed
                    or idle >= DRIVER_TAKEOVER_IDLE_TIMEOUT_S
                    or elapsed >= _ABSOLUTE_CAP_S
                ):
                    await self._finalize(standing, "expired")

    async def _retarget_takeover(self, session_id: str) -> None:
        # The input pump is not serialized against this swap, so a single human
        # gesture straddling the popup-adoption instant can split across the old
        # and new page. Bounded and self-correcting (the swap is one assignment;
        # click state resets) — a cosmetic input glitch for one gesture, not
        # worth coordinating the pump around.
        async with self._transition_lock:
            standing = self._standing
            if standing is None or standing.session_id != session_id:
                return
            entry = self._entries.get(session_id)
            page = entry.active_page if entry else None
            if page is None:
                return
            with contextlib.suppress(Exception):
                await standing.screencast.retarget(page)
            standing.injector.retarget(page)

    def _open_result(self, target: dict[str, Any], boot: str, epoch: int) -> BrowserResponse:
        # Page-blind top-level; the viewer reads and pins data.target.
        return BrowserResponse(ok=True, boot=boot, epoch=epoch, data={"target": target})

    def _close_result(self, handback: dict[str, Any], boot: str, epoch: int) -> BrowserResponse:
        # The handback IS the observation — the takeover is over, so this is not
        # page-blind; the worker reads url/snapshot/shot_path + signed_in_hosts.
        return BrowserResponse(
            ok=True,
            boot=boot,
            epoch=epoch,
            url=handback.get("url"),
            snapshot=handback.get("snapshot"),
            shot_path=handback.get("shot_path"),
            data={"signed_in_hosts": handback.get("signed_in_hosts") or []},
        )


# ── the restart ledger ────────────────────────────────────────────────────


async def _safe_title(page: Page) -> str:
    try:
        return await page.title()
    except Exception:
        return ""


def _unlink_manifest(frames_dir: Path) -> None:
    with contextlib.suppress(OSError):
        (frames_dir / "manifest.json").unlink(missing_ok=True)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_ledger(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _save_ledger(path: Path, ledger: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger), "utf-8")
    os.replace(tmp, path)
