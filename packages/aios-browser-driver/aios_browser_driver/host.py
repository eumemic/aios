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
    PROFILE_DIR,
    BrowserError,
    BrowserRequest,
    BrowserResponse,
    BrowserTab,
)
from aios_browser_driver.errors import ActionError, error_response, first_line, require_str
from aios_browser_driver.hosts import normalize_host, signed_in_hosts
from aios_browser_driver.snapshot.snapshot import take_snapshot

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

    @property
    def _ledger_path(self) -> Path:
        return self.workspace / _LEDGER_RELPATH

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
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
        if self._pw is not None:
            with contextlib.suppress(Exception):
                await self._pw.stop()

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

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

        if request.op in ("takeover_open", "takeover_close"):
            raise NotImplementedError(request.op)  # the takeover PR lands these
        await self._ready.wait()
        # Stamp the envelope from boot/epoch read at entry, so a relaunch
        # racing this request can't pair a new boot with old page data.
        boot, epoch = self.boot, self.epoch
        try:
            if request.op == "status":
                return await self._status(boot, epoch)
            if request.op == "revoke_site":
                return await self._revoke_site(request.args, boot, epoch)
            if not request.session_id:
                raise ActionError("invalid_request", f"{request.op} requires a session_id")
            async with self._session_lock(request.session_id):
                entry, restarted = await self._ensure_entry(request.session_id)
                self._last_session = request.session_id
                return await self._run_action(entry, request, restarted, deadline, boot, epoch)
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


# ── the restart ledger ────────────────────────────────────────────────────


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
