"""The lock-free control ops (_peek, _status) against a page that closes or
crashes mid-read. These exercise ``BrowserHost`` WITHOUT launching Chromium: a
fake ``Page`` whose read access raises ``TargetClosedError`` stands in for a
self-closing popup, a renderer crash, or driver-process death — Playwright
raises the SAME ``TargetClosedError`` class for all three, so a lock-free
caller cannot distinguish them.

The contract under test is the one ``_status`` already honored and ``_peek``
was introduced without (commit c43fd6e6): a lock-free control op DEGRADES —
returns an ``ok`` envelope with null/empty data — instead of letting the
``TargetClosedError`` reach ``handle``'s ``except TargetClosedError`` path,
which calls ``_trigger_relaunch`` (boot rotation, ``_entries.clear()``, a
Chromium relaunch). Letting a read-only poll tear down every session's page is
the bug; degrading and letting the next action (under the session lock) detect
genuine driver death is the fix.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import pytest
from aios_browser_driver.browser_protocol import BrowserRequest
from aios_browser_driver.host import BrowserHost, PageEntry
from playwright._impl._errors import TargetClosedError


def _make_host(tmp_path: Path) -> BrowserHost:
    # ``handle`` awaits ``_ready`` before dispatching; ``start()`` would set it
    # AND launch Chromium — the control-op route needs only the gate open, so
    # we never start, we just open the gate. The workspace subdirs match the
    # real fixture so no incidental path access trips.
    for sub in ("profile", "frames", "shots", "downloads", "input"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    h = BrowserHost(workspace=tmp_path)
    h._ready.set()
    return h


def _install(host: BrowserHost, page: Any, session_id: str = "s1") -> None:
    host._entries[session_id] = PageEntry(session_id=session_id, pages=[page])  # type: ignore[arg-type]
    host._last_session = session_id


async def _peek(host: BrowserHost, session_id: str | None = "s1") -> Any:
    request = BrowserRequest(op="peek", session_id=session_id, args={})
    return await host.handle(request, deadline=time.monotonic() + 5)


async def _status(host: BrowserHost) -> Any:
    request = BrowserRequest(op="status", session_id=None, args={})
    return await host.handle(request, deadline=time.monotonic() + 5)


def _assert_no_relaunch(host: BrowserHost, boot_before: str) -> None:
    # The bug's signature: ``_trigger_relaunch`` clears ``_ready``, creates a
    # ``_relaunch_task``, and (inside that task) rotates ``boot`` and clears
    # ``_entries``. A degraded peek touches none of these.
    assert host._relaunch_task is None, "control op must not schedule a relaunch"
    assert host._ready.is_set(), "control op must not clear the ready flag"
    assert host.boot == boot_before, "control op must not rotate boot"


class _DyingPage:
    """A fake page that raises ``TargetClosedError`` at a chosen access point —
    the single exception Playwright raises for page close, renderer crash, AND
    driver-process death (the caller cannot distinguish them)."""

    _MSG = "Target page, context or browser has been closed"

    def __init__(self, fail_at: str) -> None:
        self._fail_at = fail_at

    def is_closed(self) -> bool:
        return False

    @property
    def url(self) -> str:
        if self._fail_at == "url":
            raise TargetClosedError(self._MSG)
        return "https://example.test/"

    @property
    def viewport_size(self) -> dict[str, int]:
        if self._fail_at == "viewport":
            raise TargetClosedError(self._MSG)
        return {"width": 640, "height": 480}

    async def screenshot(self, **kwargs: Any) -> bytes:
        if self._fail_at == "screenshot":
            raise TargetClosedError(self._MSG)
        return b"\xff\xd8\xff\xe0"

    async def title(self) -> str:
        if self._fail_at == "title":
            raise TargetClosedError(self._MSG)
        return "Example Domain"


class _FakeContext:
    """The context ``_status`` reads cookies from — only the ``cookies``
    coroutine is exercised, so the rest of the BrowserContext surface is left
    unstubbed."""

    async def cookies(self) -> list[Any]:
        return []


class _DeadContext:
    """A fake context whose ``new_page`` fails — stands in for driver-process
    death on the next action's ``_ensure_entry`` recreate."""

    async def new_page(self) -> Any:
        raise TargetClosedError("the browser went away")


@pytest.mark.parametrize("fail_at", ["url", "screenshot", "viewport", "title"])
async def test_peek_degrades_when_the_page_dies_mid_read(tmp_path: Path, fail_at: str) -> None:
    # The primary vector: a self-closing popup (the active page) raises
    # TargetClosedError mid-peek. Before the fix this propagated to handle's
    # except-TargetClosedError arm and tore down all sessions; now it degrades.
    host = _make_host(tmp_path)
    boot_before = host.boot
    _install(host, _DyingPage(fail_at=fail_at))

    resp = await _peek(host)

    assert resp.ok, f"peek should degrade, not crash (fail_at={fail_at})"
    assert resp.error is None, f"peek must not surface a browser_crashed error (fail_at={fail_at})"
    assert resp.data == {"page": None}
    assert resp.boot == boot_before
    _assert_no_relaunch(host, boot_before)
    # The dying page is left in the registry untouched — the next action
    # owns the recovery (recreate on close, relaunch on driver death).
    assert "s1" in host._entries


async def test_status_also_degrades_when_the_page_dies_mid_read(tmp_path: Path) -> None:
    # _status is the sibling control op whose posture _peek now mirrors; this
    # guards that the shared contract isn't regressed for either op. _status
    # suppresses per page access (url and title in separate blocks), so the
    # failing read degrades to None without reaching handle's relaunch path —
    # the exact posture _peek now adopts (atomically, for its page dict). The
    # signed_in_hosts read needs a live context, stood in here.
    host = _make_host(tmp_path)
    host._context = _FakeContext()  # type: ignore[assignment]
    boot_before = host.boot
    _install(host, _DyingPage(fail_at="title"))

    resp = await _status(host)

    assert resp.ok and resp.error is None
    assert resp.title is None  # the title read raised TargetClosedError
    assert resp.data["signed_in_hosts"] == []
    _assert_no_relaunch(host, boot_before)


async def test_driver_death_relaunches_on_the_next_action_not_on_peek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A peek swallows driver death (degrades); genuine death is caught by the
    # NEXT action, which runs under the session lock and recreates the page
    # through ``_ensure_entry`` → ``_require_context().new_page()``. When THAT
    # raises TargetClosedError, handle's except arm relaunches — exactly the
    # deferred-detection trade-off _status already relies on. The relaunch is
    # neutralized (no chromium spawn) so we assert the scheduling, not a real
    # launch.
    host = _make_host(tmp_path)
    boot_before = host.boot
    _install(host, _DyingPage(fail_at="screenshot"))

    # The read-only poll degrades; no relaunch.
    peek = await _peek(host)
    assert peek.ok and peek.data == {"page": None}
    assert host._relaunch_task is None
    assert host.boot == boot_before

    # Drop the entry so _ensure_entry recreates it; the dead context makes that
    # recreate raise TargetClosedError — driver death on the next action.
    host._context = _DeadContext()  # type: ignore[assignment]
    host._entries.pop("s1", None)

    async def _noop_launch(self: BrowserHost) -> None:
        return None

    monkeypatch.setattr(BrowserHost, "_launch", _noop_launch)
    snap = await host.handle(
        BrowserRequest(op="snapshot", session_id="s1", args={}),
        deadline=time.monotonic() + 5,
    )

    assert not snap.ok and snap.error is not None
    assert snap.error.code == "browser_crashed"
    assert host._relaunch_task is not None  # the next action DID relaunch
    # Let the (no-op) relaunch finish and confirm boot rotated — the genuine
    # death signal a read-only poll must never emit.
    with contextlib.suppress(Exception):
        await host._relaunch_task
    assert host.boot != boot_before
