"""Replays the human's raw input events onto the takeover's active page.

Deliberately built on playwright's ``page.mouse``/``page.keyboard`` rather than
raw CDP: playwright already owns the key-name → keycode table and the
modifier/button state machine (a held ``Shift`` from ``keyboard.down`` applies
to later clicks; ``mouse.down`` then ``mouse.move`` drags), which is the bulk of
what a hand-rolled CDP injector would re-implement. The driver adds only what
the raw event stream lacks: multi-click detection (the wire vocabulary carries
no click-count) and the current pointer position for wheel/press at a point.

No credential guardrail here — a takeover is EXACTLY when the human enters
their own password; the agent is gate-blocked and page-blind meanwhile.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

from aios_browser_driver import guards

if TYPE_CHECKING:
    from playwright.async_api import Page

_Button = Literal["left", "middle", "right"]
_MULTICLICK_WINDOW_S = 0.5
_MULTICLICK_RADIUS_PX = 4.0
# Nav waits for COMMIT only — the screencast shows the load in progress, like
# a real browser. The pump applies events serially, so a full-load wait would
# freeze the human's pointer for the whole page load.
_NAV_TIMEOUT_MS = 10_000


class InputInjector:
    def __init__(self, page: Page, *, allow_private: bool = False) -> None:
        self._page = page
        self._allow_private = allow_private
        self._x = 0.0
        self._y = 0.0
        self._last_down_at = 0.0
        self._last_down_x = 0.0
        self._last_down_y = 0.0
        self._click_count = 0

    def retarget(self, page: Page) -> None:
        """Follow the active page (a popup the human opened). Pointer/click
        state resets — a gesture never spans two pages."""
        self._page = page
        self._click_count = 0

    async def apply(self, event: dict[str, Any], *, at: float) -> None:
        """Dispatch one input event. ``at`` is the batch's wall-clock time,
        used for multi-click timing."""
        etype = event.get("type")
        if etype == "pointer_move":
            await self._move(event)
        elif etype == "pointer_down":
            await self._move(event)
            count = self._click_count_for(event, at)
            await self._page.mouse.down(button=self._button(event), click_count=count)
        elif etype == "pointer_up":
            await self._move(event)
            await self._page.mouse.up(
                button=self._button(event), click_count=self._click_count or 1
            )
        elif etype == "wheel":
            await self._move(event)
            await self._page.mouse.wheel(_num(event, "dx"), _num(event, "dy"))
        elif etype == "key_down":
            await self._page.keyboard.down(_str(event, "key"))
        elif etype == "key_up":
            await self._page.keyboard.up(_str(event, "key"))
        elif etype == "text":
            await self._page.keyboard.insert_text(_str(event, "text"))
        elif etype == "navigate":
            # The viewer's URL bar. Same public-http(s) guard as agent
            # navigation: the human types their own destinations, but this
            # browser lives inside the account's network position, and a
            # takeover must not become the way to reach what the agent
            # cannot. Guard/timeout failures raise, and the pump's
            # per-event suppress drops them — the unchanged frame stream is
            # the feedback, as in a browser whose page refused to load.
            url = _str(event, "url")
            await guards.check_url(url, allow_private=self._allow_private)
            await self._page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="commit")
        elif etype == "back":
            # History moves need no guard: every entry was committed under
            # the guard (agent nav) or by the human's own driving.
            await self._page.go_back(timeout=_NAV_TIMEOUT_MS, wait_until="commit")
        elif etype == "forward":
            await self._page.go_forward(timeout=_NAV_TIMEOUT_MS, wait_until="commit")
        elif etype == "reload":
            await self._page.reload(timeout=_NAV_TIMEOUT_MS, wait_until="commit")
        # An unknown type is ignored — the vocabulary is pinned, and a human's
        # live driving must not stall on one odd line.

    async def _move(self, event: dict[str, Any]) -> None:
        if event.get("x") is not None and event.get("y") is not None:
            self._x, self._y = _num(event, "x"), _num(event, "y")
        await self._page.mouse.move(self._x, self._y)

    def _button(self, event: dict[str, Any]) -> _Button:
        button = event.get("button")
        return cast(_Button, button) if button in ("left", "middle", "right") else "left"

    def _click_count_for(self, event: dict[str, Any], at: float) -> int:
        x, y = self._x, self._y
        near = math.hypot(x - self._last_down_x, y - self._last_down_y) <= _MULTICLICK_RADIUS_PX
        if at - self._last_down_at <= _MULTICLICK_WINDOW_S and near and self._click_count:
            self._click_count = min(3, self._click_count + 1)
        else:
            self._click_count = 1
        self._last_down_at, self._last_down_x, self._last_down_y = at, x, y
        return self._click_count


def _num(event: dict[str, Any], key: str) -> float:
    value = event.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value if isinstance(value, str) else ""
