"""The eleven action ops, mapped onto playwright.

Each handler acts on the session's active page and raises
:class:`~aios_browser_driver.errors.ActionError` for classified failures;
the host's ``handle`` boundary turns those into the ``ok: false`` envelope.
Audit-only arguments (``description``; ``modifiers`` where the action has no
chord semantics) are accepted and ignored — never a validation error.

The credential guardrail lives here: typing into a password field, and
pressing keys while one is focused, is refused (``not_interactable``) — the
account owner signs in via takeover, the agent never handles credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from ulid import ULID

from aios_browser_driver import guards
from aios_browser_driver.browser_protocol import SHOTS_DIR
from aios_browser_driver.errors import (
    ActionError,
    first_line,
    require_int,
    require_number,
    require_str,
)
from aios_browser_driver.snapshot.refs import resolve_ref

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Page

    from aios_browser_driver.host import PageEntry

_Modifier = Literal["Alt", "Control", "Meta", "Shift"]
_Button = Literal["left", "middle", "right"]
_MODIFIER_NAMES = frozenset(get_args(_Modifier))
_BUTTONS = frozenset(get_args(_Button))
_DIRECTIONS = frozenset({"up", "down", "left", "right"})

# The fixed viewport — the one definition; the host launches at this size and
# "scroll one page" means exactly it.
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
_WHEEL_STEP_PX = 120
_SHOTS_SUBDIR = PurePosixPath(SHOTS_DIR).name

# Keep playwright's own timeouts strictly INSIDE dispatch's deadline so an
# element op that exhausts its budget reports not_interactable (specific)
# rather than racing dispatch's action_timeout (generic) at the same instant.
_INNER_MARGIN_MS = 500.0

_IS_PASSWORD_JS = "el => el instanceof HTMLInputElement && el.type === 'password'"
# Pierce open shadow roots to find the actually-focused element.
_FOCUSED_PASSWORD_JS = """() => {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  return !!el && el instanceof HTMLInputElement && el.type === "password";
}"""

_PASSWORD_REFUSAL = (
    "refused: this field takes a password. The agent never types credentials — "
    "the account owner signs in directly via a takeover."
)


def _remaining_ms(deadline: float) -> float:
    return max(250.0, (deadline - time.monotonic()) * 1000.0 - _INNER_MARGIN_MS)


def _modifiers(args: dict[str, Any]) -> list[_Modifier]:
    raw = args.get("modifiers") or []
    if not isinstance(raw, list):
        raise ActionError("invalid_request", "'modifiers' must be a list")
    return [cast(_Modifier, m) for m in raw if m in _MODIFIER_NAMES]


@contextlib.asynccontextmanager
async def _holding(page: Page, modifiers: list[_Modifier]) -> AsyncIterator[None]:
    """Hold modifier keys around a raw mouse gesture (which has no
    ``modifiers`` parameter of its own).

    The presses are INSIDE the try so a cancellation mid-setup still releases
    what was already pressed — a modifier left latched would silently apply to
    every later action on the page."""
    pressed: list[_Modifier] = []
    try:
        for m in modifiers:
            await page.keyboard.down(m)
            pressed.append(m)
        yield
    finally:
        for m in reversed(pressed):
            with contextlib.suppress(Exception):
                await page.keyboard.up(m)


async def run(
    entry: PageEntry,
    page: Page,
    op: str,
    args: dict[str, Any],
    *,
    deadline: float,
    allow_private_nav: bool,
    workspace: Path,
) -> str | None:
    """Run one action op; return the plane-relative shot path (screenshot only)."""
    if op == "snapshot":
        return None  # the host's post-action snapshot IS the observation
    if op == "navigate":
        await _navigate(page, args, deadline=deadline, allow_private=allow_private_nav)
    elif op == "click":
        await _click(entry, page, args, deadline=deadline)
    elif op == "click_xy":
        await _click_xy(page, args)
    elif op == "type":
        await _type(entry, page, args, deadline=deadline)
    elif op == "press_key":
        await _press_key(page, args)
    elif op == "scroll":
        await _scroll(entry, page, args)
    elif op == "drag":
        await _drag(entry, page, args)
    elif op == "hover":
        await _hover(entry, page, args, deadline=deadline)
    elif op == "select_option":
        await _select_option(entry, page, args, deadline=deadline)
    elif op == "screenshot":
        return await _screenshot(page, args, deadline=deadline, workspace=workspace)
    else:  # pragma: no cover — dispatch validated op against BrowserOp already
        raise ActionError("unknown_op", f"unknown action op {op!r}")
    await _settle(page, deadline=deadline)
    return None


async def _settle(page: Page, *, deadline: float) -> None:
    """Give the page a beat to react before the snapshot.

    Double-rAF plus a fixed 100ms: two frames flush layout and most
    same-document reactions. If the action navigated, the evaluate lands in a
    destroyed context — fall through to waiting for the new document's load
    state instead. Best-effort by design: a page that never settles should
    still be observed as-is, not fail the action that succeeded.
    """
    try:
        await page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        await asyncio.sleep(0.1)
    except Exception:
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("load", timeout=min(5_000.0, _remaining_ms(deadline)))


async def _navigate(
    page: Page, args: dict[str, Any], *, deadline: float, allow_private: bool
) -> None:
    from playwright._impl._errors import TargetClosedError
    from playwright.async_api import Error as PlaywrightError

    url = require_str(args, "url")
    await guards.check_url(url, allow_private=allow_private)
    try:
        await page.goto(url, timeout=_remaining_ms(deadline))
    except TargetClosedError:
        raise  # a crash mid-navigation is browser_crashed, not a nav failure
    except PlaywrightError as exc:
        raise ActionError("navigation_failed", first_line(exc)) from exc
    # Post-commit re-check: a redirect chain may have landed somewhere the
    # pre-flight check never saw. On violation, get off the page before
    # reporting — its content must not be observed.
    try:
        await guards.check_url(page.url, allow_private=allow_private)
    except ActionError:
        try:
            await page.goto("about:blank")
        except PlaywrightError:
            # If we cannot even blank it, the page is still on the blocked
            # address — close it so nothing downstream can snapshot it (the
            # session recreates a fresh page on its next action).
            with contextlib.suppress(PlaywrightError):
                await page.close()
        raise ActionError(
            "navigation_failed",
            f"{url} redirected to a blocked address; the page was closed",
        ) from None


async def _click(entry: PageEntry, page: Page, args: dict[str, Any], *, deadline: float) -> None:
    handle = await resolve_ref(page, entry, require_str(args, "ref"))
    await handle.click(modifiers=_modifiers(args), timeout=_remaining_ms(deadline))


async def _click_xy(page: Page, args: dict[str, Any]) -> None:
    x = require_number(args, "x")
    y = require_number(args, "y")
    button = args.get("button") or "left"
    if button not in _BUTTONS:
        raise ActionError("invalid_request", f"'button' must be one of {sorted(_BUTTONS)}")
    count = require_int(args, "count", lo=1, hi=3) if args.get("count") is not None else 1
    async with _holding(page, _modifiers(args)):
        await page.mouse.click(x, y, button=cast(_Button, button), click_count=count)


async def _type(entry: PageEntry, page: Page, args: dict[str, Any], *, deadline: float) -> None:
    handle = await resolve_ref(page, entry, require_str(args, "ref"))
    text = require_str(args, "text", allow_empty=True)  # "" clears the field
    if await handle.evaluate(_IS_PASSWORD_JS):
        raise ActionError("not_interactable", _PASSWORD_REFUSAL, guardrail=True)
    await handle.fill(text, timeout=_remaining_ms(deadline))
    if args.get("submit"):
        await handle.press("Enter", timeout=_remaining_ms(deadline))


async def _press_key(page: Page, args: dict[str, Any]) -> None:
    key = require_str(args, "key")
    if await page.evaluate(_FOCUSED_PASSWORD_JS):
        raise ActionError("not_interactable", _PASSWORD_REFUSAL, guardrail=True)
    await page.keyboard.press(key)


async def _scroll(entry: PageEntry, page: Page, args: dict[str, Any]) -> None:
    direction = require_str(args, "direction")
    if direction not in _DIRECTIONS:
        raise ActionError("invalid_request", f"'direction' must be one of {sorted(_DIRECTIONS)}")
    horizontal = direction in ("left", "right")
    if args.get("amount") is not None:
        magnitude = require_int(args, "amount", lo=1, hi=20) * _WHEEL_STEP_PX
    else:
        magnitude = VIEWPORT_WIDTH if horizontal else VIEWPORT_HEIGHT
    sign = 1 if direction in ("down", "right") else -1
    dx, dy = (sign * magnitude, 0) if horizontal else (0, sign * magnitude)

    at = args.get("at")
    if isinstance(at, dict):
        await page.mouse.move(require_number(at, "x"), require_number(at, "y"))
        await page.mouse.wheel(dx, dy)
        return
    ref = args.get("ref")
    if isinstance(ref, str) and ref:
        handle = await resolve_ref(page, entry, ref)
        await handle.evaluate(
            "(el, [dx, dy]) => el.scrollBy({left: dx, top: dy, behavior: 'instant'})", [dx, dy]
        )
        return
    await page.mouse.wheel(dx, dy)


async def _endpoint(
    entry: PageEntry, page: Page, args: dict[str, Any], key: str
) -> tuple[float, float]:
    spec = args.get(key)
    if not isinstance(spec, dict):
        raise ActionError("invalid_request", f"{key!r} must be an object with a ref or x/y")
    ref = spec.get("ref")
    if isinstance(ref, str) and ref:
        handle = await resolve_ref(page, entry, ref)
        box = await handle.bounding_box()
        if box is None:
            raise ActionError("not_interactable", f"{ref} has no on-screen position")
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    return require_number(spec, "x"), require_number(spec, "y")


async def _drag(entry: PageEntry, page: Page, args: dict[str, Any]) -> None:
    start = await _endpoint(entry, page, args, "from")
    end = await _endpoint(entry, page, args, "to")
    waypoints: list[tuple[float, float]] = []
    for point in args.get("path") or []:
        if not isinstance(point, dict):
            raise ActionError("invalid_request", "'path' must be a list of {x, y} points")
        waypoints.append((require_number(point, "x"), require_number(point, "y")))

    async with _holding(page, _modifiers(args)):
        await page.mouse.move(*start)
        await page.mouse.down()
        try:
            for x, y in [*waypoints, end]:
                await page.mouse.move(x, y, steps=12)
        finally:
            # Release the button even on cancellation — a page left mid-drag
            # swallows the next click.
            with contextlib.suppress(Exception):
                await page.mouse.up()


async def _hover(entry: PageEntry, page: Page, args: dict[str, Any], *, deadline: float) -> None:
    ref = args.get("ref")
    if isinstance(ref, str) and ref:
        handle = await resolve_ref(page, entry, ref)
        await handle.hover(timeout=_remaining_ms(deadline))
        return
    if "x" in args and "y" in args:
        await page.mouse.move(require_number(args, "x"), require_number(args, "y"))
        return
    raise ActionError("invalid_request", "hover needs a 'ref' or viewport 'x'/'y'")


async def _select_option(
    entry: PageEntry, page: Page, args: dict[str, Any], *, deadline: float
) -> None:
    handle = await resolve_ref(page, entry, require_str(args, "ref"))
    values = args.get("values")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise ActionError("invalid_request", "'values' must be a non-empty list of strings")
    await handle.select_option(cast(list[str], values), timeout=_remaining_ms(deadline))


async def _screenshot(page: Page, args: dict[str, Any], *, deadline: float, workspace: Path) -> str:
    name = f"shot-{ULID()!s}.png"
    shots_dir = workspace / _SHOTS_SUBDIR
    shots_dir.mkdir(parents=True, exist_ok=True)
    await page.screenshot(
        path=shots_dir / name,
        full_page=bool(args.get("full_page")),
        timeout=_remaining_ms(deadline),
    )
    return f"{_SHOTS_SUBDIR}/{name}"
