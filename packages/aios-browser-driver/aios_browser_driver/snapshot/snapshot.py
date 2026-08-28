"""Take and render the budgeted a11y snapshot.

``take_snapshot`` injects :mod:`walker.js <aios_browser_driver.snapshot>` into
the page's main frame, advances the entry's generation, and renders the
returned items to the opaque text the model reads. Budgets are the protocol's
``SNAPSHOT_MAX_ELEMENTS`` / ``SNAPSHOT_MAX_CHARS``; auto-dismissed dialogs
ride the snapshot text (nothing merged reads action-response ``data``).
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any

from aios_browser_driver.browser_protocol import SNAPSHOT_MAX_CHARS, SNAPSHOT_MAX_ELEMENTS

if TYPE_CHECKING:
    from playwright.async_api import Page

    from aios_browser_driver.host import PageEntry

_WALKER_JS = (resources.files("aios_browser_driver.snapshot") / "walker.js").read_text("utf-8")

_ELEMENT_BUDGET_MARKER = "[{omitted} more elements omitted — the page exceeds the snapshot budget]"
_CHAR_BUDGET_MARKER = "[snapshot truncated — the page exceeds the snapshot budget]"


async def take_snapshot(page: Page, entry: PageEntry) -> tuple[str, bool]:
    """Snapshot ``page`` under a fresh generation; return ``(text, truncated)``."""
    gen = entry.generation + 1
    result: dict[str, Any] = await page.evaluate(
        _WALKER_JS, [gen, entry.issued + 1, SNAPSHOT_MAX_ELEMENTS]
    )
    entry.generation = gen
    entry.issued += int(result["assigned"])
    return render(
        list(result["items"]),
        omitted=int(result["omitted"]),
        dialogs=entry.drain_dialogs(),
    )


def render(items: list[dict[str, Any]], *, omitted: int, dialogs: list[str]) -> tuple[str, bool]:
    """Render walker items (plus any auto-dismissed dialogs) to snapshot text."""
    lines = [f"[dialog auto-dismissed: {d}]" for d in dialogs]
    lines.extend(_render_item(item) for item in items)
    if omitted:
        lines.append(_ELEMENT_BUDGET_MARKER.format(omitted=omitted))
    truncated = omitted > 0

    text = "\n".join(lines)
    if len(text) > SNAPSHOT_MAX_CHARS:
        keep = text[: SNAPSHOT_MAX_CHARS - len(_CHAR_BUDGET_MARKER) - 1]
        text = keep.rsplit("\n", 1)[0] + "\n" + _CHAR_BUDGET_MARKER
        truncated = True
    return text, truncated


def _render_item(item: dict[str, Any]) -> str:
    role = str(item.get("role") or "generic")
    parts = [f"- {role}"]
    if role == "heading" and item.get("level"):
        parts.append(f"(h{item['level']})")
    name_part = f'"{item.get("name") or ""}"'
    if item.get("value") is not None:
        name_part += f': "{item["value"]}"'
    parts.append(name_part)
    states = [
        state
        for state, present in (
            ("checked", item.get("checked") is True),
            ("unchecked", item.get("checked") is False),
            ("disabled", item.get("disabled") is True),
            ("expanded", item.get("expanded") is True),
            ("collapsed", item.get("expanded") is False),
        )
        if present
    ]
    if states:
        parts.append(f"({', '.join(states)})")
    options = item.get("options")
    if options:
        rendered = ", ".join(
            f"{opt.get('label') or opt.get('value')!s}{'*' if opt.get('selected') else ''}"
            + (f" [value={opt['value']}]" if opt.get("value") != opt.get("label") else "")
            for opt in options
        )
        more = f", … {item['optionsOmitted']} more" if item.get("optionsOmitted") else ""
        parts.append(f"[options: {rendered}{more}]")
    if item.get("ref"):
        parts.append(f"[ref={item['ref']}]")
    return " ".join(parts)
