"""The walker/refs/actions against a real Chromium (opt-in: ``pytest -m browser``).

Skips (not fails) when playwright's chromium build is absent — the default
unit lane never launches a browser.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from aios_browser_driver import actions
from aios_browser_driver.errors import ActionError
from aios_browser_driver.host import PageEntry
from aios_browser_driver.snapshot.refs import resolve_ref
from aios_browser_driver.snapshot.snapshot import take_snapshot

if TYPE_CHECKING:
    from playwright.async_api import Page

pytestmark = pytest.mark.browser

_PAGE_HTML = """
<!doctype html>
<html><body>
  <h2>Welcome</h2>
  <button id="go" onclick="this.textContent='Clicked'">Go</button>
  <a href="/docs">Docs</a>
  <input aria-label="Email" value="a@b.c">
  <input type="password" aria-label="Password" value="hunter2secret">
  <input type="checkbox" aria-label="Keep me" checked>
  <select aria-label="Country">
    <option value="se" selected>Sweden</option>
    <option value="no">Norway</option>
  </select>
  <button style="display:none">Hidden</button>
  <div id="shadow-host"></div>
  <iframe title="checkout" srcdoc="<button>Inner</button>"></iframe>
  <script>
    document.getElementById("shadow-host")
      .attachShadow({mode: "open"})
      .innerHTML = '<button id="shadow-btn">Shadow</button>';
  </script>
</body></html>
"""


@pytest.fixture
async def page() -> AsyncIterator[Page]:
    from playwright.async_api import Error, async_playwright

    pw = await async_playwright().start()
    try:
        try:
            browser = await pw.chromium.launch(channel="chromium", headless=True)
        except Error as exc:
            pytest.skip(f"playwright chromium not installed: {exc}")
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        yield page
        await browser.close()
    finally:
        await pw.stop()


def _entry(page: Page) -> PageEntry:
    return PageEntry(session_id="sess_test", pages=[page])


def _deadline() -> float:
    return time.monotonic() + 20.0


async def test_walker_collects_refs_and_masks_passwords(page: Page) -> None:
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, truncated = await take_snapshot(page, entry)

    assert not truncated
    assert entry.generation == 1
    assert entry.issued > 0
    assert '- heading (h2) "Welcome"' in text
    assert '- button "Go" [ref=' in text
    assert '- link "Docs" [ref=' in text
    assert '- textbox "Email": "a@b.c"' in text
    assert '- checkbox "Keep me" (checked)' in text
    assert "Sweden* [value=se]" in text
    assert '- button "Shadow"' in text  # open shadow roots are walked
    assert '- iframe "checkout"' in text
    assert "Inner" not in text  # iframe content is not walked (v1)
    assert "Hidden" not in text  # display:none excluded
    # The password FIELD is listed (the model must know it exists to route
    # around it) but its value never is.
    assert '- textbox "Password"' in text
    assert "hunter2secret" not in text


async def test_ref_resolution_and_staleness(page: Page) -> None:
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, _ = await take_snapshot(page, entry)
    ref = text.split('- button "Go" [ref=')[1].split("]")[0]

    handle = await resolve_ref(page, entry, ref)
    assert await handle.evaluate("el => el.id") == "go"

    with pytest.raises(ActionError) as info:
        await resolve_ref(page, entry, "e99999")
    assert info.value.code == "no_such_ref"

    # A newer snapshot supersedes the generation the ref was minted in.
    await take_snapshot(page, entry)
    with pytest.raises(ActionError) as info:
        await resolve_ref(page, entry, ref)
    assert info.value.code == "stale_snapshot"
    assert "superseded" in info.value.message

    # Navigation replaces the document — the registry goes with it.
    await page.set_content("<button>fresh</button>")
    with pytest.raises(ActionError) as info:
        await resolve_ref(page, entry, ref)
    assert info.value.code == "stale_snapshot"


async def test_a_ref_below_the_watermark_that_no_longer_resolves_is_stale(page: Page) -> None:
    # The flat store holds only the current generation, so any superseding
    # snapshot invalidates an older ref — but its number is ≤ issued, so it is
    # stale_snapshot (was issued), never no_such_ref.
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, _ = await take_snapshot(page, entry)
    ref = text.split('- button "Go" [ref=')[1].split("]")[0]
    for _i in range(5):
        await take_snapshot(page, entry)
    with pytest.raises(ActionError) as info:
        await resolve_ref(page, entry, ref)
    assert info.value.code == "stale_snapshot"


async def test_click_round_trip(page: Page, tmp_path: Path) -> None:
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, _ = await take_snapshot(page, entry)
    ref = text.split('- button "Go" [ref=')[1].split("]")[0]

    await actions.run(
        entry,
        page,
        "click",
        {"ref": ref, "description": "click the Go button"},
        deadline=_deadline(),
        allow_private_nav=True,
        workspace=tmp_path,
    )
    text, _ = await take_snapshot(page, entry)
    assert '- button "Clicked"' in text


async def test_type_empty_string_clears_a_field(page: Page, tmp_path: Path) -> None:
    # The tool schema has no minLength on `text`, so "" (clear the field) must
    # be accepted, not bounced as invalid_request.
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, _ = await take_snapshot(page, entry)
    ref = text.split('- textbox "Email"')[1].split("[ref=")[1].split("]")[0]
    await actions.run(
        entry,
        page,
        "type",
        {"ref": ref, "text": "", "description": "clear the email field"},
        deadline=_deadline(),
        allow_private_nav=True,
        workspace=tmp_path,
    )
    assert await page.eval_on_selector("input[aria-label=Email]", "el => el.value") == ""


async def test_password_guardrails(page: Page, tmp_path: Path) -> None:
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    text, _ = await take_snapshot(page, entry)
    ref = text.split('- textbox "Password"')[1].split("[ref=")[1].split("]")[0]

    with pytest.raises(ActionError) as info:
        await actions.run(
            entry,
            page,
            "type",
            {"ref": ref, "text": "s3cret", "description": "fill the password"},
            deadline=_deadline(),
            allow_private_nav=True,
            workspace=tmp_path,
        )
    assert info.value.code == "not_interactable"
    assert info.value.guardrail

    await page.focus("input[type=password]")
    with pytest.raises(ActionError) as info:
        await actions.run(
            entry,
            page,
            "press_key",
            {"key": "a"},
            deadline=_deadline(),
            allow_private_nav=True,
            workspace=tmp_path,
        )
    assert info.value.code == "not_interactable"
    assert info.value.guardrail


async def test_screenshot_writes_under_shots(page: Page, tmp_path: Path) -> None:
    await page.set_content(_PAGE_HTML)
    entry = _entry(page)
    shot = await actions.run(
        entry,
        page,
        "screenshot",
        {},
        deadline=_deadline(),
        allow_private_nav=True,
        workspace=tmp_path,
    )
    assert shot is not None and shot.startswith("shots/") and shot.endswith(".png")
    assert (tmp_path / shot).stat().st_size > 0
