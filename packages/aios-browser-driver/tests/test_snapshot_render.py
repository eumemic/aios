"""Snapshot rendering: line shapes, budget markers, dialog notes."""

from __future__ import annotations

from typing import Any

from aios_browser_driver.browser_protocol import SNAPSHOT_MAX_CHARS
from aios_browser_driver.snapshot.snapshot import render


def _item(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"role": "button", "name": "Go", "tag": "button", "ref": "e1"}
    base.update(fields)
    return base


def test_basic_line_shapes() -> None:
    text, truncated = render(
        [
            _item(),
            _item(role="link", name="Docs", tag="a", ref="e2"),
            _item(role="textbox", name="Email", tag="input", ref="e3", value="a@b.c"),
            _item(role="checkbox", name="Keep me", tag="input", ref="e4", checked=True),
            _item(role="heading", name="Pricing", tag="h2", ref="e5", level=2),
            _item(role="button", name="Buy", ref="e6", disabled=True),
            _item(role="iframe", name="checkout", tag="iframe", ref=None),
        ],
        omitted=0,
        dialogs=[],
    )
    assert not truncated
    lines = text.splitlines()
    assert lines[0] == '- button "Go" [ref=e1]'
    assert lines[1] == '- link "Docs" [ref=e2]'
    assert lines[2] == '- textbox "Email": "a@b.c" [ref=e3]'
    assert lines[3] == '- checkbox "Keep me" (checked) [ref=e4]'
    assert lines[4] == '- heading (h2) "Pricing" [ref=e5]'
    assert lines[5] == '- button "Buy" (disabled) [ref=e6]'
    assert lines[6] == '- iframe "checkout"'  # unref'd: content not walked in v1


def test_select_options_render_inline() -> None:
    text, _ = render(
        [
            _item(
                role="combobox",
                name="Country",
                tag="select",
                ref="e1",
                options=[
                    {"value": "se", "label": "Sweden", "selected": True},
                    {"value": "no", "label": "Norway", "selected": False},
                ],
                optionsOmitted=3,
            )
        ],
        omitted=0,
        dialogs=[],
    )
    assert "Sweden* [value=se]" in text
    assert "Norway [value=no]" in text
    assert "… 3 more" in text


def test_element_budget_marker_sets_truncated() -> None:
    text, truncated = render([_item()], omitted=17, dialogs=[])
    assert truncated
    assert "[17 more elements omitted" in text


def test_char_budget_truncates_at_a_line_boundary() -> None:
    items = [_item(name="x" * 70, ref=f"e{i}") for i in range(1, 400)]
    text, truncated = render(items, omitted=0, dialogs=[])
    assert truncated
    assert len(text) <= SNAPSHOT_MAX_CHARS
    assert text.endswith("[snapshot truncated — the page exceeds the snapshot budget]")
    # No half-rendered item line before the marker.
    assert text.splitlines()[-2].endswith("]")


def test_dialog_notes_lead_the_snapshot() -> None:
    text, _ = render(
        [_item()],
        omitted=0,
        dialogs=['alert — "Session expired"'],
    )
    assert text.splitlines()[0] == '[dialog auto-dismissed: alert — "Session expired"]'
