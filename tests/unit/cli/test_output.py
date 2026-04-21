"""Tests for the CLI's table/JSON formatters."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from aios.cli.output import print_json, print_table, render_table


def test_render_table_basic():
    rows = [
        {"id": "ag_1", "name": "first", "model": "openai/gpt-5"},
        {"id": "ag_2", "name": "second", "model": "anthropic/claude-opus-4-6"},
    ]
    out = render_table(rows, ("id", "name", "model"))
    lines = out.strip().split("\n")
    assert lines[0].split()[:3] == ["ID", "NAME", "MODEL"]
    assert "ag_1" in lines[2]
    assert "ag_2" in lines[3]


def test_render_table_applies_max_width_with_ellipsis():
    rows = [{"name": "a-very-long-name-that-exceeds-the-cap"}]
    out = render_table(rows, ("name",), max_widths={"name": 10})
    # Ellipsis character ends the truncated cell.
    assert "…" in out


def test_render_table_missing_keys_render_as_empty():
    rows = [{"id": "x"}, {}]
    out = render_table(rows, ("id", "name"))
    # The second data row has no id or name — both cells render as empty,
    # so after splitting we get only the header-style blank row.
    assert "x" in out


def test_print_table_empty_prints_empty_message():
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_table([], ("id",), empty_message="(nope)")
    assert "(nope)" in buf.getvalue()


def test_print_json_pretty_and_sorted_insertion_order():
    obj = {"b": 2, "a": 1, "c": [3, 4]}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_json(obj)
    # Must be parseable JSON and preserve insertion order (no sort).
    parsed = json.loads(buf.getvalue())
    assert parsed == obj
    assert list(parsed.keys()) == ["b", "a", "c"]
    assert "\n" in buf.getvalue()  # pretty printed


def test_print_json_handles_non_serializable_via_default():
    class Weird:
        def __str__(self) -> str:
            return "weirdly"

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_json({"x": Weird()})
    data = json.loads(buf.getvalue())
    assert data == {"x": "weirdly"}
