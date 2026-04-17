"""Tests for markdown -> Signal textStyles conversion.

The converter is lifted verbatim from jarvis; these tests are a small
regression harness to ensure we don't break it during future edits.

Signal textStyle offsets are UTF-16 code units, not Python code points.
"""

from __future__ import annotations

from aios_signal.markdown import convert_markdown_to_signal_styles


def test_empty_string() -> None:
    assert convert_markdown_to_signal_styles("") == ("", [])


def test_plain_text_no_styles() -> None:
    stripped, styles = convert_markdown_to_signal_styles("Just plain text.")
    assert stripped == "Just plain text."
    assert styles == []


def test_bold_double_asterisk() -> None:
    stripped, styles = convert_markdown_to_signal_styles("hello **world** !")
    assert stripped == "hello world !"
    # "world" starts at index 6 (after "hello "), length 5
    assert "6:5:BOLD" in styles


def test_italic_single_asterisk() -> None:
    stripped, styles = convert_markdown_to_signal_styles("*italic* text")
    assert stripped == "italic text"
    assert "0:6:ITALIC" in styles


def test_strikethrough() -> None:
    stripped, styles = convert_markdown_to_signal_styles("~~gone~~ away")
    assert stripped == "gone away"
    assert "0:4:STRIKETHROUGH" in styles


def test_inline_code() -> None:
    stripped, styles = convert_markdown_to_signal_styles("call `foo()` now")
    assert stripped == "call foo() now"
    assert "5:5:MONOSPACE" in styles


def test_fenced_code_block() -> None:
    stripped, styles = convert_markdown_to_signal_styles("before\n```\nbody\n```\nafter")
    # The stripped form drops the fence lines.
    assert "body" in stripped
    assert "```" not in stripped
    assert any(s.endswith(":MONOSPACE") for s in styles)


def test_spoiler() -> None:
    stripped, styles = convert_markdown_to_signal_styles("shh ||secret|| ok")
    assert stripped == "shh secret ok"
    assert "4:6:SPOILER" in styles


def test_code_protects_inner_markdown() -> None:
    # Bold markers inside a code span should NOT be parsed as bold.
    stripped, styles = convert_markdown_to_signal_styles("`**not bold**`")
    assert stripped == "**not bold**"
    assert all("BOLD" not in s for s in styles)
    assert any("MONOSPACE" in s for s in styles)


def test_utf16_offsets_with_emoji() -> None:
    # A 4-byte emoji is 2 UTF-16 code units.
    # Emoji 🔥 before `**bold**` — the offset of "bold" in UTF-16 is 2 (emoji) + 1 (space) = 3.
    stripped, styles = convert_markdown_to_signal_styles("🔥 **bold** end")
    assert stripped == "🔥 bold end"
    # UTF-16: emoji=2, space=1 -> bold starts at 3, length 4.
    assert "3:4:BOLD" in styles


def test_utf16_offsets_with_emoji_inside_styled_span() -> None:
    # Surrogate-pair emoji INSIDE the styled content — length must count
    # the emoji as 2 UTF-16 code units, not 1 Python code point.
    stripped, styles = convert_markdown_to_signal_styles("before **a🔥b** after")
    assert stripped == "before a🔥b after"
    # "before " = 7 UTF-16 code units; "a🔥b" = 1 + 2 + 1 = 4 code units.
    assert "7:4:BOLD" in styles


def test_snake_case_underscores_not_italic() -> None:
    # Ensure `snake_case_identifier` doesn't get parsed as italic.
    stripped, styles = convert_markdown_to_signal_styles("snake_case_identifier works")
    assert stripped == "snake_case_identifier works"
    assert all("ITALIC" not in s for s in styles)
