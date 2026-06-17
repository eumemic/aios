"""Unit coverage for the markdown→mrkdwn pipeline and the hard clamps.

All pure functions, no network.  Covers the inline-grammar rewrites
(``**bold**`` → ``*bold*``, links, code-span immunity, …) and each clamp
ceiling (message 8000, section 3000, label 75, blocks ≤50) including the
ellipsis-counts-against-budget invariant.
"""

from __future__ import annotations

import pytest

from aios_slack.format import (
    LABEL_MAX_CHARS,
    MAX_BLOCKS,
    MESSAGE_MAX_CHARS,
    SECTION_TEXT_MAX_CHARS,
    clamp_blocks,
    clamp_label,
    clamp_message,
    clamp_section,
    markdown_to_mrkdwn,
    normalize_emoji,
)

# ── markdown → mrkdwn ─────────────────────────────────────────────────


def test_bold_double_star_to_single() -> None:
    assert markdown_to_mrkdwn("**bold**") == "*bold*"


def test_bold_double_underscore_to_single_star() -> None:
    assert markdown_to_mrkdwn("__bold__") == "*bold*"


def test_italic_single_star_to_underscore() -> None:
    assert markdown_to_mrkdwn("*em*") == "_em_"


def test_italic_underscore_preserved() -> None:
    assert markdown_to_mrkdwn("_em_") == "_em_"


def test_strike_double_tilde_to_single() -> None:
    assert markdown_to_mrkdwn("~~gone~~") == "~gone~"


def test_link_rewritten_to_slack_angle_form() -> None:
    assert markdown_to_mrkdwn("[Anthropic](https://anthropic.com)") == (
        "<https://anthropic.com|Anthropic>"
    )


def test_bold_and_italic_mixed() -> None:
    # ``**bold**`` becomes ``*bold*``; a separate ``*em*`` becomes ``_em_``.
    assert markdown_to_mrkdwn("**bold** and *em*") == "*bold* and _em_"


def test_inline_code_contents_immune_to_rewrites() -> None:
    # A ``**`` or ``[..](..)`` inside a code span must survive verbatim.
    assert markdown_to_mrkdwn("use `**not bold**` here") == "use `**not bold**` here"


def test_fenced_code_contents_immune_to_rewrites() -> None:
    src = "```\n**x** and [a](b)\n```"
    out = markdown_to_mrkdwn(src)
    assert "**x**" in out
    assert "[a](b)" in out


def test_fenced_code_language_tag_dropped_but_body_kept() -> None:
    out = markdown_to_mrkdwn("```python\nprint(1)\n```")
    assert "print(1)" in out
    assert out.startswith("```")


def test_blockquote_prefix_preserved() -> None:
    # Slack mrkdwn uses ``> `` line prefixes natively — pass through.
    assert markdown_to_mrkdwn("> quoted") == "> quoted"


def test_snake_case_not_italicized() -> None:
    # word-boundary guards keep ``snake_case_word`` intact.
    assert markdown_to_mrkdwn("snake_case_word") == "snake_case_word"


def test_no_html_escaping() -> None:
    # Slack mrkdwn renders ``<``/``>``/``&`` literally outside link tokens;
    # the converter must not entity-encode them (unlike telegram's HTML).
    assert markdown_to_mrkdwn("a < b & c > d") == "a < b & c > d"


# ── clamps ────────────────────────────────────────────────────────────


def test_clamp_message_under_budget_is_noop() -> None:
    text = "x" * 100
    assert clamp_message(text) == text


def test_clamp_message_truncates_to_ceiling_with_ellipsis() -> None:
    text = "x" * (MESSAGE_MAX_CHARS + 500)
    out = clamp_message(text)
    assert len(out) == MESSAGE_MAX_CHARS
    assert out.endswith("…")


def test_clamp_message_exact_boundary_is_noop() -> None:
    text = "x" * MESSAGE_MAX_CHARS
    assert clamp_message(text) == text


def test_clamp_section_ceiling() -> None:
    out = clamp_section("y" * (SECTION_TEXT_MAX_CHARS + 1))
    assert len(out) == SECTION_TEXT_MAX_CHARS
    assert out.endswith("…")


def test_clamp_label_ceiling() -> None:
    out = clamp_label("z" * (LABEL_MAX_CHARS + 10))
    assert len(out) == LABEL_MAX_CHARS
    assert out.endswith("…")


def test_clamp_label_short_is_noop() -> None:
    assert clamp_label("hello") == "hello"


def test_clamp_blocks_under_budget_is_noop() -> None:
    blocks: list[dict[str, object]] = [{"type": "section"} for _ in range(10)]
    assert clamp_blocks(blocks) == blocks


def test_clamp_blocks_truncates_to_fifty() -> None:
    blocks: list[dict[str, object]] = [{"type": "section", "i": i} for i in range(MAX_BLOCKS + 5)]
    out = clamp_blocks(blocks)
    assert len(out) == MAX_BLOCKS
    assert out[0]["i"] == 0
    assert out[-1]["i"] == MAX_BLOCKS - 1


def test_clamp_named_ceilings_match_design() -> None:
    # Guard the documented design ceilings against accidental drift.
    assert MESSAGE_MAX_CHARS == 8000
    assert SECTION_TEXT_MAX_CHARS == 3000
    assert LABEL_MAX_CHARS == 75
    assert MAX_BLOCKS == 50


# ── emoji normalization ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eyes", "eyes"),
        (":eyes:", "eyes"),
        (" :Thumbsup: ", "thumbsup"),
        (":white_check_mark:", "white_check_mark"),
        ("+1", "+1"),
        (":wave🏽:", "wave"),  # strip out-of-alphabet glyphs
    ],
)
def test_normalize_emoji(raw: str, expected: str) -> None:
    assert normalize_emoji(raw) == expected
