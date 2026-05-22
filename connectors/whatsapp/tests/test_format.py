"""Tests for the CommonMark → WhatsApp inline converter."""

from __future__ import annotations

import pytest

from aios_whatsapp.format import markdown_to_whatsapp


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Bold uses the asterisk form only — the __underscore__ form
        # is intentionally not supported so Python dunders like
        # __init__ stay verbatim.
        ("**bold**", "*bold*"),
        ("hello **world**", "hello *world*"),
        # Italic single-asterisk → underscore.
        ("*italic*", "_italic_"),
        ("hello *world*", "hello _world_"),
        # Italic single-underscore stays underscore-shaped.
        ("_italic_", "_italic_"),
        # Strike: double-tilde → single-tilde.
        ("~~strike~~", "~strike~"),
        # Inline backtick promotes to monospace fence.
        ("`code`", "```code```"),
        # Already-fenced code passes through unchanged.
        ("```already```", "```already```"),
        # No formatting → identity.
        ("plain text", "plain text"),
    ],
)
def test_markdown_to_whatsapp_inline(source: str, expected: str) -> None:
    assert markdown_to_whatsapp(source) == expected


def test_markdown_to_whatsapp_does_not_italicize_snake_case() -> None:
    # ``print_hello`` is a Python identifier, not Markdown italic
    # spanning ``hello``.  Pre-fix regression would render as
    # ``print<i>hello</i>``.
    assert markdown_to_whatsapp("call print_hello() in this") == "call print_hello() in this"


def test_markdown_to_whatsapp_does_not_italicize_multiplied_asterisks() -> None:
    # ``a * b`` is arithmetic, not italic — the asterisk-italic regex
    # requires no surrounding whitespace on the inside.
    assert markdown_to_whatsapp("result = a * b") == "result = a * b"


def test_markdown_to_whatsapp_preserves_emphasis_inside_code() -> None:
    # Code spans must NOT be re-processed by the emphasis regexes,
    # otherwise pre-fix ``**foo**`` inside a backtick would lose the
    # asterisks and the rendered output would be wrong.
    src = "see `**markdown**` syntax"
    assert markdown_to_whatsapp(src) == "see ```**markdown**``` syntax"


def test_markdown_to_whatsapp_handles_fence_with_inner_emphasis() -> None:
    src = "```\n**not bold here**\n```"
    assert markdown_to_whatsapp(src) == "```\n**not bold here**\n```"


def test_markdown_to_whatsapp_combined_emphasis() -> None:
    src = "**bold** and *italic* and ~~strike~~"
    assert markdown_to_whatsapp(src) == "*bold* and _italic_ and ~strike~"


def test_markdown_to_whatsapp_preserves_python_dunders() -> None:
    # Pre-fix regression: __init__ rendered as *init* (bold).  The
    # bold-underscore pattern now requires non-word boundaries on
    # both sides, matching italic's defensive lookarounds.
    assert (
        markdown_to_whatsapp("Call __init__ in your subclass") == "Call __init__ in your subclass"
    )
    assert markdown_to_whatsapp("__dunder__") == "__dunder__"


def test_markdown_to_whatsapp_preserves_snake_case_bold_asterisks() -> None:
    # Same defensive boundary for **foo** when foo is wedged
    # between word chars — protects identifiers like `**name`.
    # The regex's word-boundary look-around prevents an inner
    # word-char from triggering bold; passing through.
    assert markdown_to_whatsapp("snake_case**name**suffix") == "snake_case**name**suffix"


def test_markdown_to_whatsapp_sentinel_collision_falls_back() -> None:
    # Adversarial input containing the literal sentinel pattern but
    # NO actual code fences / bolds: the restore pass must not
    # IndexError; it falls back to leaving the literal match
    # in place.
    src = "before \x01SLOT\x015\x01ENDSLOT\x01 after"
    # No code spans, no bolds stashed → placeholders is empty,
    # idx=5 is out of range → restore returns the original
    # match unchanged.
    assert markdown_to_whatsapp(src) == src
