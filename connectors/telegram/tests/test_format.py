"""Markdown → Telegram-HTML conversion smoke tests.

The converter handles the subset Telegram's HTML parse mode supports
(``b``, ``i``, ``u``, ``s``, ``a``, ``code``, ``pre``, ``blockquote``,
spoiler).  We don't aim for full Markdown spec — only that what models
typically write renders correctly.
"""

from __future__ import annotations

from aios_telegram.format import markdown_to_telegram_html


def test_plain_text_passthrough() -> None:
    assert markdown_to_telegram_html("hello world") == "hello world"


def test_bold_double_asterisks() -> None:
    assert markdown_to_telegram_html("**hi**") == "<b>hi</b>"


def test_bold_double_underscores() -> None:
    assert markdown_to_telegram_html("__hi__") == "<b>hi</b>"


def test_italic_single_asterisk() -> None:
    assert markdown_to_telegram_html("*hi*") == "<i>hi</i>"


def test_italic_single_underscore() -> None:
    assert markdown_to_telegram_html("_hi_") == "<i>hi</i>"


def test_strikethrough() -> None:
    assert markdown_to_telegram_html("~~gone~~") == "<s>gone</s>"


def test_spoiler() -> None:
    assert markdown_to_telegram_html("||secret||") == '<span class="tg-spoiler">secret</span>'


def test_inline_code() -> None:
    assert markdown_to_telegram_html("`x = 1`") == "<code>x = 1</code>"


def test_inline_code_escapes_html_chars() -> None:
    assert (
        markdown_to_telegram_html("`a < b && c > d`") == "<code>a &lt; b &amp;&amp; c &gt; d</code>"
    )


def test_fenced_code_block_no_language() -> None:
    md = "```\nprint(x)\n```"
    assert markdown_to_telegram_html(md) == "<pre>print(x)\n</pre>"


def test_fenced_code_block_with_language() -> None:
    md = "```python\nprint(x)\n```"
    assert (
        markdown_to_telegram_html(md)
        == '<pre><code class="language-python">print(x)\n</code></pre>'
    )


def test_link() -> None:
    assert markdown_to_telegram_html("[home](https://x.io)") == '<a href="https://x.io">home</a>'


def test_link_url_html_escaped() -> None:
    assert (
        markdown_to_telegram_html("[search](https://x.io?q=a&b=c)")
        == '<a href="https://x.io?q=a&amp;b=c">search</a>'
    )


def test_blockquote_single_line() -> None:
    assert markdown_to_telegram_html("> quoted") == "<blockquote>quoted</blockquote>"


def test_blockquote_multiline() -> None:
    md = "> line 1\n> line 2"
    assert markdown_to_telegram_html(md) == "<blockquote>line 1\nline 2</blockquote>"


def test_html_chars_in_plain_text_are_escaped() -> None:
    assert markdown_to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_markdown_inside_code_is_not_processed() -> None:
    """Markdown emphasis tokens inside code stay literal."""
    assert markdown_to_telegram_html("`**not bold**`") == "<code>**not bold**</code>"


def test_mixed_inline_styles() -> None:
    assert (
        markdown_to_telegram_html("**bold** and *italic* and `code`")
        == "<b>bold</b> and <i>italic</i> and <code>code</code>"
    )
