"""Render a small Markdown subset to Telegram's HTML parse mode.

Telegram's ``HTML`` parse mode supports a fixed tag list (see
https://core.telegram.org/bots/api#html-style):

    <b> <i> <u> <s> <span class="tg-spoiler"> <a href="..."> <code>
    <pre> <pre><code class="language-X"> <blockquote>

Anything else is rejected with ``Bad Request: can't parse entities``.

The model writes Markdown naturally; this converter maps the common
inline and block constructs to that subset.  Designed to be obvious
on read rather than fully spec-compliant — Telegram's grammar is
permissive about what it accepts as long as the tag set is right
and ``<``/``>``/``&`` outside of tags are properly escaped.
"""

from __future__ import annotations

import re
from functools import partial
from html import escape

# Body-context escape: ``<``, ``>``, ``&`` only.  Quotes and apostrophes
# render literally in Telegram body text, no need to entity-encode them.
_escape_html = partial(escape, quote=False)

# Capture order matters: code fences before inline code, links before
# emphasis (so ``[text](_url_)`` doesn't get its url mangled).

_FENCE_RE = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
# Link runs AFTER the global escape, so the captured URL is already
# HTML-escaped — don't re-escape inside the substitution.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"__(.+?)__", re.DOTALL)
# Italic patterns reject leading/trailing whitespace and snake_case word
# boundaries — otherwise ``snake_case`` would render as ``snake<i>case</i>``.
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)(?!\s)(.+?)(?<!\s)(?<!\*)\*(?!\*)", re.DOTALL)
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(?!_)(?!\s)(.+?)(?<!\s)(?<!_)_(?!\w)", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Telegram-style spoiler.  Markdown has no canonical spoiler; ``||x||``
# is the de-facto convention several chat clients use.
_SPOILER_RE = re.compile(r"\|\|(.+?)\|\|", re.DOTALL)

# Sentinels avoid ``_``, ``*``, ``~``, ``|`` so the inline-emphasis
# regexes can't chew them up between escape and tag swap.
_BQ_OPEN = "\x01BQOPEN\x01"
_BQ_CLOSE = "\x01BQCLOSE\x01"


def _convert_inline(text: str) -> str:
    """Inline emphasis + links.  Operates on already-escaped text."""
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_UNDER_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _SPOILER_RE.sub(r'<span class="tg-spoiler">\1</span>', text)
    text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)
    return text


def _stash_blockquote_sentinels(text: str) -> str:
    """Group consecutive ``> `` lines into ``\x01BQOPEN\x01...\x01BQCLOSE\x01``.

    The sentinels are inserted *before* HTML escaping so the literal
    ``>`` characters at line starts get consumed (and never reach the
    escape pass that would turn them into ``&gt;``).  After escape and
    inline conversion, :func:`markdown_to_telegram_html` swaps the
    sentinels for actual ``<blockquote>`` tags.
    """
    lines = text.split("\n")
    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append(_BQ_OPEN + "\n".join(buffer) + _BQ_CLOSE)
            buffer.clear()

    for line in lines:
        if line.startswith("> "):
            buffer.append(line[2:])
        elif line == ">":
            buffer.append("")
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def markdown_to_telegram_html(text: str) -> str:
    """Convert Markdown to Telegram's HTML parse-mode subset.

    Pipeline:

    1. Stash fenced + inline code spans behind opaque placeholders so
       their contents are immune to later substitution.
    2. Convert ``> `` line prefixes to blockquote sentinels (so the
       literal ``>`` doesn't survive into the escape pass).
    3. HTML-escape the remaining text.
    4. Apply inline emphasis and link substitutions to the escaped text.
    5. Swap blockquote sentinels for real ``<blockquote>`` tags.
    6. Restore code placeholders.
    """
    placeholders: dict[str, str] = {}
    counter = 0

    def stash(html: str) -> str:
        nonlocal counter
        key = f"\x00CODE{counter}\x00"
        placeholders[key] = html
        counter += 1
        return key

    def fence_repl(m: re.Match[str]) -> str:
        lang = m.group(1)
        body = _escape_html(m.group(2))
        if lang:
            return stash(f'<pre><code class="language-{lang}">{body}</code></pre>')
        return stash(f"<pre>{body}</pre>")

    def inline_code_repl(m: re.Match[str]) -> str:
        return stash(f"<code>{_escape_html(m.group(1))}</code>")

    text = _FENCE_RE.sub(fence_repl, text)
    text = _INLINE_CODE_RE.sub(inline_code_repl, text)
    text = _stash_blockquote_sentinels(text)
    text = _escape_html(text)
    text = _convert_inline(text)
    text = text.replace(_BQ_OPEN, "<blockquote>").replace(_BQ_CLOSE, "</blockquote>")
    for key, html in placeholders.items():
        text = text.replace(key, html)
    return text
