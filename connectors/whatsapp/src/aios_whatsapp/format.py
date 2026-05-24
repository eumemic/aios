"""Minimal CommonMark → WhatsApp inline-syntax converter.

The model writes Markdown naturally; this transform maps the common
inline emphasis constructs to WhatsApp's own four-character grammar:

    **bold**   or  __bold__     →  *bold*
    *italic*   or  _italic_     →  _italic_
    ~~strike~~                  →  ~strike~
    `code`                      →  ```code```
    ```fence```                 →  ```fence```   (no change)

Block-level constructs (lists, headings, paragraphs, links, images)
fall through as text; WhatsApp has no rendering for them and a
naive transform would make the message uglier rather than prettier.
The model can be told via prompts.py to lean inline-only.

Operation order matters: code spans are protected before inline
emphasis runs (otherwise a backtick'd literal ``*foo*`` would be
italicized).  Bold runs before italic since both compete for the
asterisk character.
"""

from __future__ import annotations

import re

# Order-sensitive patterns.  Code fences and inline code first so
# their contents don't get touched by the emphasis regexes.

_FENCE_RE = re.compile(r"```(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
# Bold pattern: only the asterisk form (``**bold**``).  CommonMark
# also accepts ``__bold__`` but supporting that here would mis-render
# Python dunders like ``__init__`` as bold ``*init*`` — the outer
# boundary check can't tell ``__init__`` apart from a legitimate
# bold ``__phrase__``, since both have non-word chars on the outside
# and word chars inside.  Models reliably write ``**`` for bold; the
# ``__`` form is the rare-enough case we choose to surrender.
_BOLD_STAR_RE = re.compile(r"(?<![\w*])\*\*(?!\s)(.+?)(?<!\s)\*\*(?![\w*])", re.DOTALL)
# Italic patterns reject snake_case / surrounded-by-word_chars contexts
# so ``print_hello`` doesn't render as ``print<i>hello</i>``.
_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)

# Sentinels picked from C0 control range so they can't appear in
# normal message text.  Code spans and bold outputs are both stashed
# behind sentinels during emphasis processing so the italic regex
# can't chew them up: WhatsApp bold is ``*x*`` and CommonMark italic
# is ``*x*`` too, so the bold output collides with the italic
# pattern unless protected.
_SLOT_OPEN = "\x01SLOT\x01"
_SLOT_CLOSE = "\x01ENDSLOT\x01"


def markdown_to_whatsapp(text: str) -> str:
    """Render Markdown-with-CommonMark-emphasis to WhatsApp inline syntax."""
    placeholders: list[str] = []

    def stash(rendered: str) -> str:
        placeholders.append(rendered)
        return f"{_SLOT_OPEN}{len(placeholders) - 1}{_SLOT_CLOSE}"

    # Phase 1: hide code spans behind sentinels.  Fences keep their
    # WhatsApp ```...``` shape; inline backticks promote to
    # ``` ``` since WhatsApp has no single-backtick equivalent.
    text = _FENCE_RE.sub(lambda m: stash(f"```{m.group(1)}```"), text)
    text = _INLINE_CODE_RE.sub(lambda m: stash(f"```{m.group(1)}```"), text)

    # Phase 2: bold first, with the WhatsApp ``*x*`` output stashed so
    # the italic regex can't re-match it as italic.
    text = _BOLD_STAR_RE.sub(lambda m: stash(f"*{m.group(1)}*"), text)

    # Phase 3: italic (now safe from bold collision) and strike.
    text = _ITALIC_STAR_RE.sub(r"_\1_", text)
    text = _ITALIC_UNDER_RE.sub(r"_\1_", text)
    text = _STRIKE_RE.sub(r"~\1~", text)

    # Phase 4: restore stashed slots.  Bounds-check the index so an
    # adversarial input literally containing the sentinel pattern
    # (e.g. terminal-session paste with binary content) restores to
    # the literal match instead of raising IndexError.
    def restore(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(placeholders):
            return placeholders[idx]
        return m.group(0)

    return re.sub(rf"{_SLOT_OPEN}(\d+){_SLOT_CLOSE}", restore, text)
