"""Render a small Markdown subset to Slack ``mrkdwn`` + the hard clamps.

The model writes ordinary Markdown (``**bold**``, ``[label](url)``,
fenced code, …); Slack's message-formatting flavor — ``mrkdwn`` — uses a
*different* inline grammar:

    *bold*            (single asterisk, not ``**``)
    _italic_
    ~strike~
    `code`            (inline; same as Markdown)
    ```code```        (fenced; same as Markdown)
    > quote           (line prefix; same as Markdown)
    <url|label>       (link; angle-bracketed, not ``[label](url)``)

See https://api.slack.com/reference/surfaces/formatting#basics.  This
module maps the common Markdown constructs the model emits onto that
grammar with :func:`markdown_to_mrkdwn`, and applies the design's hard
length clamps (§3.5, §6 acceptance line) with :func:`clamp_message` /
:func:`clamp_section` / :func:`clamp_label` / :func:`clamp_blocks` —
**every Web API call runs its text through the relevant clamp first** so
a runaway model reply can never trip a Slack ``msg_too_long`` /
``invalid_blocks`` rejection (which would surface to the model as an
opaque failed tool call instead of a truncated-but-delivered message).

The clamp ceilings are Slack's documented limits, recorded as named
constants so the call sites read intent rather than magic numbers:

* :data:`MESSAGE_MAX_CHARS` — ``chat.postMessage`` ``text`` hard cap.
* :data:`SECTION_TEXT_MAX_CHARS` — a Block Kit ``section`` block's
  ``text`` cap (Block Kit is deferred to a later slice, but the clamp
  ships now so the IR pipeline is complete and the helper is unit-tested
  against the real ceiling).
* :data:`LABEL_MAX_CHARS` — a Block Kit interactive-element label cap.
* :data:`MAX_BLOCKS` — the per-message Block Kit block-count cap.

Truncation appends a single-character ellipsis (``…``) and counts it
against the budget so the *result* never exceeds the ceiling.
"""

from __future__ import annotations

import re

# ── Slack's documented hard limits (the clamp ceilings) ───────────────

#: ``chat.postMessage`` rejects a ``text`` longer than 40k bytes, but the
#: design pins the message clamp at 8000 chars (§3.5) — comfortably under
#: the wire limit and a sane single-message size.
MESSAGE_MAX_CHARS = 8000
#: A Block Kit ``section`` block's ``text`` field caps at 3000 chars.
SECTION_TEXT_MAX_CHARS = 3000
#: A Block Kit interactive-element / option label caps at 75 chars.
LABEL_MAX_CHARS = 75
#: A single message carries at most 50 Block Kit blocks.
MAX_BLOCKS = 50

_ELLIPSIS = "…"


# ── Markdown → Slack mrkdwn inline grammar ────────────────────────────

# Capture order matters: code spans are stashed first so their contents
# are immune to the emphasis/link rewrites; links run before emphasis so a
# ``[text](_url_)`` url isn't mangled by the italic rule.

_FENCE_RE = re.compile(r"```(?:\w*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
# ``[label](url)`` → ``<url|label>``.  The url is the first non-space run
# inside the parens (Slack links don't carry a title).
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
# ``**bold**`` / ``__bold__`` → ``*bold*`` (Slack's single-asterisk bold).
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"__(.+?)__", re.DOTALL)
# ``~~strike~~`` → ``~strike~`` (Slack's single-tilde strike).
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Single-asterisk italic in Markdown maps to Slack's ``_italic_``.  Reject
# leading/trailing whitespace and a neighbouring ``*`` so a ``**bold**``
# already rewritten above (or a bare ``*`` bullet) is left alone.
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)(?!\s)(.+?)(?<!\s)(?<!\*)\*(?!\*)", re.DOTALL)
# Single-underscore italic stays ``_italic_`` but we still normalize it so
# snake_case words aren't treated as emphasis (word-boundary guards).
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(?!_)(?!\s)(.+?)(?<!\s)(?<!_)_(?!\w)", re.DOTALL)


def markdown_to_mrkdwn(text: str) -> str:
    """Convert the common Markdown subset the model emits to Slack ``mrkdwn``.

    Pipeline:

    1. Stash fenced + inline code spans behind opaque placeholders so
       their contents survive the inline rewrites verbatim (a ``**`` or
       ``[..](..)`` inside a code span must NOT be reinterpreted).
    2. Rewrite links ``[label](url)`` → ``<url|label>``.
    3. Rewrite bold (``**x**`` / ``__x__`` → ``*x*``) and strike
       (``~~x~~`` → ``~x~``) to Slack's single-delimiter forms.
    4. Rewrite single-asterisk italic ``*x*`` → ``_x_``; single-underscore
       italic is already Slack-native (kept, with word-boundary guards).
    5. Restore the code placeholders.

    Slack ``mrkdwn`` does not HTML-escape; ``<``/``>``/``&`` render
    literally except where they form a ``<…>`` link/mention token, so —
    unlike the Telegram HTML converter — there is no escape pass.
    """
    placeholders: dict[str, str] = {}
    counter = 0

    def stash(rendered: str) -> str:
        nonlocal counter
        key = f"\x00CODE{counter}\x00"
        placeholders[key] = rendered
        counter += 1
        return key

    def fence_repl(m: re.Match[str]) -> str:
        return stash(f"```{m.group(1)}```")

    def inline_code_repl(m: re.Match[str]) -> str:
        return stash(f"`{m.group(1)}`")

    text = _FENCE_RE.sub(fence_repl, text)
    text = _INLINE_CODE_RE.sub(inline_code_repl, text)

    # Stash each converted span behind a placeholder so a *later* rule
    # can't chew it up — notably the single-asterisk italic rule would
    # otherwise re-match the ``*bold*`` the bold rule just produced and
    # downgrade it to ``_bold_``.
    text = _LINK_RE.sub(lambda m: stash(f"<{m.group(2)}|{m.group(1)}>"), text)
    text = _BOLD_STAR_RE.sub(lambda m: stash(f"*{m.group(1)}*"), text)
    text = _BOLD_UNDER_RE.sub(lambda m: stash(f"*{m.group(1)}*"), text)
    text = _STRIKE_RE.sub(lambda m: stash(f"~{m.group(1)}~"), text)
    text = _ITALIC_STAR_RE.sub(lambda m: stash(f"_{m.group(1)}_"), text)
    text = _ITALIC_UNDER_RE.sub(lambda m: stash(f"_{m.group(1)}_"), text)

    for key, rendered in placeholders.items():
        text = text.replace(key, rendered)
    return text


# ── the hard clamps (applied before every Web API call) ───────────────


def _clamp(text: str, limit: int) -> str:
    """Truncate ``text`` to at most ``limit`` chars, ellipsis included.

    A no-op when already within budget.  When over, keep ``limit - 1``
    chars and append a single ``…`` so the result length is exactly
    ``limit`` — never over.  ``limit <= 0`` yields the empty string.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + _ELLIPSIS


def clamp_message(text: str) -> str:
    """Clamp a ``chat.postMessage`` body to :data:`MESSAGE_MAX_CHARS`."""
    return _clamp(text, MESSAGE_MAX_CHARS)


def clamp_section(text: str) -> str:
    """Clamp a Block Kit ``section`` text to :data:`SECTION_TEXT_MAX_CHARS`."""
    return _clamp(text, SECTION_TEXT_MAX_CHARS)


def clamp_label(text: str) -> str:
    """Clamp a Block Kit element label to :data:`LABEL_MAX_CHARS`."""
    return _clamp(text, LABEL_MAX_CHARS)


def clamp_blocks(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Clamp a Block Kit block list to at most :data:`MAX_BLOCKS` blocks.

    Returns the input unchanged when within budget, else the first
    :data:`MAX_BLOCKS` blocks.  Block Kit is deferred (§7), but the clamp
    ships with the IR pipeline so the helper is complete and tested.
    """
    if len(blocks) <= MAX_BLOCKS:
        return blocks
    return blocks[:MAX_BLOCKS]


# ── emoji normalization (slack_react) ─────────────────────────────────

_EMOJI_STRIP_RE = re.compile(r"[^a-z0-9_+-]")


def normalize_emoji(emoji: str) -> str:
    """Normalize a model-supplied emoji name to Slack's ``reactions.add`` form.

    Slack's ``reactions.add`` ``name`` is the bare shortcode — ``thumbsup``,
    not ``:thumbsup:``.  The model may pass either form (or with stray
    whitespace / uppercase), so we strip surrounding colons, lowercase,
    trim, and drop any character outside Slack's shortcode alphabet
    (``a-z``, ``0-9``, ``_``, ``+``, ``-`` — the last two for names like
    ``+1`` and ``e-mail``).  A skin-tone suffix (``::skin-tone-2``) is
    preserved as ``_skin-tone-2`` only incidentally; v0 does not special-case
    it.  Returns the cleaned shortcode (possibly empty if the input held no
    valid characters — the caller decides what to do with that).
    """
    name = emoji.strip().strip(":").strip().lower()
    return _EMOJI_STRIP_RE.sub("", name)
