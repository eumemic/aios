"""Markdown to Signal textStyles conversion.

Converts a subset of markdown formatting to Signal's textStyles format.
Signal textStyles are strings of the form "start:length:STYLE" where start
and length are UTF-16 code unit offsets (not Python code point offsets).

Supported styles:
- **text** or __text__ -> BOLD
- *text* or _text_ -> ITALIC
- ~~text~~ -> STRIKETHROUGH
- `text` -> MONOSPACE (inline code)
- ```...``` -> MONOSPACE (fenced code block)
- ||text|| -> SPOILER
- # Header -> BOLD (strips the # prefix)
"""

from __future__ import annotations

import re
from typing import NamedTuple


def _utf16_len(text: str) -> int:
    # Each UTF-16 code unit is 2 bytes; divide by 2 to get code-unit count.
    return len(text.encode("utf-16-le")) // 2


def _codepoint_to_utf16_offset(text: str, cp_offset: int) -> int:
    return _utf16_len(text[:cp_offset])


class _StyleSpan(NamedTuple):
    # Code point indices into the ORIGINAL text (before delimiter removal)
    content_start: int  # first char of content (after opening delimiter)
    content_end: int  # one past last char of content (before closing delimiter)
    style: str  # BOLD, ITALIC, STRIKETHROUGH, MONOSPACE, SPOILER
    # The delimiter ranges to remove: list of (start, length) in original text
    delimiter_ranges: tuple[tuple[int, int], ...]


def _overlaps_protected(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    """Return True if [start, end) overlaps a protected range without fully containing it.

    Allows outer formatting to wrap code (bold around backticks) but prevents
    inner formatting inside code (bold inside a backtick span).
    """
    return any(start < pe and end > ps and not (start <= ps and end >= pe) for ps, pe in protected)


# Compiled once at import; matchers are on the per-outbound-message hot path.
_FENCED_RE = re.compile(r"```([a-zA-Z]*)\n?(.*?)\n?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?$", re.MULTILINE)
_BOLD_STAR_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"__(?!\s)(.+?)(?<!\s)__", re.DOTALL)
# *text* — not ** on either side
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)(?!\s)(.+?)(?<!\s)(?<!\*)\*(?!\*)", re.DOTALL)
# _text_ — not preceded/followed by word chars (avoids snake_case)
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(?!_)(?!\s)(.+?)(?<!\s)(?<!_)_(?!\w)", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_SPOILER_RE = re.compile(r"\|\|(.+?)\|\|", re.DOTALL)


def convert_markdown_to_signal_styles(
    text: str,
) -> tuple[str, list[str]]:
    """Convert markdown formatting to Signal textStyles.

    Parses a subset of markdown and returns the stripped text along with
    Signal textStyles annotations. Offsets in the returned styles are
    UTF-16 code unit offsets (as required by signal-cli).

    Args:
        text: Input text possibly containing markdown formatting.

    Returns:
        A tuple of (stripped_text, styles) where:
        - stripped_text has delimiter characters removed
        - styles is a list of "start:length:STYLE" strings

    Notes:
        - Content inside code spans/blocks is not further parsed for markdown.
        - Unmatched or empty delimiters are left as-is.
        - Underscores inside words (e.g. snake_case) are not treated as italic.
    """
    if not text:
        return text, []

    spans: list[_StyleSpan] = []
    # Protected ranges: code spans/blocks where no further markdown parsing occurs
    # Each entry is (start_cp, end_cp) in original text (entire match including delimiters)
    protected: list[tuple[int, int]] = []

    # -------------------------------------------------------------------------
    # Phase 1: Find fenced code blocks first (highest priority, protect content)
    # -------------------------------------------------------------------------
    for m in _FENCED_RE.finditer(text):
        lang = m.group(1)
        content = m.group(2)
        # Calculate where the content starts in original text
        # Opening delimiter: ``` + optional lang tag + optional newline
        open_delim_end = m.start() + 3 + len(lang)
        # Skip the newline after lang tag if present
        if open_delim_end < len(text) and text[open_delim_end] == "\n":
            open_delim_end += 1
        content_start = open_delim_end
        content_end = content_start + len(content)

        # Opening delimiter range: m.start() to content_start
        # Closing delimiter range: content_end to m.end() (handles trailing \n + ```)
        # But we need to skip the trailing newline before ``` if present
        # We want to remove: content_end .. m.end() which is "\n```" or just "```"
        delim_ranges = (
            (m.start(), content_start - m.start()),  # opening: ``` + lang + \n
            (content_end, m.end() - content_end),  # closing: \n``` or ```
        )

        spans.append(
            _StyleSpan(
                content_start=content_start,
                content_end=content_end,
                style="MONOSPACE",
                delimiter_ranges=delim_ranges,
            )
        )
        # Protect the entire match
        protected.append((m.start(), m.end()))

    # -------------------------------------------------------------------------
    # Phase 2: Inline code (backtick spans) - protect content
    # -------------------------------------------------------------------------
    for m in _INLINE_CODE_RE.finditer(text):
        if _overlaps_protected(m.start(), m.end(), protected):
            continue
        content_start = m.start() + 1
        content_end = m.end() - 1
        delim_ranges = (
            (m.start(), 1),  # opening `
            (content_end, 1),  # closing `
        )
        spans.append(
            _StyleSpan(
                content_start=content_start,
                content_end=content_end,
                style="MONOSPACE",
                delimiter_ranges=delim_ranges,
            )
        )
        protected.append((m.start(), m.end()))

    # -------------------------------------------------------------------------
    # Phase 3: Headers (# text) -> BOLD (strip # prefix, render content bold)
    # -------------------------------------------------------------------------
    for m in _HEADER_RE.finditer(text):
        if _overlaps_protected(m.start(), m.end(), protected):
            continue
        content_start = m.start(2)  # start of header text
        content_end = m.end(2)  # end of header text
        if content_start >= content_end:
            continue
        # Remove everything before content: "## " prefix
        prefix_len = content_start - m.start()
        # Also remove trailing " ##" if present (closing ATX style)
        delim_ranges_list: list[tuple[int, int]] = [
            (m.start(), prefix_len),  # opening: "## "
        ]
        # If the full match extends beyond group(2), there's a trailing suffix
        if m.end() > content_end:
            delim_ranges_list.append(
                (content_end, m.end() - content_end),
            )
        spans.append(
            _StyleSpan(
                content_start=content_start,
                content_end=content_end,
                style="BOLD",
                delimiter_ranges=tuple(delim_ranges_list),
            )
        )
        protected.append((m.start(), m.end()))

    # -------------------------------------------------------------------------
    # Phase 4: Bold (**text** or __text__)
    # -------------------------------------------------------------------------
    for pat in (_BOLD_STAR_RE, _BOLD_UNDER_RE):
        for m in pat.finditer(text):
            if _overlaps_protected(m.start(), m.end(), protected):
                continue
            content_start = m.start() + 2
            content_end = m.end() - 2
            if content_start >= content_end:
                continue
            delim_ranges = (
                (m.start(), 2),  # opening ** or __
                (content_end, 2),  # closing ** or __
            )
            spans.append(
                _StyleSpan(
                    content_start=content_start,
                    content_end=content_end,
                    style="BOLD",
                    delimiter_ranges=delim_ranges,
                )
            )

    # -------------------------------------------------------------------------
    # Phase 5: Italic (*text* or _text_)
    # -------------------------------------------------------------------------
    for pat in (_ITALIC_STAR_RE, _ITALIC_UNDER_RE):
        for m in pat.finditer(text):
            if _overlaps_protected(m.start(), m.end(), protected):
                continue
            content_start = m.start() + 1
            content_end = m.end() - 1
            if content_start >= content_end:
                continue
            delim_ranges = (
                (m.start(), 1),  # opening * or _
                (content_end, 1),  # closing * or _
            )
            spans.append(
                _StyleSpan(
                    content_start=content_start,
                    content_end=content_end,
                    style="ITALIC",
                    delimiter_ranges=delim_ranges,
                )
            )

    # -------------------------------------------------------------------------
    # Phase 6: Strikethrough (~~text~~)
    # -------------------------------------------------------------------------
    for m in _STRIKE_RE.finditer(text):
        if _overlaps_protected(m.start(), m.end(), protected):
            continue
        content_start = m.start() + 2
        content_end = m.end() - 2
        if content_start >= content_end:
            continue
        delim_ranges = (
            (m.start(), 2),
            (content_end, 2),
        )
        spans.append(
            _StyleSpan(
                content_start=content_start,
                content_end=content_end,
                style="STRIKETHROUGH",
                delimiter_ranges=delim_ranges,
            )
        )

    # -------------------------------------------------------------------------
    # Phase 7: Spoiler (||text||)
    # -------------------------------------------------------------------------
    for m in _SPOILER_RE.finditer(text):
        if _overlaps_protected(m.start(), m.end(), protected):
            continue
        content_start = m.start() + 2
        content_end = m.end() - 2
        if content_start >= content_end:
            continue
        delim_ranges = (
            (m.start(), 2),
            (content_end, 2),
        )
        spans.append(
            _StyleSpan(
                content_start=content_start,
                content_end=content_end,
                style="SPOILER",
                delimiter_ranges=delim_ranges,
            )
        )

    if not spans:
        return text, []

    # -------------------------------------------------------------------------
    # Phase 8: Collect all delimiter ranges and build stripped text
    # -------------------------------------------------------------------------
    # Collect all (start, length) removal ranges, deduplicate, sort
    all_removals: list[tuple[int, int]] = []
    for span in spans:
        for dr in span.delimiter_ranges:
            if dr[1] > 0 and dr not in all_removals:
                all_removals.append(dr)

    # Sort by start position ascending (used by _adjusted_cp_offset below)
    all_removals_sorted_asc = sorted(all_removals, key=lambda x: x[0])

    # Build stripped text by removing delimiter ranges left-to-right
    result_chars = list(text)
    # Remove right-to-left to preserve indices
    for start, length in sorted(all_removals, key=lambda x: x[0], reverse=True):
        del result_chars[start : start + length]
    stripped = "".join(result_chars)

    # -------------------------------------------------------------------------
    # Phase 9: Compute Signal textStyles with UTF-16 offsets
    # -------------------------------------------------------------------------
    def _adjusted_cp_offset(original_pos: int) -> int:
        """Compute where original_pos lands in stripped text (code point offset)."""
        shift = 0
        for rem_start, rem_len in all_removals_sorted_asc:
            if rem_start < original_pos:
                # How much of this removal is before original_pos?
                overlap = min(rem_len, original_pos - rem_start)
                shift += overlap
        return original_pos - shift

    # Sort spans by their content_start in the original text
    spans_sorted = sorted(spans, key=lambda s: s.content_start)

    styles: list[str] = []
    for span in spans_sorted:
        adj_start = _adjusted_cp_offset(span.content_start)
        adj_end = _adjusted_cp_offset(span.content_end)
        length_cp = adj_end - adj_start
        if length_cp <= 0:
            continue
        # Convert to UTF-16 offsets
        utf16_start = _codepoint_to_utf16_offset(stripped, adj_start)
        utf16_length = _utf16_len(stripped[adj_start:adj_end])
        styles.append(f"{utf16_start}:{utf16_length}:{span.style}")

    return stripped, styles
