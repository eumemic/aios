"""Fuzzy matching module for file operations.

Vendored from hermes-agent tools/fuzzy_match.py at commit 34d06a9. The only
substantive change from upstream is style modernization for aios's Python
3.13 + mypy strict house conventions:

- ``from __future__ import annotations`` at the top.
- ``Tuple[...]``, ``Optional[...]``, ``List[...]``, ``Callable`` typing
  imports replaced with builtin generics (``tuple``, ``list``, etc.) and
  ``| None``.
- All function annotations made mypy-strict compatible (no bare ``list``
  without a type parameter).

Implements a multi-strategy matching chain to robustly find and replace
text, accommodating variations in whitespace, indentation, and escaping
common in LLM-generated code.

The 8-strategy chain (inspired by OpenCode), tried in order:

1. Exact match — direct string comparison
2. Line-trimmed — strip leading/trailing whitespace per line
3. Whitespace normalized — collapse multiple spaces/tabs to single space
4. Indentation flexible — ignore indentation differences entirely
5. Escape normalized — convert ``\\n`` literals to actual newlines
6. Trimmed boundary — trim first/last line whitespace only
7. Block anchor — match first+last lines, use similarity for middle
8. Context-aware — 50% line similarity threshold

Multi-occurrence matching is handled via the ``replace_all`` flag.

Usage::

    from aios.vendor.hermes_files.fuzzy_match import fuzzy_find_and_replace

    new_content, match_count, error = fuzzy_find_and_replace(
        content="def foo():\\n    pass",
        old_string="def foo():",
        new_string="def bar():",
        replace_all=False,
    )
"""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher

UNICODE_MAP = {
    "\u201c": '"',
    "\u201d": '"',  # smart double quotes
    "\u2018": "'",
    "\u2019": "'",  # smart single quotes
    "\u2014": "--",
    "\u2013": "-",  # em/en dashes
    "\u2026": "...",
    "\u00a0": " ",  # ellipsis and non-breaking space
}


def _unicode_normalize(text: str) -> str:
    """Normalize Unicode characters to their standard ASCII equivalents."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


def fuzzy_find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, int, str | None]:
    """Find and replace text using a chain of increasingly fuzzy strategies.

    Args:
        content: The file content to search in.
        old_string: The text to find.
        new_string: The replacement text.
        replace_all: If True, replace all occurrences; if False, require uniqueness.

    Returns:
        ``(new_content, match_count, error_message)``. On success,
        ``(modified_content, number_of_replacements, None)``. On failure,
        ``(original_content, 0, error_description)``.
    """
    if not old_string:
        return content, 0, "old_string cannot be empty"

    if old_string == new_string:
        return content, 0, "old_string and new_string are identical"

    strategies: list[tuple[str, Callable[[str, str], list[tuple[int, int]]]]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for _strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, old_string)

        if matches:
            if len(matches) > 1 and not replace_all:
                return (
                    content,
                    0,
                    (
                        f"Found {len(matches)} matches for old_string. "
                        "Provide more context to make it unique, or use replace_all=True."
                    ),
                )

            new_content = _apply_replacements(content, matches, new_string)
            return new_content, len(matches), None

    return content, 0, "Could not find a match for old_string in the file"


def _apply_replacements(content: str, matches: list[tuple[int, int]], new_string: str) -> str:
    """Apply replacements at the given positions.

    Sorts matches by position (descending) to replace from end to start,
    which preserves positions of earlier matches.
    """
    sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)

    result = content
    for start, end in sorted_matches:
        result = result[:start] + new_string + result[end:]

    return result


# ─── Matching strategies ──────────────────────────────────────────────────────


def _strategy_exact(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 1: exact string match."""
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        start = pos + 1
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 2: match with line-by-line whitespace trimming."""
    pattern_lines = [line.strip() for line in pattern.split("\n")]
    pattern_normalized = "\n".join(pattern_lines)

    content_lines = content.split("\n")
    content_normalized_lines = [line.strip() for line in content_lines]

    return _find_normalized_matches(
        content,
        content_lines,
        content_normalized_lines,
        pattern,
        pattern_normalized,
    )


def _strategy_whitespace_normalized(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 3: collapse multiple whitespace to single space."""

    def normalize(s: str) -> str:
        return re.sub(r"[ \t]+", " ", s)

    pattern_normalized = normalize(pattern)
    content_normalized = normalize(content)

    matches_in_normalized = _strategy_exact(content_normalized, pattern_normalized)

    if not matches_in_normalized:
        return []

    return _map_normalized_positions(content, content_normalized, matches_in_normalized)


def _strategy_indentation_flexible(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 4: ignore indentation differences entirely."""
    content_lines = content.split("\n")
    content_stripped_lines = [line.lstrip() for line in content_lines]
    pattern_lines = [line.lstrip() for line in pattern.split("\n")]

    return _find_normalized_matches(
        content,
        content_lines,
        content_stripped_lines,
        pattern,
        "\n".join(pattern_lines),
    )


def _strategy_escape_normalized(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 5: convert escape sequences (``\\n`` → newline) to actual characters."""

    def unescape(s: str) -> str:
        return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    pattern_unescaped = unescape(pattern)

    if pattern_unescaped == pattern:
        return []

    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 6: trim whitespace from first and last lines only."""
    pattern_lines = pattern.split("\n")
    if not pattern_lines:
        return []

    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()

    modified_pattern = "\n".join(pattern_lines)

    content_lines = content.split("\n")

    matches: list[tuple[int, int]] = []
    pattern_line_count = len(pattern_lines)

    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i : i + pattern_line_count]

        check_lines = block_lines.copy()
        check_lines[0] = check_lines[0].strip()
        if len(check_lines) > 1:
            check_lines[-1] = check_lines[-1].strip()

        if "\n".join(check_lines) == modified_pattern:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


def _strategy_block_anchor(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 7: match by anchoring on first and last lines.

    Uses permissive similarity thresholds and unicode normalization on the
    middle section.
    """
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)

    pattern_lines = norm_pattern.split("\n")
    if len(pattern_lines) < 2:
        return []

    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()

    norm_content_lines = norm_content.split("\n")
    orig_content_lines = content.split("\n")

    pattern_line_count = len(pattern_lines)

    potential_matches: list[int] = []
    for i in range(len(norm_content_lines) - pattern_line_count + 1):
        if (
            norm_content_lines[i].strip() == first_line
            and norm_content_lines[i + pattern_line_count - 1].strip() == last_line
        ):
            potential_matches.append(i)

    matches: list[tuple[int, int]] = []
    candidate_count = len(potential_matches)

    # 0.10 for unique matches (max flexibility), 0.30 for multiple candidates.
    threshold = 0.10 if candidate_count == 1 else 0.30

    for i in potential_matches:
        if pattern_line_count <= 2:
            similarity = 1.0
        else:
            content_middle = "\n".join(norm_content_lines[i + 1 : i + pattern_line_count - 1])
            pattern_middle = "\n".join(pattern_lines[1:-1])
            similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()

        if similarity >= threshold:
            start_pos, end_pos = _calculate_line_positions(
                orig_content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


def _strategy_context_aware(content: str, pattern: str) -> list[tuple[int, int]]:
    """Strategy 8: line-by-line similarity with 50% threshold.

    Finds blocks where at least 50% of lines have high similarity.
    """
    pattern_lines = pattern.split("\n")
    content_lines = content.split("\n")

    if not pattern_lines:
        return []

    matches: list[tuple[int, int]] = []
    pattern_line_count = len(pattern_lines)

    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i : i + pattern_line_count]

        high_similarity_count = 0
        for p_line, c_line in zip(pattern_lines, block_lines, strict=False):
            sim = SequenceMatcher(None, p_line.strip(), c_line.strip()).ratio()
            if sim >= 0.80:
                high_similarity_count += 1

        if high_similarity_count >= len(pattern_lines) * 0.5:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _calculate_line_positions(
    content_lines: list[str],
    start_line: int,
    end_line: int,
    content_length: int,
) -> tuple[int, int]:
    """Calculate start and end character positions from line indices.

    Args:
        content_lines: List of lines (without newlines).
        start_line: Starting line index (0-based).
        end_line: Ending line index (exclusive, 0-based).
        content_length: Total length of the original content string.

    Returns:
        ``(start_pos, end_pos)`` in the original content.
    """
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    if end_pos >= content_length:
        end_pos = content_length
    return start_pos, end_pos


def _find_normalized_matches(
    content: str,
    content_lines: list[str],
    content_normalized_lines: list[str],
    pattern: str,
    pattern_normalized: str,
) -> list[tuple[int, int]]:
    """Find matches in normalized content and map back to original positions."""
    pattern_norm_lines = pattern_normalized.split("\n")
    num_pattern_lines = len(pattern_norm_lines)

    matches: list[tuple[int, int]] = []

    for i in range(len(content_normalized_lines) - num_pattern_lines + 1):
        block = "\n".join(content_normalized_lines[i : i + num_pattern_lines])

        if block == pattern_normalized:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + num_pattern_lines, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


def _map_normalized_positions(
    original: str,
    normalized: str,
    normalized_matches: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map positions from normalized string back to original (best effort)."""
    if not normalized_matches:
        return []

    orig_to_norm: list[int] = []  # orig_to_norm[i] = position in normalized

    orig_idx = 0
    norm_idx = 0

    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in " \t" and normalized[norm_idx] == " ":
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            if orig_idx < len(original) and original[orig_idx] not in " \t":
                norm_idx += 1
        elif original[orig_idx] in " \t":
            orig_to_norm.append(norm_idx)
            orig_idx += 1
        else:
            orig_to_norm.append(norm_idx)
            orig_idx += 1

    while orig_idx < len(original):
        orig_to_norm.append(len(normalized))
        orig_idx += 1

    norm_to_orig_start: dict[int, int] = {}
    norm_to_orig_end: dict[int, int] = {}

    for orig_pos, norm_pos in enumerate(orig_to_norm):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos
        norm_to_orig_end[norm_pos] = orig_pos

    original_matches: list[tuple[int, int]] = []
    for norm_start, norm_end in normalized_matches:
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)

        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)

        while orig_end < len(original) and original[orig_end] in " \t":
            orig_end += 1

        original_matches.append((orig_start, min(orig_end, len(original))))

    return original_matches
