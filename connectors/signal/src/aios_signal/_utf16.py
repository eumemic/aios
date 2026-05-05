"""UTF-16 offset helpers shared by markdown and mention encoding.

Signal's textStyles AND mentions both use UTF-16 code-unit offsets — a
JVM-era choice carried over from signal-cli's Java roots — and Python
strings index by code points.  These two helpers bridge the gap.
"""

from __future__ import annotations


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def codepoint_to_utf16_offset(text: str, cp_offset: int) -> int:
    return utf16_len(text[:cp_offset])
