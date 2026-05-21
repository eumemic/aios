"""Plain-text composition helpers for the system-prompt builders."""

from __future__ import annotations


def join_blocks(*blocks: str) -> str:
    """Join non-empty blocks with ``\\n\\n``; drop empty/falsy ones.

    Used by the system-prompt augmenters (focal-paradigm, memory-stores,
    skills, instructions) that compose multiple optional sections
    without producing a leading or trailing ``\\n\\n`` when one half
    is empty.
    """
    return "\n\n".join(block for block in blocks if block)
