"""Shared validator for channel-path-like strings.

Used by both inbound-message ``path`` (must be non-empty) and routing-rule
``prefix`` (may be empty — the per-connection catch-all).
"""

from __future__ import annotations


def validate_path_segments(s: str, *, allow_empty: bool) -> None:
    """Raise ``ValueError`` if ``s`` is not a well-formed path."""
    if s == "":
        if allow_empty:
            return
        raise ValueError("must be non-empty")
    for seg in s.split("/"):
        if seg == "":
            raise ValueError("must not contain empty segments (leading/trailing/doubled '/')")
        if seg == "..":
            raise ValueError("must not contain '..' segments")
