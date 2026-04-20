"""Shared validator for channel-path-like strings.

Used by both inbound-message ``path`` (must be non-empty) and routing-rule
``prefix`` (may be empty — the per-connection catch-all).  A single
source of truth keeps the two callers from drifting on what
"well-formed" means.

Well-formed: ``""`` (iff ``allow_empty=True``) or one-or-more ``/``-joined
non-empty segments, none of which equal ``".."``.  Rejects leading/trailing
slashes and doubled slashes implicitly (they produce empty segments when
split).  A single ``.`` segment is permitted — only the literal ``".."``
parent-reference is rejected.
"""

from __future__ import annotations


def validate_path_segments(s: str, *, allow_empty: bool) -> None:
    """Raise ``ValueError`` if ``s`` is not a well-formed path.

    ``allow_empty=True`` accepts ``""`` as a valid catch-all.
    """
    if s == "":
        if allow_empty:
            return
        raise ValueError("must be non-empty")
    for seg in s.split("/"):
        if seg == "":
            raise ValueError("must not contain empty segments (leading/trailing/doubled '/')")
        if seg == "..":
            raise ValueError("must not contain '..' segments")
