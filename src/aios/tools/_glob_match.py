"""Path-glob matcher for ``http_request`` route allowlists.

Pattern syntax:

* ``*`` matches a single path segment (no slashes).
* ``**`` matches any number of segments (including zero).
* Any other character is a literal.

Leading and trailing slashes on both pattern and path are stripped
before comparison, so ``/devices/*`` and ``devices/*`` are equivalent.
"""

from __future__ import annotations


def match_glob(pattern: str, path: str) -> bool:
    """Return True if ``path`` matches the segment-glob ``pattern``."""
    p = pattern.strip("/").split("/")
    s = path.strip("/").split("/")
    if p == [""]:
        p = []
    if s == [""]:
        s = []
    return _match(p, s)


def _match(p: list[str], s: list[str]) -> bool:
    if not p:
        return not s
    head, rest = p[0], p[1:]
    if head == "**":
        if not rest:
            return True
        return any(_match(rest, s[i:]) for i in range(len(s) + 1))
    if not s:
        return False
    if head == "*" or head == s[0]:
        return _match(rest, s[1:])
    return False
