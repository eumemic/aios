"""Shared validators for path-shaped strings.

Two distinct concepts live here:

* :func:`validate_path_segments` — channel-path strings (inbound-message
  ``path``, routing-rule ``prefix``). May be empty for the per-connection
  catch-all; rejects ``..`` segments.
* :data:`ABSOLUTE_PATH_PATTERN` + :func:`check_no_traversal_segments` —
  absolute mount/memory paths (``MemoryPath``, ``GithubRepositoryResource.
  mount_path``). Must match the pattern AND have no ``.``/``..`` segments
  so ``host_dir / path.lstrip("/")`` can't escape the per-resource dir.
"""

from __future__ import annotations

# Absolute, slash-separated path with non-empty non-NUL segments.
# Used as the Pydantic ``Field(pattern=...)`` for mount-like paths.
ABSOLUTE_PATH_PATTERN = r"^(/[^/\x00]+)+$"


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


def check_no_traversal_segments(path: str) -> None:
    """Reject ``.`` and ``..`` segments in an absolute path.

    Path-traversal guard at the input boundary — without this,
    ``host_dir / path.lstrip("/")`` could escape the per-resource dir.
    """
    for segment in path.lstrip("/").split("/"):
        if segment in (".", ".."):
            raise ValueError(f"path segment {segment!r} is not allowed (would enable traversal)")
