"""Atomic file write/delete primitives for memory-store mirroring.

Tool-driven and API-driven memory writes mirror to the shared host
directory after the durable DB write commits. With multiple containers
attached to the same source dir, a non-atomic write would let other
containers' readers observe partial bytes mid-mirror. Writing to a temp
file first and renaming gives POSIX-atomic visibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from aios.sandbox.volumes import ensure_owned_dir


def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via temp + ``os.replace``.

    Creates parent directories as needed. Existing files at ``path`` are
    replaced; concurrent readers either see the old or the new content,
    never a partial write.
    """
    ensure_owned_dir(path.parent)
    tmp = path.parent / f".tmp.{uuid4().hex}"
    try:
        tmp.write_text(content)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_delete(path: Path) -> None:
    """Remove ``path`` if it exists; symmetric counterpart to :func:`atomic_write`."""
    path.unlink(missing_ok=True)
