"""Payload loaders for ``--file``, ``--stdin``, and ``--data`` flags.

Every create/update command accepts one of:

* ``--file PATH``   — read JSON from a file on disk
* ``--stdin``       — read JSON from stdin
* ``--data JSON``   — inline JSON string

These functions parse whichever was provided and return a dict. They also
walk a directory to synthesize a Skills ``files`` dict (path→contents).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class PayloadError(Exception):
    """Raised when payload parsing fails. Callers convert to CLI errors."""


def load_payload(
    file: Path | None,
    stdin: bool,
    data: str | None,
) -> dict[str, Any]:
    """Load a JSON object payload from exactly one source.

    Exactly one of ``file``, ``stdin``, ``data`` must be provided.
    Raises :class:`PayloadError` otherwise.
    """
    sources = [s for s in (file is not None, stdin, data is not None) if s]
    if len(sources) == 0:
        raise PayloadError("no payload provided; pass --file PATH, --stdin, or --data JSON")
    if len(sources) > 1:
        raise PayloadError("only one of --file, --stdin, or --data may be given")

    if file is not None:
        try:
            raw = file.read_text()
        except OSError as exc:
            raise PayloadError(f"could not read {file}: {exc}") from exc
        return _decode(raw, source=str(file))
    if stdin:
        raw = sys.stdin.read()
        return _decode(raw, source="<stdin>")
    assert data is not None
    return _decode(data, source="<--data>")


def _decode(raw: str, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise PayloadError(f"payload in {source} must be a JSON object, not {type(value).__name__}")
    return value


def walk_skill_dir(root: Path) -> dict[str, str]:
    """Build the ``files`` dict a skills endpoint expects from a directory.

    Keys are POSIX-style paths relative to ``root``. Values are file contents
    (text, UTF-8). A ``SKILL.md`` must exist directly under ``root``; the
    server validates this but we check early for a friendlier error.
    """
    if not root.is_dir():
        raise PayloadError(f"not a directory: {root}")
    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        raise PayloadError(f"missing SKILL.md at {skill_md}")

    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PayloadError(f"non-UTF-8 file in skill: {path}: {exc}") from exc
    return files
