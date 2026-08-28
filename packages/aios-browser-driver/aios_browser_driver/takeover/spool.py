"""The input-spool tailer.

Tails ``input/spool.jsonl`` — the append-only JSONL the API writes the human's
input to. Two hazards drive the design (red-team fold #13):

* The reaper UNLINKS and recreates the spool, so the tailer is inode-aware:
  it adopts a new file by ``open``-then-``fstat`` (never ``stat``-then-``open``,
  which could adopt a different inode than it checked), and drains the old fd
  to EOF before switching so no line is lost across the swap.
* A partial line (the API's append and our read interleave) is buffered until
  its newline arrives; a line that will not parse is skipped and logged, never
  fatal to the tailer.

The tailer returns parsed batch dicts; the caller (the controller) owns the
grant/epoch/seq filtering, since only it knows the standing takeover.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("aios_browser_driver.takeover.spool")

_READ_CHUNK = 65536


class SpoolTailer:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self._ino: int | None = None
        self._buf = b""

    def arm(self) -> None:
        """Open the spool at its CURRENT end, so input written before the
        takeover opened is never replayed. Absent spool → armed empty; the
        first :meth:`poll` adopts it when it appears."""
        try:
            fd = os.open(self._path, os.O_RDONLY)
        except FileNotFoundError:
            return
        os.lseek(fd, 0, os.SEEK_END)
        self._fd, self._ino = fd, os.fstat(fd).st_ino

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd, self._ino = None, None
        self._buf = b""

    def poll(self) -> list[dict[str, Any]]:
        """Return every complete batch appended since the last poll, following
        an inode rotation if the reaper recreated the spool."""
        lines: list[bytes] = []
        if self._fd is not None:
            lines += self._drain(self._fd)
        lines += self._follow_rotation()
        return [b for b in (self._parse(raw) for raw in lines) if b is not None]

    def _follow_rotation(self) -> list[bytes]:
        try:
            new_fd = os.open(self._path, os.O_RDONLY)
        except FileNotFoundError:
            return []  # unlinked and not yet recreated — retry next poll
        new_ino = os.fstat(new_fd).st_ino
        if self._ino == new_ino:
            os.close(new_fd)  # same file, already drained above
            return []
        # A fresh inode: the old fd was drained to EOF above; adopt the new
        # file from its start (a recreated spool begins empty). Drop any
        # buffered partial — its completion lived on the old inode, now gone.
        if self._fd is not None:
            os.close(self._fd)
        self._fd, self._ino, self._buf = new_fd, new_ino, b""
        return self._drain(new_fd)

    def _drain(self, fd: int) -> list[bytes]:
        while True:
            chunk = os.read(fd, _READ_CHUNK)
            if not chunk:
                break
            self._buf += chunk
        if b"\n" not in self._buf:
            return []
        *complete, self._buf = self._buf.split(b"\n")
        return complete

    def _parse(self, raw: bytes) -> dict[str, Any] | None:
        if not raw.strip():
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            log.warning("dropping malformed spool line (%d bytes)", len(raw))
            return None
        if not isinstance(doc, dict):
            return None
        return doc
