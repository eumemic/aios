"""SQLite-backed answered-tool_call_id spool.

A connector container that keeps memory-only state loses its set of
answered tool_call_ids on restart.  When the SSE stream reconnects it
backfills every pending call again — including ones we already
executed — and we'd double-execute (e.g. send a Telegram message
twice).

The spool persists the answered ids on disk, each mapped to the
serialized tool-result payload that was sent for it (or ``None`` when
no result is persisted, e.g. management calls).  On replay the
connector re-POSTs the persisted result instead of re-running the
side-effecting tool body.  Subclass :class:`HttpConnector` and wire it
in::

    class MyConnector(HttpConnector):
        def __init__(self) -> None:
            super().__init__()
            self._spool = SqliteAnsweredSpool("/var/lib/myconn/answered.sqlite")

        async def load_answered(self) -> dict[str, str | None]:
            return self._spool.load()

        async def save_answered(
            self, tool_call_id: str, result: str | None = None
        ) -> None:
            self._spool.add(tool_call_id, result)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteAnsweredSpool:
    """Tiny on-disk map of ``tool_call_id`` → serialized result."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS answered ("
            "  tool_call_id TEXT PRIMARY KEY,"
            "  result TEXT,"
            # ``unixepoch('subsec')`` (sub-second resolution) only exists on
            # SQLite >= 3.42; on the bookworm runtime (3.40) it yields NULL,
            # which would violate NOT NULL and make ``INSERT OR IGNORE``
            # silently drop every row.  COALESCE to integer-second
            # ``unixepoch()`` keeps NOT NULL satisfied everywhere.
            "  added_at REAL NOT NULL "
            "DEFAULT (COALESCE(unixepoch('subsec'), unixepoch()))"
            ")"
        )
        self._conn.commit()

    def load(self) -> dict[str, str | None]:
        rows = self._conn.execute("SELECT tool_call_id, result FROM answered").fetchall()
        return {row[0]: row[1] for row in rows}

    def add(self, tool_call_id: str, result: str | None = None) -> None:
        # INSERT OR IGNORE keeps the first persisted result for an id: a
        # replay re-POST never overwrites the original send result.
        self._conn.execute(
            "INSERT OR IGNORE INTO answered (tool_call_id, result) VALUES (?, ?)",
            (tool_call_id, result),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
