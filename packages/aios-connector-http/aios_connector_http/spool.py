"""SQLite-backed answered-tool_call_id spool.

A connector container that keeps memory-only state loses its set of
answered tool_call_ids on restart.  When the SSE stream reconnects it
backfills every pending call again — including ones we already
executed — and we'd double-execute (e.g. send a Telegram message
twice).

The spool persists the answered ids on disk.  Subclass
:class:`HttpConnector` and wire it in::

    class MyConnector(HttpConnector):
        def __init__(self) -> None:
            super().__init__()
            self._spool = SqliteAnsweredSpool("/var/lib/myconn/answered.sqlite")

        async def load_answered(self) -> set[str]:
            return self._spool.load()

        async def save_answered(self, tool_call_id: str) -> None:
            self._spool.add(tool_call_id)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteAnsweredSpool:
    """Tiny on-disk set of ``tool_call_id`` strings."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS answered ("
            "  tool_call_id TEXT PRIMARY KEY,"
            "  added_at REAL NOT NULL DEFAULT (unixepoch('subsec'))"
            ")"
        )
        self._conn.commit()

    def load(self) -> set[str]:
        rows = self._conn.execute("SELECT tool_call_id FROM answered").fetchall()
        return {row[0] for row in rows}

    def add(self, tool_call_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO answered (tool_call_id) VALUES (?)", (tool_call_id,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
