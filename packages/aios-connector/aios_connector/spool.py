"""SQLite-backed spool for inbound notifications.

The spool closes the worker-died-mid-pipe window: when the connector
emits an inbound notification, it persists to disk first, then sends
the notification over stdio.  If aios crashes before acking, the next
subprocess startup replays unacked entries before resuming live
delivery.  Combined with the worker-side dedup ledger
(``connector_inbound_acks``), this gives exactly-once event append.

Deliberate non-features:

* No size cap.  Plan decision #3 punts unbounded growth to the
  operator — multi-day aios outages can fill disk, log a warning if
  >10MB but never block emission.  See plan §"Open items #3".
* No per-row TTL.  Spool entries clear on ack; that's the only
  pruning path.

Schema lives at ``~/.aios/connectors/<name>/spool.sqlite`` (the cwd
the supervisor stamps on every connector subprocess).  Single
``inbound`` table:

    event_id    TEXT PRIMARY KEY  — ULID, also the dedup-ledger key
    payload     BLOB NOT NULL     — raw JSON-RPC params bytes
    created_at  REAL NOT NULL     — unix epoch seconds; sort key
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

# Soft warning threshold per plan §"Open items #3".  Spool growth past
# this is a sign of a multi-day aios outage; operators see it logged
# without anything blocking.
SOFT_WARN_BYTES = 10 * 1024 * 1024


class Spool:
    """Append-only persistence for unacked inbound notifications.

    Synchronous because SQLite is — running it on the asyncio loop
    would buffer rows in Python while waiting on a thread.  Each
    method opens a short-lived connection (``connect()`` + ``close()``);
    SQLite's WAL mode plus the per-call connection means concurrent
    readers (replay) and the writer (live emit) don't block.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        # Isolation level None means we manage transactions ourselves;
        # we never need read-vs-write isolation on this single-table
        # store, only durability of each commit.
        conn = sqlite3.connect(self._path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS inbound ("
                "  event_id   TEXT PRIMARY KEY, "
                "  payload    BLOB NOT NULL, "
                "  created_at REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS inbound_created_at_idx ON inbound (created_at)"
            )

    def add(self, event_id: str, payload: bytes, *, created_at: float) -> None:
        """Persist an inbound entry.  Raises on duplicate ``event_id``.

        The ULID generation lives in the caller (the base class) so the
        ID survives a crash between ``add()`` and the stdio write —
        replay re-uses the same ID and the worker-side dedup ledger
        treats both deliveries as the same event.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO inbound (event_id, payload, created_at) VALUES (?, ?, ?)",
                (event_id, payload, created_at),
            )

    def ack(self, event_id: str) -> bool:
        """Delete an entry by ID.  Returns ``True`` iff a row was removed.

        Idempotent: a worker that retries the ack after a successful
        delete (e.g. NOTIFY arrived twice) hits the no-op path and the
        ``False`` return is expected.
        """
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM inbound WHERE event_id = ?", (event_id,))
            return cur.rowcount > 0

    def unacked(self) -> list[tuple[str, bytes]]:
        """All entries oldest-first.  Used at startup before live emission."""
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "SELECT event_id, payload FROM inbound ORDER BY created_at, event_id"
            )
            return [(row[0], row[1]) for row in cur.fetchall()]

    def size_bytes(self) -> int:
        """On-disk size of the spool file (or 0 if missing).

        Surfaced so the base class can warn at the soft threshold
        without forcing the caller to stat the file themselves.
        """
        try:
            return self._path.stat().st_size
        except FileNotFoundError:
            return 0
