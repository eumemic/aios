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

Schema lives at ``~/.aios/instances/<instance_id>/connectors/<name>/spool.sqlite``
— the cwd the supervisor stamps on every connector subprocess (#238).
Per-instance cloistering keeps concurrent dev instances from clobbering
each other's spools.  Single ``inbound`` table:

    event_id    TEXT PRIMARY KEY  — ULID, also the dedup-ledger key
    payload     BLOB NOT NULL     — raw JSON-RPC params bytes
    created_at  REAL NOT NULL     — unix epoch seconds; sort key
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

# Soft warning threshold per plan §"Open items #3".  Spool growth past
# this is a sign of a multi-day aios outage; operators see it logged
# without anything blocking.
SOFT_WARN_BYTES = 10 * 1024 * 1024


class Spool:
    """Append-only persistence for unacked inbound notifications.

    Holds one long-lived :class:`sqlite3.Connection` for the lifetime
    of the connector subprocess (which is single-threaded — the SDK's
    asyncio loop drives all callers serially).  WAL + per-statement
    autocommit means each ``add`` / ``ack`` durably commits without
    paying connection-setup or PRAGMA-replay cost on every emit.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``isolation_level=None`` puts us in autocommit mode — every
        # ``execute`` is its own atomic write.  WAL keeps replay and
        # live writes from blocking each other across process restarts.
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS inbound ("
            "  event_id   TEXT PRIMARY KEY, "
            "  payload    BLOB NOT NULL, "
            "  created_at REAL NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS inbound_created_at_idx ON inbound (created_at)"
        )

    def add(self, event_id: str, payload: bytes, *, created_at: float) -> None:
        """Persist an inbound entry.  Raises on duplicate ``event_id``.

        The ULID generation lives in the caller (the base class) so the
        ID survives a crash between ``add()`` and the stdio write —
        replay re-uses the same ID and the worker-side dedup ledger
        treats both deliveries as the same event.
        """
        self._conn.execute(
            "INSERT INTO inbound (event_id, payload, created_at) VALUES (?, ?, ?)",
            (event_id, payload, created_at),
        )

    def ack(self, event_id: str) -> bool:
        """Delete an entry by ID.  Returns ``True`` iff a row was removed.

        Idempotent: a worker that retries the ack after a successful
        delete (e.g. NOTIFY arrived twice) hits the no-op path and the
        ``False`` return is expected.
        """
        cur = self._conn.execute("DELETE FROM inbound WHERE event_id = ?", (event_id,))
        return cur.rowcount > 0

    def unacked(self) -> list[tuple[str, bytes]]:
        """All entries oldest-first.  Used at startup before live emission."""
        cur = self._conn.execute(
            "SELECT event_id, payload FROM inbound ORDER BY created_at, event_id"
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def size_bytes(self) -> int:
        """Total on-disk size of the spool (main DB + WAL).

        WAL mode keeps recent writes in ``<path>-wal`` until a checkpoint
        merges them into the main DB; reporting only the main file would
        understate usage during an outage backlog.  The companion files
        (``-shm`` shared memory) are tiny and ignored.
        """
        total = 0
        for suffix in ("", "-wal"):
            with contextlib.suppress(FileNotFoundError):
                total += self._path.with_name(self._path.name + suffix).stat().st_size
        return total
