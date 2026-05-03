"""Spool round-trip tests.

Spool is the durability piece: ``add()`` then crash, then ``unacked()``
returns the entries.  These tests don't go through the SDK base class —
they verify the storage primitive in isolation so a regression in
SQLite-handling shows up clearly without anyio noise.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from aios_connector.spool import Spool


def test_round_trip(tmp_path: Path) -> None:
    spool = Spool(tmp_path / "spool.sqlite")
    spool.add("01ULID0001", b"first", created_at=time.time())
    spool.add("01ULID0002", b"second", created_at=time.time() + 0.001)

    pending = spool.unacked()
    assert [event_id for event_id, _ in pending] == ["01ULID0001", "01ULID0002"]
    assert pending[0][1] == b"first"

    assert spool.ack("01ULID0001") is True
    assert [event_id for event_id, _ in spool.unacked()] == ["01ULID0002"]


def test_ack_unknown_returns_false(tmp_path: Path) -> None:
    """Idempotent ack: a second call after delete returns False, not an error.

    The worker calls ack fire-and-forget after commit; if the NOTIFY
    arrives twice (rare but possible if the connector replays before
    the worker's first ack hits), both calls should succeed cleanly.
    """
    spool = Spool(tmp_path / "spool.sqlite")
    spool.add("01ULID0001", b"x", created_at=time.time())
    assert spool.ack("01ULID0001") is True
    assert spool.ack("01ULID0001") is False


def test_duplicate_event_id_raises(tmp_path: Path) -> None:
    """ULIDs are unique by construction — a duplicate is a bug to surface.

    The spool's PK enforces this at the storage layer so a callsite
    handing in the same ULID twice (e.g. a botched retry path) fails
    loudly instead of corrupting the ordering.
    """
    import sqlite3

    spool = Spool(tmp_path / "spool.sqlite")
    spool.add("01ULID0001", b"first", created_at=time.time())
    with pytest.raises(sqlite3.IntegrityError):
        spool.add("01ULID0001", b"duplicate", created_at=time.time())


def test_persists_across_instances(tmp_path: Path) -> None:
    """Reopening the spool returns the same unacked entries.

    Mirrors the connector restart path: the subprocess is killed,
    a new one opens the same SQLite file, ``unacked()`` returns the
    pending replay set.
    """
    path = tmp_path / "spool.sqlite"
    a = Spool(path)
    a.add("01ULID0001", b"data", created_at=time.time())
    del a

    b = Spool(path)
    assert [event_id for event_id, _ in b.unacked()] == ["01ULID0001"]


def test_size_bytes_grows(tmp_path: Path) -> None:
    """``size_bytes`` reflects the on-disk file — used for the soft-warn path."""
    spool = Spool(tmp_path / "spool.sqlite")
    initial = spool.size_bytes()
    for i in range(20):
        spool.add(f"01ULID{i:04d}", b"x" * 4096, created_at=time.time() + i)
    assert spool.size_bytes() > initial
