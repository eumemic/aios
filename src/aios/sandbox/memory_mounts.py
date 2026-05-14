"""Materialize memory-store content into the shared host directory.

Each memory store gets exactly one host directory at
``<workspace_root>/_memory_stores/<store_id>/``, bind-mounted into every
attached session's container at ``/mnt/memory/<store_name>/``. Sharing
the source dir is what makes cross-session reads live: a tool write from
session A appears in session B's mount immediately.

Materialization is **lazy and one-shot per store**. The first time any
session provisions for store S we acquire a file lock, dump every
non-deleted memory from DB to the host dir, and drop a ``.materialized``
marker file. Subsequent provisioning sees the marker and is a no-op.

Tool/API writes after materialization keep the host dir in sync via the
atomic mirror helpers in :mod:`aios.sandbox.atomic_mirror`. Bash writes
hit the host dir directly and are reconciled back to the DB by the
post-exec hook in :mod:`aios.tools.bash_memory_reconcile`.
"""

from __future__ import annotations

import fcntl
from typing import Any

import asyncpg

from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.atomic_mirror import atomic_write
from aios.sandbox.volumes import memory_store_host_dir, memory_store_lock_path

log = get_logger("aios.sandbox.memory_mounts")

MATERIALIZED_MARKER = ".materialized"


async def materialize_store_to_host(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
) -> None:
    """Ensure ``store_id``'s host dir is populated from DB. Idempotent.

    Acquires a file lock so concurrent provisioning across sessions
    serializes; the loser observes the marker and returns immediately.
    Once a store is materialized, this function is a constant-time no-op.
    Subsequent DB drift is propagated by the per-write mirror helpers,
    not by re-materialization.
    """
    host_dir = memory_store_host_dir(store_id)
    marker = host_dir / MATERIALIZED_MARKER
    if marker.exists():
        return

    lock_path = memory_store_lock_path(store_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if marker.exists():
                return  # another waiter materialized while we blocked
            host_dir.mkdir(parents=True, exist_ok=True)

            entries = await queries.list_active_memory_paths_and_content(conn, store_id)
            for path, content in entries:
                # Memory paths are guaranteed to start with "/" by the SQL CHECK.
                atomic_write(host_dir / path.lstrip("/"), content or "")

            # Marker last so a crash mid-materialization leaves an unmarked
            # dir that the next attempt will redo.
            marker.touch()
            log.info(
                "memory.materialized",
                store_id=store_id,
                count=len(entries),
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
