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
from pathlib import Path
from typing import Any

import asyncpg

from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.atomic_mirror import atomic_write
from aios.sandbox.volumes import (
    ensure_owned_dir,
    memory_store_host_dir,
    memory_store_lock_path,
)

log = get_logger("aios.sandbox.memory_mounts")

MATERIALIZED_MARKER = ".materialized"
_MARKER_VERSION = "v1"


def _dir_identity(host_dir: Path) -> int:
    """Inode of the store root directory.

    ``st_dev`` is deliberately excluded: device numbers of network/overlay
    filesystems can change across a reboot/remount, which would make every
    store look "replaced" and turn an unreconciled delete into a restore.
    """
    return host_dir.stat().st_ino


def _read_marker_identity(marker: Path) -> int | None:
    """Parse the directory identity recorded in ``marker``; None if absent/legacy."""
    try:
        parts = marker.read_text().split()
    except OSError:
        return None
    if len(parts) != 2 or parts[0] != _MARKER_VERSION:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _write_marker(marker: Path, identity: int) -> None:
    atomic_write(marker, f"{_MARKER_VERSION} {identity}\n")


async def materialize_store_to_host(
    conn: asyncpg.Connection[Any],
    *,
    store_id: str,
    account_id: str,
) -> bool:
    """Ensure ``store_id``'s host dir is populated from DB.

    Returns whether a marked but rolled-back directory was repaired.

    Acquires a file lock so concurrent provisioning across sessions
    serializes; the loser observes the marker and returns immediately.
    Once a store is materialized, this function is a no-op for as long as
    the same directory stays in place: subsequent DB drift is propagated by
    the per-write mirror helpers and bash reconcile, not by
    re-materialization. A DB row missing from that directory is NOT evidence
    of a rollback — a sandbox ``rm`` whose reconcile has not landed yet (a
    peer session's bash still running, or a reconcile scan that timed out)
    looks exactly the same, and restoring it would resurrect the delete.

    A rollback is detected only when the marker records a different
    directory identity (root ``st_ino``) than the directory now at
    ``host_dir`` — i.e. the directory was replaced or restored wholesale,
    which no sandbox command can do to its own bind-mount root. Only then
    are missing files filled in from the DB (never overwriting files on
    disk).
    """
    host_dir = memory_store_host_dir(store_id)
    marker = host_dir / MATERIALIZED_MARKER
    lock_path = memory_store_lock_path(store_id)
    ensure_owned_dir(lock_path.parent)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            was_materialized = marker.exists()
            ensure_owned_dir(host_dir)
            if was_materialized:
                recorded = _read_marker_identity(marker)
                current = _dir_identity(host_dir)
                if recorded is None:
                    # Legacy (pre-#1721) empty marker: there is no record of
                    # which directory was materialized, so a missing file is
                    # indistinguishable from an unreconciled sandbox delete.
                    # Adopt the current directory without restoring anything.
                    _write_marker(marker, current)
                    return False
                if recorded == current:
                    # Same directory we materialized. Any DB row missing on
                    # disk was removed IN PLACE — by a sandbox command whose
                    # reconcile has not run yet (or timed out), or by a mirror
                    # delete. Writing it back would resurrect a deletion
                    # (F1); the bash reconcile is the authority here.
                    return False
                # The marker describes a different directory than the one now
                # at host_dir: the directory was replaced/restored wholesale
                # (a rollback). A sandbox cannot cause this — the store root
                # is a bind-mount point, which cannot be removed or replaced
                # from inside the container — so filling gaps from the DB
                # cannot undo a sandbox delete.

            entries = await queries.list_active_memory_paths_and_content(
                conn, store_id, account_id=account_id
            )
            live_paths = {path for path, _content in entries}
            disk_paths = {
                "/" + str(path.relative_to(host_dir)).replace("\\", "/")
                for path in host_dir.rglob("*")
                if path.is_file() and path.name != MATERIALIZED_MARKER
            }
            restoring = was_materialized and not live_paths.issubset(disk_paths)

            for path, content in entries:
                # Memory paths are guaranteed to start with "/" by the SQL CHECK.
                target = host_dir / path.lstrip("/")
                # Defer to concurrent writers. host_dir was created empty
                # just above (mkdir), so any file already at ``target`` was
                # placed there by an ``_mirror_to_host`` call that fired
                # during this function's snapshot-read yield — i.e. by a
                # caller who committed to DB AFTER our snapshot. Their
                # value supersedes ours. Without this guard the stale-
                # snapshot ``atomic_write`` clobbers the fresher mirror
                # and leaves DB/disk permanently inconsistent.
                # On a rollback repair the same guard matters more: a file
                # still on disk may hold a sandbox-side edit not yet
                # reconciled to the DB; restore only fills in what is
                # missing so it never discards degraded-window writes.
                if target.exists():
                    log.info(
                        "memory.materialize_skipped_existing",
                        store_id=store_id,
                        path=path,
                    )
                    continue
                atomic_write(target, content or "")

            # Marker last so a crash mid-materialization leaves an unmarked
            # dir that the next attempt will redo. It records the identity of
            # the directory it describes so a later provision can tell an
            # in-place sandbox delete (same directory) from a wholesale
            # rollback/replacement (different directory).
            _write_marker(marker, _dir_identity(host_dir))
            log.info(
                "memory.materialized",
                store_id=store_id,
                count=len(entries),
                restored=restoring,
            )
            return restoring
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
