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


def _record_path(lock_path: Path) -> Path:
    """Out-of-tree materialization record for the store whose lock is ``lock_path``.

    Lives next to the lock file (``<store_id>.materialized``), i.e. OUTSIDE the
    store's host directory, so restoring the host directory from a backup
    cannot carry an old copy of it back into place.
    """
    return lock_path.with_suffix(".materialized")


def materialization_record_path(store_id: str) -> Path:
    """Public accessor so store deletion can drop the out-of-tree record."""
    return _record_path(memory_store_lock_path(store_id))


def _dir_identity(host_dir: Path) -> str | None:
    """Identity of the materialized directory instance; None if the marker is gone.

    Combines the root directory inode with the in-tree marker file's inode and
    **ctime**. A wholesale restore (``rm -rf dir; cp -a backup dir``, rsync of a
    snapshot, …) can reuse the freed root inode — ext4 hands the just-freed
    number straight back — and it carries the old marker bytes, but it cannot
    reproduce the marker's ctime: ctime is set by the kernel to "now" whenever
    the inode is created or its metadata changes and is not settable from
    userspace (``cp -a``/``copy2`` preserve mtime/atime only). An in-place
    sandbox ``rm``/edit of other files never touches the marker inode, so the
    identity stays stable across it (F1).

    ``st_dev`` is deliberately excluded: device numbers of network/overlay
    filesystems can change across a reboot/remount, which would make every
    store look "replaced" and turn an unreconciled delete into a restore.
    """
    try:
        root = host_dir.stat()
        mst = (host_dir / MATERIALIZED_MARKER).stat()
    except OSError:
        return None
    return f"{root.st_ino}:{mst.st_ino}:{mst.st_ctime_ns}"


def _read_record(record: Path) -> str | None:
    """Parse the identity stored in the out-of-tree record; None if absent/garbled."""
    try:
        parts = record.read_text().split()
    except OSError:
        return None
    if len(parts) != 2 or parts[0] != _MARKER_VERSION:
        return None
    return parts[1]


def _stamp(host_dir: Path, record: Path) -> None:
    """Write the in-tree marker, then record its identity out of tree.

    The in-tree marker stays because the bash reconcile and the audit gate on
    it (an unmarked dir is "not materialized here"). The out-of-tree record is
    what rollback detection trusts.
    """
    atomic_write(host_dir / MATERIALIZED_MARKER, f"{_MARKER_VERSION}\n")
    identity = _dir_identity(host_dir)
    if identity is not None:
        atomic_write(record, f"{_MARKER_VERSION} {identity}\n")


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

    A rollback is detected by comparing the identity of the directory now at
    ``host_dir`` (see :func:`_dir_identity`) with the identity recorded in an
    out-of-tree record next to the lock file at the last materialization.
    The record cannot be rolled back with the directory, and the identity
    includes the in-tree marker's ctime, which a restored copy cannot carry
    over even when the filesystem reuses the root inode. A mismatch (or a
    vanished marker) means the directory was replaced/restored wholesale —
    something no sandbox command can do to its own bind-mount root. Only then
    are missing files filled in from the DB (never overwriting files on disk).
    """
    host_dir = memory_store_host_dir(store_id)
    marker = host_dir / MATERIALIZED_MARKER
    lock_path = memory_store_lock_path(store_id)
    ensure_owned_dir(lock_path.parent)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            record = _record_path(lock_path)
            recorded = _read_record(record)
            marker_present = marker.exists()
            ensure_owned_dir(host_dir)
            if marker_present:
                current = _dir_identity(host_dir)
                if recorded is None:
                    # Marker with no out-of-tree record: a legacy marker
                    # (pre-#1721 empty file, or a pre-record identity marker).
                    # There is no trustworthy record of which directory was
                    # materialized, so a missing file is indistinguishable from
                    # an unreconciled sandbox delete. Adopt the current
                    # directory without restoring anything.
                    _stamp(host_dir, record)
                    return False
                if recorded == current:
                    # Same directory instance we materialized. Any DB row
                    # missing on disk was removed IN PLACE — by a sandbox
                    # command whose reconcile has not run yet (or timed out),
                    # or by a mirror delete. Writing it back would resurrect a
                    # deletion (F1); the bash reconcile is the authority here.
                    return False
            # Reaching here with ``recorded`` set means the directory at
            # host_dir is not the instance we materialized: it was replaced /
            # restored from a copy (identity mismatch) or wiped (marker gone).
            # A sandbox cannot replace its bind-mount root, so filling gaps
            # from the DB cannot undo a sandbox delete.
            was_materialized = recorded is not None

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
            # dir that the next attempt will redo. The out-of-tree record
            # captures this directory instance's identity so a later
            # provision can tell an in-place sandbox delete (same instance)
            # from a wholesale rollback/replacement (different instance).
            _stamp(host_dir, record)
            log.info(
                "memory.materialized",
                store_id=store_id,
                count=len(entries),
                restored=restoring,
            )
            return restoring
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
