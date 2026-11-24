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

import ctypes
import fcntl
import os
import struct
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
# Version tag of the out-of-tree record format (root-inode identity).
_RECORD_VERSION = "v2"


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


_AT_FDCWD = -100
_STATX_INO = 0x100
_STATX_BTIME = 0x800
_FS_IOC_GETVERSION = 0x80087601  # _IOR('v', 1, long): inode generation


def _statx_btime_ns(path: Path) -> int | None:
    """Birth time (ns) of ``path`` via ``statx(2)``; None if unsupported.

    Python 3.13 has no ``os.statx`` and ``os.stat`` exposes no birth time on
    Linux, so call glibc's ``statx`` directly. Layout per ``struct statx``:
    ``stx_mask`` at offset 0, ``stx_btime`` (s64 sec, u32 nsec) at offset 80.
    """
    statx = getattr(os, "statx", None)
    if statx is not None:  # pragma: no cover - Python >= 3.15
        try:
            st = statx(path, _STATX_BTIME)
            bt = getattr(st, "stx_birthtime_ns", None)
            return int(bt) if bt else None
        except OSError:
            return None
    libc = _libc()
    if libc is None or not hasattr(libc, "statx"):
        return None
    buf = ctypes.create_string_buffer(256)
    if libc.statx(_AT_FDCWD, os.fsencode(path), 0, _STATX_INO | _STATX_BTIME, buf) != 0:
        return None
    (mask,) = struct.unpack_from("I", buf, 0)
    if not mask & _STATX_BTIME:
        return None
    sec, nsec = struct.unpack_from("qI", buf, 80)
    return int(sec) * 1_000_000_000 + int(nsec)


def _libc() -> Any:
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:  # pragma: no cover - non-glibc platform
        return None


def _inode_generation(path: Path) -> int | None:
    """Inode generation number (``FS_IOC_GETVERSION``); None if unsupported.

    ext4 assigns a fresh random generation every time an inode number is
    (re)allocated, which disambiguates a reused root inode even when two
    directory creations fall in the same coarse-clock tick (identical btime).
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None
    try:
        raw = fcntl.ioctl(fd, _FS_IOC_GETVERSION, b"\0" * 8)
        return int(struct.unpack("l", raw)[0])
    except OSError:
        return None
    finally:
        os.close(fd)


def _dir_identity(host_dir: Path) -> str | None:
    """Identity of the directory INSTANCE at ``host_dir``; None if absent.

    Built ONLY from immutable properties of the root directory inode — its
    inode number, birth time (``statx`` btime) and inode generation — none of
    which a process can change on an existing inode with chmod/chown/touch/
    utime/link/rename/unlink inside the directory. The sandbox sees
    ``host_dir`` as its bind-mount root, so it cannot replace the root inode
    either: anything it does there happens IN PLACE and leaves this identity
    unchanged (F1/F4).

    A wholesale restore (``rm -rf dir; cp -a backup dir``) creates a NEW root
    inode. Even when ext4 hands back the just-freed inode number, the new
    inode gets a new birth time and a fresh random generation, and neither is
    copied by ``cp -a``/rsync/``copy2`` (F3).

    Deliberately NOT used: the in-tree ``.materialized`` marker (the sandbox
    can chmod/touch/link/delete it, bumping its ctime or inode), any ctime or
    mtime (changed by in-place metadata ops), and ``st_dev`` (unstable across
    remounts of network/overlay filesystems).
    """
    try:
        root = host_dir.stat()
    except OSError:
        return None
    btime = _statx_btime_ns(host_dir)
    gen = _inode_generation(host_dir)
    return f"{root.st_ino}:{btime if btime is not None else '-'}:{gen if gen is not None else '-'}"


def _read_record(record: Path) -> str | None:
    """Parse the identity stored in the out-of-tree record; None if absent/garbled."""
    try:
        parts = record.read_text().split()
    except OSError:
        return None
    if len(parts) != 2 or parts[0] != _RECORD_VERSION:
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
        atomic_write(record, f"{_RECORD_VERSION} {identity}\n")


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
    The record cannot be rolled back with the directory, and the identity is
    taken only from the root directory inode (number + birth time + inode
    generation), which no in-place operation — including metadata changes to
    the in-tree marker or the root — can alter, and which a restored copy
    cannot carry over even when the filesystem reuses the root inode number.
    A mismatch means the directory was replaced/restored wholesale —
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
            current = _dir_identity(host_dir)
            if recorded is not None and recorded == current:
                # Same directory instance we materialized (root inode, birth
                # time and generation unchanged). Everything a sandbox can do
                # inside its bind mount — rm, edits, chmod/chown/touch/link on
                # the marker or root, even deleting the marker — happens IN
                # PLACE and leaves this identity intact. Any DB row missing on
                # disk was therefore removed in place, by a sandbox command
                # whose reconcile has not run yet (or timed out) or by a
                # mirror delete. Writing it back would resurrect a deletion
                # (F1/F4); the bash reconcile is the authority here.
                if not marker_present:
                    atomic_write(marker, f"{_MARKER_VERSION}\n")
                return False
            if marker_present and recorded is None:
                # Marker with no (current-format) out-of-tree record: a legacy
                # marker (pre-#1721 empty file, or a pre-record identity
                # marker). There is no trustworthy record of which directory
                # was materialized, so a missing file is indistinguishable
                # from an unreconciled sandbox delete. Adopt the current
                # directory without restoring anything.
                _stamp(host_dir, record)
                return False
            # Reaching here with ``recorded`` set means the directory at
            # host_dir is not the instance we materialized: it was replaced /
            # restored from a copy or wiped and recreated (identity mismatch).
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
