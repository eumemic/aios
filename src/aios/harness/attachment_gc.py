"""Worker startup orphan attachment GC.

Walks ``<workspace_root>/_attachments/`` and removes any staged file
not referenced by an event in the DB. Files become orphans when an
inbound stages successfully but its dedup transaction rolls back, or
when staging crashes between rename and append.

Runs once at startup; sessions with no on-disk dir contribute zero
work. Empty session dirs are left in place so the bind-mount source
exists for the next provision.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from typing import Any

import asyncpg

from aios.db import queries
from aios.sandbox.volumes import attachments_root, uploads_root

# Files staged within this many seconds of sweep time are considered
# in-flight: ``stage_inbound_attachments`` writes to disk BEFORE the
# ``_append_with_dedup`` transaction commits, so a sweep that runs in
# the gap between rename and commit would otherwise race-delete the
# file before the API's commit lands the referencing event. 300s is
# generous compared to a healthy staging-txn latency (single-digit ms),
# while still small enough that orphans get reaped promptly on the
# next worker boot.
_IN_FLIGHT_AGE_S = 300.0


class AttachmentGcError(RuntimeError):
    """Raised when the orphan sweep failed to delete one or more
    unreferenced files (perm drift, FS gone read-only, etc.).

    Worker startup deliberately surfaces this rather than silently
    accumulating un-collectable orphans across boots.  The exception
    message lists every failed path so the operator can act on the
    real cause.
    """


async def sweep_orphan_attachments(pool: asyncpg.Pool[Any]) -> int:
    """Delete files in ``_attachments/`` that no event row references.

    Returns the number of files deleted. Raises
    :class:`AttachmentGcError` if any file we identified as orphaned
    could not be unlinked — those failures (permission drift, FS gone
    read-only, root squash on the bind) are real signals about
    worker provisioning and silently swallowing them lets orphans
    accumulate forever across reboots.

    Note on session dirs without referencing events: if a session row
    or its events were purged but the on-disk
    ``_attachments/<session_id>/`` dir survived, this sweep will
    delete every file in that dir (the events query returns 0 rows
    → 0 referenced paths → all on-disk files look orphaned). That is
    the intended behavior given the design splits cleanup of stranded
    files from cleanup of empty dirs (the latter is a future polish
    item alongside session deletion).
    """
    root = attachments_root()
    if not root.exists():
        return 0

    # Skip files staged within the in-flight window — they may be
    # mid-transaction in another process, with their referencing
    # event not yet committed. Without this guard, a worker restart
    # that lands between ``stage_inbound_attachments`` rename and
    # ``_append_with_dedup`` commit would silently delete the just-
    # staged bytes, leaving the post-commit event pointing at a
    # missing file (renderer ``FileNotFoundError`` on every wake).
    in_flight_cutoff = time.time() - _IN_FLIGHT_AGE_S

    on_disk_by_session: dict[str, dict[str, Path]] = {}
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        session_files: dict[str, Path] = {}
        for connector_dir in session_dir.iterdir():
            if not connector_dir.is_dir():
                continue
            for file_path in connector_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.stat().st_mtime > in_flight_cutoff:
                    continue
                sandbox_path = f"/mnt/attachments/{connector_dir.name}/{file_path.name}"
                session_files[sandbox_path] = file_path
        if session_files:
            on_disk_by_session[session_dir.name] = session_files

    if not on_disk_by_session:
        return 0

    async with pool.acquire() as conn:
        referenced_by_session = await queries.list_attachment_paths_for_sessions(
            conn, list(on_disk_by_session)
        )

    deleted = 0
    failures: list[tuple[Path, OSError]] = []
    for session_id, session_files in on_disk_by_session.items():
        referenced = referenced_by_session.get(session_id, set())
        for sandbox_path, file_path in session_files.items():
            if sandbox_path in referenced:
                continue
            try:
                file_path.unlink()
                deleted += 1
            except OSError as err:
                failures.append((file_path, err))

    if failures:
        rendered = ", ".join(f"{p}: {e}" for p, e in failures)
        raise AttachmentGcError(
            f"failed to unlink {len(failures)} orphan attachment(s): {rendered}"
        )

    return deleted


async def sweep_orphan_uploads(pool: asyncpg.Pool[Any]) -> int:
    """Reconcile ``_uploads`` with live sessions and rows in ``files``.

    Directories for deleted sessions are removed wholesale.  For live sessions,
    unreferenced files older than the staging grace are removed; recent files
    may still be between their atomic rename and DB insert.
    """
    root = uploads_root()
    if not root.exists():
        return 0
    session_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not session_dirs:
        return 0
    async with pool.acquire() as conn:
        referenced = await queries.list_upload_paths_for_sessions(
            conn, [path.name for path in session_dirs]
        )

    cutoff = time.time() - _IN_FLIGHT_AGE_S
    deleted = 0
    failures: list[tuple[Path, OSError]] = []
    for session_dir in session_dirs:
        if session_dir.name not in referenced:
            files = [path for path in session_dir.rglob("*") if path.is_file()]
            try:
                shutil.rmtree(session_dir)
                deleted += len(files)
            except OSError as err:
                failures.append((session_dir, err))
            continue
        retained = referenced[session_dir.name]
        if retained is None:
            continue
        for file_path in session_dir.rglob("*"):
            if not file_path.is_file() or str(file_path) in retained:
                continue
            if file_path.stat().st_mtime > cutoff:
                continue
            try:
                file_path.unlink()
                deleted += 1
            except OSError as err:
                failures.append((file_path, err))
        for directory in sorted(
            (path for path in session_dir.rglob("*") if path.is_dir()), reverse=True
        ):
            with contextlib.suppress(OSError):
                directory.rmdir()

    if failures:
        rendered = ", ".join(f"{path}: {error}" for path, error in failures)
        raise AttachmentGcError(f"failed to remove {len(failures)} orphan upload(s): {rendered}")
    return deleted
