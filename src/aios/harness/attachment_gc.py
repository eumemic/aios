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

from pathlib import Path
from typing import Any

import asyncpg

from aios.db import queries
from aios.logging import get_logger
from aios.sandbox.volumes import attachments_root

log = get_logger("aios.harness.attachment_gc")


async def sweep_orphan_attachments(pool: asyncpg.Pool[Any]) -> int:
    """Delete files in ``_attachments/`` that no event row references.

    Returns the number of files deleted. Per-file unlink errors are
    logged and skipped so one bad permission bit doesn't strand the
    rest of startup.
    """
    root = attachments_root()
    if not root.exists():
        return 0

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
    for session_id, session_files in on_disk_by_session.items():
        referenced = referenced_by_session.get(session_id, set())
        for sandbox_path, file_path in session_files.items():
            if sandbox_path in referenced:
                continue
            try:
                file_path.unlink()
                deleted += 1
            except OSError as err:
                log.warning(
                    "attachment_gc.unlink_failed",
                    path=str(file_path),
                    error=str(err),
                )

    return deleted
