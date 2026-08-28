"""File queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.errors import (
    NotFoundError,
)
from aios.models.files import File

# ─── files ───────────────────────────────────────────────────────────────


def _row_to_file(row: asyncpg.Record) -> File:
    return File(
        id=row["id"],
        session_id=row["session_id"],
        filename=row["filename"],
        host_path=row["host_path"],
        in_sandbox_path=row["in_sandbox_path"],
        size=row["size"],
        content_type=row["content_type"],
        sha256=row["sha256"],
        created_at=row["created_at"],
    )


async def insert_file(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    file_id: str,
    session_id: str,
    filename: str,
    host_path: str,
    in_sandbox_path: str,
    size: int,
    content_type: str,
    sha256: str,
) -> File:
    """Insert a row for an already-staged upload.

    Caller has already written the bytes to ``host_path`` and computed
    ``sha256`` + ``size`` during streaming. Raises :class:`NotFoundError`
    if ``session_id`` doesn't exist (FK violation).
    """
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO files (
                id, session_id, filename, host_path, in_sandbox_path,
                size, content_type, sha256, account_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            file_id,
            session_id,
            filename,
            host_path,
            in_sandbox_path,
            size,
            content_type,
            sha256,
            account_id,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"session {session_id} not found",
            detail={"session_id": session_id},
        ) from exc
    assert row is not None
    return _row_to_file(row)


async def get_file(
    conn: asyncpg.Connection[Any], session_id: str, file_id: str, *, account_id: str
) -> File:
    """Scoped file read for the #179 image-serve slice.

    Raises :class:`NotFoundError` when the file doesn't exist, belongs to a
    different session, or isn't owned by ``account_id`` — same shape as
    :func:`get_session_bare`'s 404, so a wrong session or a cross-account
    file id are indistinguishable from a missing file.
    """
    row = await conn.fetchrow(
        "SELECT * FROM files WHERE id = $1 AND session_id = $2 AND account_id = $3",
        file_id,
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"file {file_id} not found",
            detail={"session_id": session_id, "file_id": file_id},
        )
    return _row_to_file(row)


async def list_upload_paths_for_sessions(
    conn: asyncpg.Connection[Any], session_ids: list[str]
) -> dict[str, set[str] | None]:
    """Return referenced upload host paths for requested live sessions.

    A missing key identifies a deleted session. ``None`` identifies a live
    session with no file rows, where the database cannot determine whether
    on-disk files are orphaned. A non-empty set authoritatively identifies the
    known files and permits per-file reconciliation.
    """
    if not session_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT requested.id AS session_id, f.host_path
          FROM unnest($1::text[]) AS requested(id)
          JOIN sessions s ON s.id = requested.id
          LEFT JOIN files f ON f.session_id = s.id
        """,
        session_ids,
    )
    result: dict[str, set[str] | None] = {}
    for row in rows:
        session_id = row["session_id"]
        host_path = row["host_path"]
        if host_path is None:
            result.setdefault(session_id, None)
            continue
        paths = result.get(session_id)
        if paths is None:
            paths = set()
            result[session_id] = paths
        paths.add(host_path)
    return result
