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

# ─── files ───────────────────────────────────────────────────────────────────


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
