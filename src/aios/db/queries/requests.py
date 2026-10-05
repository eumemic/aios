"""Queries for captured requests (#2471): the ``request_blobs`` table and the
request record a span carries.

Blobs are content-addressed per account (migration 0186). A request span is a
``model_request_start`` (a native model call) or a ``model_workflow_park`` (a
workflow-as-model launch) whose ``data`` carries a ``request`` record.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.db.queries.events import _row_to_event
from aios.models.events import Event


async def insert_blobs(
    conn: asyncpg.Connection[Any], *, account_id: str, blobs: dict[str, bytes]
) -> None:
    """Store each ``sha256 → body`` the account doesn't already have."""
    if not blobs:
        return
    shas = list(blobs)
    await conn.execute(
        "INSERT INTO request_blobs (account_id, sha256, body) "
        "SELECT $1, sha, body FROM unnest($2::text[], $3::bytea[]) AS b(sha, body) "
        "ON CONFLICT (account_id, sha256) DO NOTHING",
        account_id,
        shas,
        [blobs[sha] for sha in shas],
    )


async def get_blobs(
    conn: asyncpg.Connection[Any], *, account_id: str, shas: list[str]
) -> dict[str, bytes]:
    """The account's blobs among ``shas``; a sha it doesn't have is absent."""
    rows = await conn.fetch(
        "SELECT sha256, body FROM request_blobs WHERE account_id = $1 AND sha256 = ANY($2)",
        account_id,
        shas,
    )
    return {row["sha256"]: bytes(row["body"]) for row in rows}


async def get_events_by_seq(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str, seqs: list[int]
) -> list[Event]:
    """The session's events at ``seqs``, in seq order."""
    rows = await conn.fetch(
        "SELECT * FROM events WHERE session_id = $1 AND account_id = $2 AND seq = ANY($3) "
        "ORDER BY seq",
        session_id,
        account_id,
        seqs,
    )
    return [_row_to_event(row) for row in rows]
