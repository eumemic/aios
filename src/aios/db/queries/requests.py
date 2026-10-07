"""Queries for captured requests (#2471): the ``request_blobs`` table and the
request record a span carries.

Blobs are content-addressed per account (migration 0186). A request span is a
``model_request_start`` (a native model call) or a ``model_workflow_park`` (a
workflow-as-model launch) whose ``data`` carries a ``request`` record.
"""

from __future__ import annotations

from datetime import datetime
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


async def present_blob_shas(
    conn: asyncpg.Connection[Any], *, account_id: str, shas: list[str]
) -> set[str]:
    """Which of ``shas`` the account has, without reading the bodies."""
    rows = await conn.fetch(
        "SELECT sha256 FROM request_blobs WHERE account_id = $1 AND sha256 = ANY($2)",
        account_id,
        shas,
    )
    return {row["sha256"] for row in rows}


# The request spans of one agent's sessions in ``[start, end)``, one per payload,
# capped per (session, UTC day), in a seeded order. A request counts only if it was
# answered: a ``model_request_start`` closed by a successful ``model_request_end``
# (a cancelled end carries no ``model``; a refusal's end, ``finish_reason`` =
# ``content_filter``, persists no assistant message), or the last
# ``model_workflow_park`` for a run that was harvested without error (a relaunch
# re-parks the same run, and only the last park's request was sent). Only an
# answered request has its own assistant message: the first one after its span,
# since a session runs one inference at a time. The closing span may land after
# ``end``, so the span read runs a day past it. Sessions a replay run spawned (its
# tree is the only run stamped ``session`` that doesn't act for a session) are eval
# traffic, not production, and are skipped.
_SAMPLE_REQUEST_SPANS_SQL = """
WITH agent_sessions AS (
    SELECT s.id
      FROM sessions s
     WHERE s.account_id = $1 AND s.agent_id = $2
       AND NOT EXISTS (
           SELECT 1 FROM wf_runs r
            WHERE r.id = s.parent_run_id
              AND r.principal <> 'session' AND r.visibility = 'session'
       )
),
spans AS (
    SELECT e.id, e.session_id, e.seq, e.created_at, e.data
      FROM agent_sessions a
      JOIN events e ON e.session_id = a.id
     WHERE e.account_id = $1 AND e.kind = 'span'
       AND e.created_at >= $3 AND e.created_at < $4::timestamptz + interval '1 day'
       AND e.data->>'event' IN (
           'model_request_start', 'model_workflow_park',
           'model_request_end', 'model_workflow_harvest_end'
       )
),
-- A request is answered if a span closed it: the start's id on a successful end, or
-- the park's run id on a clean harvest. Uncorrelated IN lists, so each is hashed once
-- instead of scanned per request (the planner can't size these jsonb filters).
answered AS (
    SELECT DISTINCT ON (data->'request'->>'payload_sha')
           id, session_id, seq, created_at, data->'request' AS record
      FROM spans
     WHERE created_at < $4
       AND data->>'event' IN ('model_request_start', 'model_workflow_park')
       AND NOT data ? 'purpose'
       AND data->'request' ? 'payload_sha'
       AND data->'request'->'binding'->>'agent_id' = $2
       AND (
           (data->>'event' = 'model_request_start' AND id IN (
               SELECT data->>'model_request_start_id' FROM spans
                WHERE data->>'event' = 'model_request_end'
                  AND data->>'is_error' = 'false' AND data ? 'model'
                  AND data->>'finish_reason' IS DISTINCT FROM 'content_filter'))
           OR (data->>'event' = 'model_workflow_park'
               AND data->>'run_id' IN (
                   SELECT data->>'run_id' FROM spans
                    WHERE data->>'event' = 'model_workflow_harvest_end'
                      AND data->>'is_error' = 'false')
               AND id IN (
                   SELECT DISTINCT ON (data->>'run_id') id FROM spans
                    WHERE data->>'event' = 'model_workflow_park'
                    ORDER BY data->>'run_id', seq DESC))
       )
     ORDER BY data->'request'->>'payload_sha', created_at, id
),
capped AS (
    SELECT *, row_number() OVER (
               PARTITION BY session_id, date_trunc('day', created_at AT TIME ZONE 'UTC')
               ORDER BY md5($5 || ':' || id)
           ) AS rank_in_cluster
      FROM answered
)
SELECT c.id, c.session_id, c.created_at, c.record,
       (SELECT m.id FROM events m
         WHERE m.session_id = c.session_id AND m.kind = 'message' AND m.seq > c.seq
           AND m.data->>'role' = 'assistant'
         ORDER BY m.seq LIMIT 1) AS response_event_id
  FROM capped c
 WHERE c.rank_in_cluster <= $6
 ORDER BY md5($5 || '|' || c.id)
 LIMIT $7
"""


async def sample_request_spans(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    agent_id: str,
    start: datetime,
    end: datetime,
    seed: str,
    cluster_cap: int,
    n: int,
) -> list[asyncpg.Record]:
    """A seeded sample of the answered requests ``agent_id`` sent in ``[start, end)``
    (see :data:`_SAMPLE_REQUEST_SPANS_SQL`). Rows: the span ``id``, ``session_id``,
    ``created_at``, its ``request`` ``record``, and the ``response_event_id`` of the
    assistant message that answered it."""
    rows: list[asyncpg.Record] = await conn.fetch(
        _SAMPLE_REQUEST_SPANS_SQL, account_id, agent_id, start, end, seed, cluster_cap, n
    )
    return rows


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
