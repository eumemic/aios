"""Low-cadence edge-owned-lifetime invariant reconciler (C5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from aios.config import Settings, get_settings
from aios.db.queries.sessions import session_active_predicate
from aios.logging import get_logger
from aios.services import sessions as sessions_service
from aios.services.sessions import REVOCATION_KINDS

log = get_logger("aios.harness.invariant_sweep")

# The per-request ``deadline_seconds`` override (abandoned-api clause) accepts an
# integer or decimal. Bound as a QUERY PARAMETER ($6), not embedded in the SQL
# literal: a bound param's bytes reach PostgreSQL's regex engine verbatim, with no
# SQL-string-literal escaping layer in between — so ``\.`` unambiguously means a
# literal decimal point regardless of ``standard_conforming_strings``. (Embedding it
# forces a double-escape dance where an over-escaped ``\\.`` silently matches a literal
# backslash instead of the dot, dropping fractional overrides like ``1.5``.)
_DEADLINE_SECONDS_REGEX = r"^[0-9]+(\.[0-9]+)?$"

# Clauses that are SELECTED and logged but never actuate an archive. The
# abandoned-api-lease clause stays measurement-only until its dry-run cohort is
# reviewed for legitimate long jobs.
_MEASUREMENT_ONLY_CLAUSES: frozenset[str] = frozenset({"abandoned_api_lease"})

# One bounded query deliberately keeps the creation-edge lateral pinned to the
# parent run.  In particular, a later inbound Ask must never reclassify a Tell.
INVARIANT_CANDIDATES_SQL = f"""
WITH candidate AS (
 SELECT s.id AS session_id, s.account_id, s.created_at, s.parent_run_id,
        ce.data AS creation, ce.created_at AS creation_at,
        cr.data AS creation_response,
        (SELECT max(r.created_at) FROM events r
          WHERE r.session_id=s.id AND r.account_id=s.account_id
            AND r.kind='lifecycle' AND r.data->>'event'='request_response') AS last_closure
 FROM sessions s
 LEFT JOIN LATERAL (
   SELECT e.data, e.created_at FROM events e
    WHERE e.session_id=s.id AND e.account_id=s.account_id
      AND e.kind='lifecycle' AND e.data->>'event'='request_opened'
      AND e.data->'caller'->>'kind' = 'run'
      AND e.data->'caller'->>'id' = s.parent_run_id
    ORDER BY e.seq LIMIT 1
 ) ce ON true
 LEFT JOIN LATERAL (
   SELECT r.data FROM events r
    WHERE r.session_id=s.id AND r.account_id=s.account_id
      AND r.kind='lifecycle' AND r.data->>'event'='request_response'
      AND r.data->>'request_id'=ce.data->>'request_id'
    ORDER BY r.seq DESC LIMIT 1
 ) cr ON true
 WHERE s.archived_at IS NULL AND ($7::text IS NULL OR s.id = $7::text)
), open_edges AS (
 SELECT c.*, e.data AS edge, e.created_at AS edge_created
 FROM candidate c JOIN events e ON e.session_id=c.session_id AND e.account_id=c.account_id
 WHERE e.kind='lifecycle' AND e.data->>'event'='request_opened'
   AND COALESCE((e.data->>'awaited')::boolean, true)
   AND NOT EXISTS (SELECT 1 FROM events rr WHERE rr.session_id=e.session_id
     AND rr.account_id=e.account_id AND rr.kind='lifecycle'
     AND rr.data->>'event'='request_response'
     AND rr.data->>'request_id'=e.data->>'request_id')
), selected AS (
 SELECT c.session_id, c.account_id, 'revoked_lease'::text AS clause
 FROM candidate c JOIN wf_runs p ON p.id=c.parent_run_id AND p.account_id=c.account_id
 WHERE p.status=ANY($1::text[]) AND c.created_at < now()-interval '10 minutes'
   AND COALESCE((c.creation->>'awaited')::boolean, true)
   AND (c.creation IS NULL OR c.creation_response IS NULL
        OR c.creation_response->'error'->>'kind' = ANY($4::text[]))
   AND NOT EXISTS (SELECT 1 FROM open_edges o WHERE o.session_id=c.session_id
     AND (o.edge->'caller'->>'kind'='api'
       OR (o.edge->'caller'->>'kind'='run' AND EXISTS (SELECT 1 FROM wf_runs lr
           WHERE lr.id=o.edge->'caller'->>'id' AND lr.status<>ALL($1::text[])))
       OR (o.edge->'caller'->>'kind'='session' AND EXISTS (SELECT 1 FROM sessions ls
           WHERE ls.id=o.edge->'caller'->>'id' AND ls.archived_at IS NULL))))
 UNION ALL
 SELECT c.session_id,c.account_id,'expired_idle' FROM candidate c JOIN sessions s ON s.id=c.session_id
 WHERE s.archive_when_idle AND c.created_at < now()-interval '10 minutes'
   AND COALESCE(c.last_closure,c.created_at) < now()-interval '10 minutes'
   AND NOT {session_active_predicate("s")}
   AND NOT EXISTS (SELECT 1 FROM open_edges o WHERE o.session_id=c.session_id)
 UNION ALL
 SELECT DISTINCT c.session_id,c.account_id,'abandoned_api_lease' FROM candidate c
 JOIN open_edges o ON o.session_id=c.session_id JOIN sessions s ON s.id=c.session_id
 WHERE s.archive_when_idle AND o.edge->'caller'->>'kind'='api'
   AND o.edge_created + make_interval(secs => COALESCE(
       CASE WHEN (o.edge->>'deadline_seconds') ~ $6
            THEN (o.edge->>'deadline_seconds')::double precision END, $2)) < now()
 UNION ALL
 SELECT c.session_id,c.account_id,'lease_ceiling' FROM candidate c JOIN sessions s ON s.id=c.session_id
 WHERE s.archive_when_idle
   AND COALESCE(c.last_closure,c.created_at) < now()-make_interval(secs=>$3)
   AND NOT EXISTS (SELECT 1 FROM open_edges o WHERE o.session_id=c.session_id)
)
SELECT DISTINCT ON (session_id) session_id,account_id,clause FROM selected
ORDER BY session_id, CASE clause WHEN 'revoked_lease' THEN 1 WHEN 'expired_idle' THEN 2
                                WHEN 'abandoned_api_lease' THEN 3 ELSE 4 END
LIMIT $5
"""


@dataclass(frozen=True, slots=True)
class InvariantSweepResult:
    selected: int
    archived: int


def _candidate_query_args(
    settings: Settings, *, batch_size: int, session_id: str | None
) -> list[Any]:
    """Positional args for :data:`INVARIANT_CANDIDATES_SQL`. ``session_id`` pins the
    scan to a single session ($7) for the under-lock re-validation; ``None`` selects
    the whole batch."""
    return [
        ["completed", "errored", "cancelled"],
        settings.api_lease_deadline_seconds,
        settings.lease_ceiling_seconds,
        list(REVOCATION_KINDS),
        batch_size,
        _DEADLINE_SECONDS_REGEX,
        session_id,
    ]


async def _revalidate_and_archive(
    pool: asyncpg.Pool[Any], session_id: str, account_id: str, *, settings: Settings
) -> bool:
    """Actuate an archive ONLY if an archiving clause STILL holds, re-checked under the
    session row lock (read/actuation TOCTOU fix).

    The sole writer that opens a new awaited inbound edge (``append_event``) flips the
    session's ``last_event_seq`` under the sessions row lock and is fenced by
    ``archived_at IS NULL``; taking that SAME row lock ``FOR UPDATE`` here serializes this
    actuation against a concurrently-opening edge — and against an expired-idle candidate
    going active. We re-run the IDENTICAL selection predicate pinned to this one session,
    so a candidate that has since gained a live awaited inbound edge, gone active, or
    otherwise stopped satisfying its clause is SKIPPED and left for the next sweep, never
    force-archived (preserving the zero-open-edge / live-caller / fulfilled-edge-active-
    before-ceiling protections). The archive itself runs in the SAME transaction via
    :func:`sessions_service.archive_session_conn`, so no edge can open between the re-check
    and the flip. Returns True iff this call archived.
    """
    async with pool.acquire() as conn, conn.transaction():
        locked = await conn.fetchrow(
            "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2 "
            "AND archived_at IS NULL FOR UPDATE",
            session_id,
            account_id,
        )
        if locked is None:
            return False  # already archived / deleted since selection — nothing to do
        rows = await conn.fetch(
            INVARIANT_CANDIDATES_SQL,
            *_candidate_query_args(settings, batch_size=1, session_id=session_id),
        )
        still_selected = next(
            (r for r in rows if r["clause"] not in _MEASUREMENT_ONLY_CLAUSES), None
        )
        if still_selected is None:
            # Raced: no archiving clause holds at actuation time — leave for next sweep.
            log.info("invariant_sweep.revalidation_skip", session_id=session_id)
            return False
        commit = await sessions_service.archive_session_conn(
            conn, session_id, account_id=account_id, idempotent=True
        )
    await sessions_service.archive_session_post_commit(
        pool, session_id, commit, account_id=account_id
    )
    return True


async def sweep_session_invariants(
    pool: asyncpg.Pool[Any], *, dry_run: bool | None = None, batch_size: int = 200
) -> InvariantSweepResult:
    settings = get_settings()
    if dry_run is None:
        dry_run = settings.invariant_sweep_dry_run
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            INVARIANT_CANDIDATES_SQL,
            *_candidate_query_args(settings, batch_size=batch_size, session_id=None),
        )
    archived = 0
    for row in rows:
        clause = row["clause"]
        log.info(
            "invariant_sweep.candidate",
            session_id=row["session_id"],
            account_id=row["account_id"],
            clause=clause,
            dry_run=dry_run,
        )
        # API abandonment remains measurement-only even after the other clauses arm.
        if dry_run or clause in _MEASUREMENT_ONLY_CLAUSES or not settings.cancel_cascade_enabled:
            continue
        # Re-validate under the session row lock immediately before archiving: the rows
        # above were selected on a since-released connection, so a new awaited inbound
        # edge (or a re-activated idle candidate) may have raced in. Skip any row whose
        # clause no longer holds — never archive a now-protected session.
        if await _revalidate_and_archive(
            pool, row["session_id"], row["account_id"], settings=settings
        ):
            archived += 1
    if not dry_run and settings.cancel_cascade_enabled:
        async with pool.acquire() as conn:
            await conn.execute("""UPDATE session_cancel_markers m SET harvested_at=now()
                FROM sessions s WHERE s.id=m.session_id AND s.account_id=m.account_id
                AND s.archived_at IS NOT NULL AND m.harvested_at IS NULL""")
    return InvariantSweepResult(len(rows), archived)
