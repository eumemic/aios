"""Add ``events_tool_confirmed_allow_recent_idx`` — a ``created_at``-keyed
partial index that lets the sweep's cross-session confirmed-rows detector
prune at the index rather than heap-fetching every confirmed-allow row ever
(#1740, sub-issue of the zero-inference-gap epic #1733).

Every sweep pass runs ``sweep.CONFIRMED_ROWS_SQL`` (composed from
``queries.confirmed_unresolved_predicate``) to find sessions with a
``tool_confirmed``/``allow`` lifecycle event whose ``tool_call_id`` has no
result yet — case (c) of ``find_sessions_needing_inference``. The only index
covering this predicate, ``events_tool_confirmed_allow_idx`` (migration 0065),
is keyed ``(session_id, (data->>'tool_call_id'))``: it serves the per-session
dispatch resolver (``queries.list_confirmed_unresolved_tool_calls``, which
seeks on ``session_id = $1``), but the sweep's cross-session query has no
``session_id`` equality to seek on, so it heap-fetches every confirmed-allow
row in the table before ``confirmed_dispatch_max_age_seconds`` (default 1h)
gets a chance to prune anything post-fetch — cost grows with total
confirmation history, not with the age window.

This migration adds ``events_tool_confirmed_allow_recent_idx``, keyed on the
single column ``(created_at)`` — with ``session_id`` carried as an
``INCLUDE`` payload column so the sweep's ``SELECT ... session_id`` can be
served as an Index Only Scan without a heap fetch — under the same partial
predicate as 0065 (``kind = 'lifecycle' AND data->>'event' = 'tool_confirmed'
AND data->>'result' = 'allow'``). Without the ``INCLUDE``, both this index
and 0065's ``events_tool_confirmed_allow_idx`` satisfy the partial predicate
equally (neither is a strict cost winner over the tiny fixture the
integration test seeds), so the planner is free to pick either — and on the
composite ``(session_id, tool_call_id)`` index the age bound only re-checks
post-fetch, defeating the point of this migration. The narrower, cheaper
Index Only Scan this index enables makes it the unambiguous planner choice.
After the age-window prune the survivor set is tiny — a range scan over this
index followed by a heap re-check of the partial predicate's own conditions
(already implied by the index, so cheap) is the access path the sweep
detector needs. This is additive: 0065 is left untouched (it still serves
the per-session resolver).

This migration alone would not help if the age clause stayed the
``($N::bigint IS NULL OR created_at >= now() - make_interval(...))`` OR-form
(non-sargable under a generic plan) — the companion code change in
``confirmed_unresolved_predicate`` (``src/aios/db/queries/events.py``) drops
the ``IS NULL`` OR-arm in favor of a plain range comparison so the planner can
actually push the bound into this index.

Built with ``CREATE INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migrations 0065 / 0128.

Revision ID: 0134
Revises: 0133
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0134"
down_revision: str = "0133"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_tool_confirmed_allow_recent_idx "
            "ON events (created_at) INCLUDE (session_id) "
            "WHERE kind = 'lifecycle' "
            "AND data->>'event' = 'tool_confirmed' "
            "AND data->>'result' = 'allow'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_tool_confirmed_allow_recent_idx")
