"""Partial index on the lifecycle arm of read_windowed_context_events.

``read_windowed_context_events`` (``src/aios/db/queries/events.py``) reads
the context slate as a ``UNION ALL`` of a message arm and a lifecycle arm
(``kind='lifecycle' AND data->>'event' = ANY(...)``, the
``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist). The message arm is served by
the partial index ``events_session_cumtokens_idx`` (``WHERE kind='message'``,
migration 0012). The lifecycle arm has no supporting partial index:
``events_turn_error_idx`` (migration 0062) is
``WHERE kind='lifecycle' AND data->>'stop_reason'='error'`` — a narrower
predicate the planner cannot use here — so the lifecycle arm falls back to
the non-partial ``events_session_seq_idx (session_id, seq)`` and
heap-filters ``kind='lifecycle'`` across the whole session. In the
``drop=None`` branch there is no seq lower bound, so it examines the whole
session, and lifecycle rows include per-inference ``request_opened`` /
``request_response`` events, so examined rows scale O(session-size).

This adds ``events_session_lifecycle_seq_idx`` on
``(session_id, seq) WHERE kind = 'lifecycle'`` so the planner can serve the
lifecycle arm with an index scan restricted to lifecycle rows only, for both
the ``drop=None`` branch (session prefix + ``ORDER BY seq``) and the
``drop=N`` branch's seq range. The index is intentionally generic
(``WHERE kind = 'lifecycle'`` only, no ``data->>'event'`` predicate) so it
keeps matching the query even as ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` grows.

Built with ``CREATE INDEX CONCURRENTLY`` (outside a transaction via
``autocommit_block``) so it never takes an ACCESS EXCLUSIVE lock on the
live-written ``events`` table — same pattern as migrations 0023 / 0062.

Revision ID: 0135
Revises: 0134
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0135"
down_revision: str = "0134"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "events_session_lifecycle_seq_idx "
            "ON events (session_id, seq) "
            "WHERE kind = 'lifecycle'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS events_session_lifecycle_seq_idx")
