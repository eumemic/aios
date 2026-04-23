"""Partial index for the per-model token-count correction lookup.

Issue #160. ``model_token_ratio`` aggregates the most recent N successful
``model_request_end`` spans for a given model. This partial index lets that
subquery run as an index-only scan keyed on ``(data->>'model', seq DESC)``,
filtered to rows that actually carry the calibration fields stamped by the
new ``harness/loop.py`` code.

Historical spans emitted before this change lack ``local_tokens`` / ``model``
keys and are auto-excluded by the predicate — no backfill needed.

Built with ``CREATE INDEX CONCURRENTLY`` to avoid an ACCESS EXCLUSIVE lock on
the live ``events`` table; alembic's implicit BEGIN is disabled via
``autocommit_block()`` so CONCURRENTLY is allowed.

Revision ID: 0024
Revises: 0023
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0024"
down_revision: str = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY events_model_request_end_calibration_idx "
            "ON events ((data->>'model'), seq DESC) "
            "WHERE kind = 'span' "
            "AND data->>'event' = 'model_request_end' "
            "AND (data->>'is_error')::boolean = false "
            "AND data ? 'local_tokens' "
            "AND data ? 'model'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "events_model_request_end_calibration_idx"
        )
