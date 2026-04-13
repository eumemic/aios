"""Session-level token usage tracking.

Add four bigint columns to the sessions table for cumulative token usage
across all model calls in the session. These are atomically incremented
after each inference step, so ``GET /sessions/:id`` returns totals
without an aggregation query.

Per-call token breakdowns live in ``span.model_request_end`` events in
the event log.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE sessions ADD COLUMN input_tokens bigint NOT NULL DEFAULT 0;"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN output_tokens bigint NOT NULL DEFAULT 0;"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN cache_read_input_tokens bigint NOT NULL DEFAULT 0;"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN cache_creation_input_tokens bigint NOT NULL DEFAULT 0;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS cache_creation_input_tokens;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS cache_read_input_tokens;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS output_tokens;")
    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS input_tokens;")
