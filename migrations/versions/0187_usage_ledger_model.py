"""Record which model each inference charge paid for (#2472 D3).

Revision ID: 0187
Revises: 0186

``inference_usage_ledger`` rows carry a session's or a run's usage, but not the
model, so a parent run reading its sub-runs' facts couldn't say which models they
called or what each cost. The eval gate (#2476) needs both: it prices arms per
model and refuses a judge whose model family overlaps an arm's.

Nullable: rows written before this revision, or by an application image from
before it, have no model. The column add is catalog-only.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0187"
down_revision: str | None = "0186"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE inference_usage_ledger ADD COLUMN model text")


def downgrade() -> None:
    op.execute("ALTER TABLE inference_usage_ledger DROP COLUMN model")
