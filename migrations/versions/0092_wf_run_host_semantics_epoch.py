"""Pin workflow runs to a host-semantics epoch.

Revision ID: 0092
Revises: 0091
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0092"
down_revision: str = "0091"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


HOST_SEMANTICS_EPOCH = 1


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN host_semantics_epoch integer "
        f"NOT NULL DEFAULT {HOST_SEMANTICS_EPOCH}"
    )
    op.execute("ALTER TABLE wf_runs ALTER COLUMN host_semantics_epoch DROP DEFAULT")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN host_semantics_epoch")
