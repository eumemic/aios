"""Record the nearest budgeted ancestor of a sub-run (#2476 F2).

Revision ID: 0192
Revises: 0191

A run with ``budget_usd`` caps spend over its whole creation subtree, but before this
only that run's own ``agent()``/``call_llm`` calls were refused once it was spent, so a
sub-run with no budget of its own could keep spending. ``budget_run_id`` names the
nearest ancestor run with a budget; a sub-run with no budget of its own is held to that
run's ceiling.

NULL on root runs, on sub-runs with no budgeted ancestor, and on every row written
before this revision or by an application image from before it; those keep the old
behaviour (no inherited ceiling). The column add is catalog-only. No FK: the archive
prune deletes run rows, and a descendant whose budget run is gone is treated as having
spent its budget rather than as unbounded.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0192"
down_revision: str | None = "0191"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE wf_runs ADD COLUMN budget_run_id text")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN budget_run_id")
