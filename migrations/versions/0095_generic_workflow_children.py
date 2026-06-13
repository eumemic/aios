"""Generic workflow children and default child model.

Revision ID: 0095
Revises: 0094
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0095"
down_revision: str = "0094"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions ALTER COLUMN agent_id DROP NOT NULL")
    op.execute("ALTER TABLE sessions ALTER COLUMN agent_version DROP NOT NULL")
    op.execute("ALTER TABLE sessions ADD COLUMN model text")
    op.execute("ALTER TABLE wf_runs ADD COLUMN default_child_model text")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_agent_version_pair_ck "
        "CHECK (parent_run_id IS NULL OR ((agent_id IS NULL) = (agent_version IS NULL)))"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_agentless_workflow_child_ck "
        "CHECK (agent_id IS NOT NULL OR (parent_run_id IS NOT NULL AND model IS NOT NULL))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_agentless_workflow_child_ck")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_agent_version_pair_ck")
    op.execute("ALTER TABLE wf_runs DROP COLUMN default_child_model")
    op.execute("ALTER TABLE sessions DROP COLUMN model")
    op.execute("ALTER TABLE sessions ALTER COLUMN agent_version SET NOT NULL")
    op.execute("ALTER TABLE sessions ALTER COLUMN agent_id SET NOT NULL")
