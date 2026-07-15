"""Persist workflow workspace mode and live bind pointer.

Revision ID: 0151
Revises: 0149
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0151"
down_revision: str = "0149"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ADD COLUMN workspace_mode text NOT NULL DEFAULT 'fresh'")
    op.execute("ALTER TABLE wf_runs ADD COLUMN workspace_path text")
    op.execute("ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_workspace_mode_check CHECK (workspace_mode IN ('shared', 'fresh'))")

def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT wf_runs_workspace_mode_check")
    op.execute("ALTER TABLE wf_runs DROP COLUMN workspace_path")
    op.execute("ALTER TABLE wf_runs DROP COLUMN workspace_mode")
