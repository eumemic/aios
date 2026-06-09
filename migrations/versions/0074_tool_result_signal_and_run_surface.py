"""Workflow `tool()` capability: a `tool_result` signal kind + a run's snapshotted surface.

Two additions for slice 2 (runs calling their declared network/credential tools):

1. ``wf_run_signals.kind`` gains ``'tool_result'`` — the side-marker a fire-and-forget
   worker tool-task writes on completion (harvested into a ``call_result`` by the next
   step, keeping ``wf_run_events`` single-writer; mirrors the ``gate_resume`` path). Plain
   drop+re-add of the inline CHECK (PG-default name ``wf_run_signals_kind_check`` from
   0064), as 0072 did for ``cancel``.

2. ``wf_runs`` gains ``tools``/``mcp_servers``/``http_servers`` jsonb columns — the run's
   **snapshot** of its workflow's declared surface, copied at ``create_run`` alongside
   ``script``/``script_sha``. Pinning the surface at launch (rather than reading a live
   ``get_workflow``) keeps the run deterministic and its tool-authority fixed even when the
   workflow is later updated. Mirrors the ``workflows`` columns from 0073.

Both are small workflow-runtime tables (not the hot ``events`` table), so plain
in-transaction DDL is fine.

Revision ID: 0074
Revises: 0073
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0074"
down_revision: str = "0073"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done','cancel','tool_result'))"
    )
    op.execute("ALTER TABLE wf_runs ADD COLUMN tools jsonb NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN mcp_servers jsonb NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN http_servers jsonb NOT NULL DEFAULT '[]'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS http_servers")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS mcp_servers")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS tools")
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done','cancel'))"
    )
