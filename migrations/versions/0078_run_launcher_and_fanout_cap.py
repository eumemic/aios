"""wf_runs.launcher_session_id + the fan-out cap's count index.

Persists WHO launched a run (the agent session that called the ``create_run``
builtin; NULL = the operator/HTTP path), enabling the horizontal fan-out cap:
a per-launcher-session and per-account ceiling on OUTSTANDING (non-terminal)
runs, counted under a per-account advisory lock at launch time. The vertical
depth cap (parent_run_id chain) bounds nesting; this bounds sibling fan-out —
without it a model in a loop could launch unbounded root runs.

``ON DELETE SET NULL``: session rows are hard-deleted via the operator API, so
a bare FK would break session deletion; a deleted launcher reclassifies its
live runs as operator-launched (account cap only) — an operator-only surface.

Backfill is NULL for ALL existing rows, including past agent-launched runs
(the launcher was never persisted before this migration). Those rows escape
the per-launcher count until they reach a terminal status — a one-time grace
window, bounded by the account cap like any operator launch.

The partial index serves both cap counts (account via prefix); its status
list must stay verbatim-identical to ``count_active_runs`` and the
``wf_runs_active_idx`` predicate (0064) so the planner can use it.
"""

from __future__ import annotations

from alembic import op

revision: str = "0078"
down_revision: str = "0077"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN launcher_session_id text "
        "REFERENCES sessions(id) ON DELETE SET NULL"
    )
    op.execute(r"""
        CREATE INDEX wf_runs_launcher_active_idx
            ON wf_runs (account_id, launcher_session_id)
            WHERE archived_at IS NULL AND status IN ('pending','running','suspended')
    """)


def downgrade() -> None:
    op.execute("DROP INDEX wf_runs_launcher_active_idx")
    op.execute("ALTER TABLE wf_runs DROP COLUMN launcher_session_id")
