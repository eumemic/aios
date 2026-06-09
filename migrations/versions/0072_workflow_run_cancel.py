"""Workflows: a ``cancelled`` terminal status + a ``cancel`` resume-signal kind.

User-initiated cancellation (``POST /v1/runs/{id}/cancel``). The request handler
must not write the journal directly (it is single-writer, gapless-seq, under the
run's procrastinate lock), so cancel rides the existing side-marker path: it records
a ``cancel`` ``wf_run_signals`` row and wakes the run; the next ``run_workflow_step``
harvests it and finalizes the run as ``cancelled`` — a distinct terminal status so
the UI can tell a user-cancel from an error.

Two CHECK constraints widen:
- ``wf_runs.status``        gains ``'cancelled'`` (a new terminal value). The
  active-sweep partial index lists only the non-terminal statuses, so ``cancelled``
  is naturally excluded — no index change.
- ``wf_run_signals.kind``   gains ``'cancel'`` (the side-marker kind).

Both tables are small workflow-runtime tables (not the hot ``events`` table), so a
plain drop+re-add of the anonymous inline CHECK constraints (PG-default names
``wf_runs_status_check`` / ``wf_run_signals_kind_check`` from migration 0064) is
fine — brief ACCESS EXCLUSIVE + a validation scan, no ``CONCURRENTLY`` needed.

Revision ID: 0072
Revises: 0071
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0072"
down_revision: str = "0071"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT wf_runs_status_check")
    op.execute(
        "ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_status_check "
        "CHECK (status IN ('pending','running','suspended','completed','errored','cancelled'))"
    )
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done','cancel'))"
    )


def downgrade() -> None:
    # Reverting assumes no rows use the widened values (a cancelled run / cancel
    # signal would fail the narrower CHECK) — acceptable for a down-migration.
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done'))"
    )
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT wf_runs_status_check")
    op.execute(
        "ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_status_check "
        "CHECK (status IN ('pending','running','suspended','completed','errored'))"
    )
