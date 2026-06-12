"""Widen ``wf_run_signals.kind`` CHECK to include ``sandbox_result``.

The workflow ``sandbox()`` capability (#988) runs a shell command in a
run-scoped sandbox container via a fire-and-forget worker task, which writes a
``sandbox_result`` ``wf_run_signals`` row on completion (harvested into a
``call_result`` by the next step, keeping ``wf_run_events`` single-writer —
mirrors the ``tool_result`` path from 0074). The inline CHECK from 0064 (widened
to ``cancel`` in 0072 and ``tool_result`` in 0074) enumerated only the prior
kinds, so it must be widened again or every ``sandbox_result`` insert raises.

Drop + re-add of the inline CHECK (PG-default name ``wf_run_signals_kind_check``
from 0064), exactly as 0074 widened it to ``tool_result``. Additive and
backward-compatible: existing rows are unaffected, and pre-deploy code never
writes the new kind — so deploy ordering is unconstrained.

Revision ID: 0089
Revises: 0088
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0089"
down_revision: str = "0088"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done','cancel','tool_result','sandbox_result'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_run_signals DROP CONSTRAINT wf_run_signals_kind_check")
    op.execute(
        "ALTER TABLE wf_run_signals ADD CONSTRAINT wf_run_signals_kind_check "
        "CHECK (kind IN ('gate_resume','child_done','cancel','tool_result'))"
    )
