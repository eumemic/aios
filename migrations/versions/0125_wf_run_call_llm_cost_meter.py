"""Run-level ``call_llm`` inference-cost meter on ``wf_runs``.

Part of the **Workflows-as-Models** epic (#1633). Adds the per-run inference
ledger the ``call_llm()`` capability charges and the over-budget gate reads.

**Why a new column, not the child-session rollup.** Until now a run's only
spend was its child sessions' rows, summed through ``sessions.parent_run_id``
(``run_children_usage``). ``call_llm`` runs raw inference **on the worker, at
the run's own inference site** — there is no child session row to carry its
cost. So the run needs a ledger of its own: ``call_llm_cost_microusd``, a
non-null ``bigint`` defaulting to ``0`` that the ``call_llm`` resolver
increments by LiteLLM's per-request USD figure (in micro-USD) **once at the
inference site**, and that the ``budget_usd`` gate adds to the child-session
rollup before comparing against the ceiling.

**Default 0, non-null.** Every existing run (and every run in a raw snapshot)
reads ``0`` — no ``call_llm`` spend yet — so the rollup is unchanged for runs
that never call it. New increments are an atomic ``UPDATE … SET col = col +
$delta`` under the run lock the step already holds, so concurrent child-session
spend and ``call_llm`` spend never race on the same row.

``downgrade()`` drops the column.

Revision ID: 0125
Revises: 0124
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0125"
down_revision: str = "0124"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN call_llm_cost_microusd bigint NOT NULL DEFAULT 0 "
        "CHECK (call_llm_cost_microusd >= 0)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN call_llm_cost_microusd")
