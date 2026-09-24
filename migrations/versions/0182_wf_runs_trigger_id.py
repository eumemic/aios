"""Record the launching trigger on a workflow run (#2446 d).

Revision ID: 0182
Revises: 0181

``wf_runs.trigger_id`` is the durable trigger→run link the opt-in
``WorkflowAction.max_outstanding_runs`` cap counts against. It is written in the
SAME transaction as the run INSERT (under the per-account fan-out advisory
lock), which is what makes the per-trigger count race-safe;
``trigger_runs.result_id`` cannot serve, because it is written only after
``create_run`` commits.

Deliberately NO foreign key: a one-shot trigger's row is deleted BEFORE its
action launches the run, and run history outlives trigger deletion.

The partial index mirrors ``wf_runs_launcher_active_idx`` (0078): only live,
non-terminal, trigger-launched rows, which is exactly the cap's COUNT
predicate. The column add is metadata-only (nullable, no default); the index
builds concurrently so the migration stays online on a populated ``wf_runs``.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0182"
down_revision: str | None = "0181"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "wf_runs_trigger_active_idx"
INDEX_DEFINITION = (
    "ON wf_runs (trigger_id) "
    "WHERE trigger_id IS NOT NULL AND archived_at IS NULL "
    "AND status IN ('pending','running','suspended')"
)


def upgrade() -> None:
    # The column add is metadata-only but still takes ACCESS EXCLUSIVE on
    # wf_runs. Behind a long transaction it would queue that lock and block
    # every run INSERT behind it; bound the wait (0169's mechanism) so the
    # deploy fails fast and rolls back instead. SET LOCAL ends at the commit
    # that opens the autocommit block, so the concurrent index build is not
    # bounded by it.
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE wf_runs ADD COLUMN IF NOT EXISTS trigger_id text")
    with op.get_context().autocommit_block():
        # An interrupted concurrent build can leave an invalid same-named index.
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(f"CREATE INDEX CONCURRENTLY {INDEX_NAME} {INDEX_DEFINITION}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS trigger_id")
