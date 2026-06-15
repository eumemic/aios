"""Add the run's inbound request-edge carriers — ``request_id`` + ``caller``
(+ the request's per-call ``request_output_schema``) on ``wf_runs`` (#1126).

The ``invoke_workflow`` run-caller (#1129) spawns a sub-run *in service of a
request*: the sub-run must record which request it answers so that, at its
terminal ``_complete_run`` chokepoint, it can emit the symmetric
``request_response`` keyed on that ``request_id`` (the "one missing edge"). The
caller-kind-agnostic ``caller`` ({kind:'run'|'session'|'api', id}) mirrors the
session-side ``request_opened`` provenance; ``request_output_schema`` is the
JSON Schema the request demands of the sub-run's terminal output, validated
fail-loud at completion (``output_schema_violation``).

Purely additive, nullable columns: an ordinary operator/HTTP run leaves all
three NULL (edgeless — it answers no request, so the emitter no-ops). Safe in
the post-deploy new-code/old-schema window: the new code's reads tolerate the
old schema (a run created before this migration simply has no request edge),
and the columns are invisible to the running container until the migration
completes.

Revision ID: 0101
Revises: 0100
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0101"
down_revision: str = "0100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE wf_runs ADD COLUMN request_id text")
    op.execute("ALTER TABLE wf_runs ADD COLUMN caller jsonb")
    op.execute("ALTER TABLE wf_runs ADD COLUMN request_output_schema jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP COLUMN request_output_schema")
    op.execute("ALTER TABLE wf_runs DROP COLUMN caller")
    op.execute("ALTER TABLE wf_runs DROP COLUMN request_id")
