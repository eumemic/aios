"""Immutable workflow definition history (``workflow_versions``) — Phase 1 of 3.

Gives workflows the version-snapshot history agents already have via
``agent_versions`` (migration ``0003``). Today ``update_workflow`` overwrites the
definition in place — it bumps ``workflows.version`` but keeps no snapshot, so the
old script + surface survive only as inline copies on whatever runs used them and
are lost once those runs are GC'd.

``workflows`` *started* with ``UNIQUE (account_id, name, version)`` (``0064``) — a
version-qualified key that only makes sense with retained version rows — then
``0075`` dropped it to in-place updates for lack of a clean snapshot home. This
restores that intent properly, as a separate table (the way ``agents``/``0003``
did it), not the single-table form ``0064`` fumbled.

This is Phase 1 (additive, self-contained): the version-history table + read API.
It does NOT touch runs (Phase 2) and does NOT remove the run's inline script
(deferred Phase 3).

Pieces, in order:

1. **Parent unique** ``workflows_id_account_id_key UNIQUE (id, account_id)``.
   ``0093`` added this to vaults/sessions/wf_runs but skipped workflows; the
   composite tenant FK below needs it (online-safe — ``id`` is already the PK).

2. **``workflow_versions``** — immutable, copy-on-write, mirror of
   ``agent_versions``. ``name`` IS versioned (a rename mints a new version). The
   ``UNIQUE (workflow_id, version, account_id)`` is the parent unique a Phase-2
   run→version composite FK will reference; the composite tenant FK
   ``(workflow_id, account_id) → workflows(id, account_id)`` is the ``0093``-style
   pattern.

3. **Insert-only trigger** ``BEFORE UPDATE ... RAISE``. Version rows must NEVER
   mutate — unlike ``agent_versions``, which ``0083`` once bulk-``UPDATE``d
   harmlessly. That latitude must not transfer: a future phase makes these rows
   replay-load-bearing, where a single stray UPDATE would terminally error every
   concurrent run of the workflow.

4. **Backfill** — one ``workflow_versions`` row per existing workflow at its
   current version. **Pre-edit history is unrecoverable**: ``update_workflow``
   overwrote definitions in place with no snapshot, so every version *before* a
   workflow's current one is gone for good. We can only snapshot the head that
   survives on the ``workflows`` row; the history a workflow accrued before this
   migration cannot be reconstructed.

Copy-on-write writes from ``create_workflow`` / ``update_workflow`` (the
same-transaction dual-write) land in the application layer, not here.

Revision ID: 0112
Revises: 0111
Create Date: 2026-06-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0112"
down_revision: str = "0111"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Parent unique (prereq, FIRST): the composite tenant FK below references
    #    workflows(id, account_id). Online-safe — ``id`` is already the PK, so the
    #    pair is already unique; this just names a constraint a FK can target.
    op.execute(
        "ALTER TABLE workflows ADD CONSTRAINT workflows_id_account_id_key UNIQUE (id, account_id)"
    )

    # 2. The immutable version-history table (mirror of agent_versions / 0003).
    op.execute(
        """
        CREATE TABLE workflow_versions (
            workflow_id   text    NOT NULL,
            account_id    text    NOT NULL,
            version       integer NOT NULL,
            name          text    NOT NULL,
            script        text    NOT NULL,
            input_schema  jsonb,
            output_schema jsonb,
            description   text,
            tools         jsonb   NOT NULL DEFAULT '[]'::jsonb,
            mcp_servers   jsonb   NOT NULL DEFAULT '[]'::jsonb,
            http_servers  jsonb   NOT NULL DEFAULT '[]'::jsonb,
            created_at    timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (workflow_id, version),
            UNIQUE (workflow_id, version, account_id),
            FOREIGN KEY (workflow_id, account_id)
                REFERENCES workflows(id, account_id) ON DELETE CASCADE
        )
        """
    )

    # 3. Insert-only enforcement: version rows must NEVER mutate. A future phase
    #    makes these rows replay-load-bearing, where a single stray UPDATE would
    #    terminally error every concurrent run of the workflow — so we refuse the
    #    UPDATE at the database, not merely by convention.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION workflow_versions_reject_update()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'workflow_versions rows are immutable (no UPDATE allowed)';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER workflow_versions_no_update
        BEFORE UPDATE ON workflow_versions
        FOR EACH ROW
        EXECUTE FUNCTION workflow_versions_reject_update()
        """
    )

    # 4. Backfill: snapshot every existing workflow as its current version. Only
    #    the surviving head is recoverable — pre-edit history was overwritten in
    #    place and is gone (see the docstring).
    op.execute(
        """
        INSERT INTO workflow_versions (
            workflow_id, account_id, version, name, script,
            input_schema, output_schema, description, tools, mcp_servers, http_servers,
            created_at
        )
        SELECT id, account_id, version, name, script,
               input_schema, output_schema, description, tools, mcp_servers, http_servers,
               created_at
          FROM workflows
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS workflow_versions_no_update ON workflow_versions")
    op.execute("DROP FUNCTION IF EXISTS workflow_versions_reject_update()")
    op.execute("DROP TABLE IF EXISTS workflow_versions")
    op.execute("ALTER TABLE workflows DROP CONSTRAINT IF EXISTS workflows_id_account_id_key")
