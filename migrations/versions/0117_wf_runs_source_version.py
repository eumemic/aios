"""Bind each run to the workflow version it ran — ``wf_runs.source_version`` +
composite FK to ``workflow_versions`` — Phase 2 of 3.

Phase 1 (``0112``) gave workflows an immutable ``workflow_versions`` history.
This phase records, on every run, WHICH version it snapshotted, and binds that
pointer with a **strict composite (tenant-scoped) FK** so a run's bound source
definition always resolves — no dangling audit pointer.

Pieces, in order:

1. **``wf_runs.source_version``** — ``integer``, **nullable**. New runs set it to
   the version they snapshotted (in ``insert_wf_run`` / ``create_run``). Legacy
   rows stay NULL; a best-effort backfill (below) fills the unambiguous ones.

2. **Composite FK** ``(workflow_id, source_version, account_id) →
   workflow_versions (workflow_id, version, account_id)``, added ``NOT VALID``
   (the ``0093`` pattern). ``MATCH SIMPLE`` (the default) exempts rows where ANY
   FK column is NULL — so legacy / unbackfillable rows (``source_version IS
   NULL``) are not checked, while every new run's non-NULL pointer must resolve.

3. **Best-effort historical backfill** — set ``source_version`` on existing runs
   where the run's ``script_sha`` matches EXACTLY ONE kept version row's script
   (``HAVING count(*) = 1``). A rename-only edit bumps the version with a
   byte-identical script, so a sha can match >1 version; in that case we leave
   NULL (NULL-on-ambiguity). Purely audit — runs still exec their inline script.

4. **VALIDATE** the constraint. The backfill only ever sets pointers that
   resolve by construction (it joins ``workflow_versions``), and new writes go
   through the FK, so the validating full-scan succeeds. ``VALIDATE`` takes
   ``SHARE UPDATE EXCLUSIVE`` (concurrent reads/writes proceed); ``wf_runs`` has
   no row GC so the scan is bounded. Inlined here for a single forward-migration;
   a hot prod system would split ``VALIDATE`` to an off-peak window.

The run STILL snapshots ``script`` + ``script_sha`` inline and execs its own
copy — unchanged. Reading the script *through* the FK (and dropping
``wf_runs.script``) is the deferred Phase 3.

Revision ID: 0117
Revises: 0116
Create Date: 2026-06-21
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0117"
down_revision: str = "0116"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. The nullable selector column. Legacy rows stay NULL (the FK's MATCH
    #    SIMPLE exempts them); new runs set it to the snapshotted version.
    op.execute("ALTER TABLE wf_runs ADD COLUMN source_version integer")

    # 2. Composite tenant FK, NOT VALID (the 0093 pattern). Parent unique is
    #    ``workflow_versions (workflow_id, version, account_id)`` from 0112.
    op.execute(
        """
        ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_source_version_fkey
            FOREIGN KEY (workflow_id, source_version, account_id)
            REFERENCES workflow_versions (workflow_id, version, account_id) NOT VALID
        """
    )

    # 3. Best-effort historical backfill: set source_version only where the run's
    #    script_sha matches EXACTLY ONE kept version's script (NULL-on-ambiguity —
    #    a rename-only bump yields a byte-identical script across versions). The
    #    sha is computed in-SQL (sha256 over the version's script bytes) so the
    #    match needs no application code. Account- + workflow-scoped join keeps it
    #    a true composite match (never a cross-tenant sha collision).
    op.execute(
        """
        WITH unambiguous AS (
            SELECT workflow_id, account_id,
                   encode(sha256(convert_to(script, 'UTF8')), 'hex') AS sha,
                   min(version) AS version
              FROM workflow_versions
             GROUP BY workflow_id, account_id, encode(sha256(convert_to(script, 'UTF8')), 'hex')
            HAVING count(*) = 1
        )
        UPDATE wf_runs r
           SET source_version = u.version
          FROM unambiguous u
         WHERE r.workflow_id = u.workflow_id
           AND r.account_id = u.account_id
           AND r.script_sha = u.sha
           AND r.source_version IS NULL
        """
    )

    # 4. VALIDATE: full-scan wf_runs to promote the FK to fully-enforced. Every
    #    non-NULL source_version was either just backfilled from a resolving join
    #    or (for rows written after this migration) inserted through the FK, so
    #    the scan succeeds.
    op.execute("ALTER TABLE wf_runs VALIDATE CONSTRAINT wf_runs_source_version_fkey")


def downgrade() -> None:
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT IF EXISTS wf_runs_source_version_fkey")
    op.execute("ALTER TABLE wf_runs DROP COLUMN IF EXISTS source_version")
