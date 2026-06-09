"""Workflow runs as credentialed principals: ``wf_run_vaults`` + a declared surface.

Two additive changes that let a workflow declare a tool surface like an agent and a
run bind vaults like a session — the substrate for credentialed tool calls from a run:

1. ``wf_run_vaults`` — the run↔vault junction, mirroring ``session_vaults`` (migration
   0005) with rank-based first-match resolution, but with ``account_id`` baked in from
   the start (``session_vaults`` only gained it retroactively in 0043/0044) and
   ``run_id`` cascading from ``wf_runs``.
2. ``workflows.{tools, mcp_servers, http_servers}`` — three ``jsonb`` columns mirroring
   the agent envelope (agents migration 0052), each defaulting ``'[]'::jsonb`` so
   existing rows Just Work. They declare the run's reachable tool surface; enforcement
   lands with the ``tool()`` dispatch in a later slice.

Both tables are small workflow-runtime tables (not the hot ``events`` table), so plain
in-transaction DDL is fine — no ``CONCURRENTLY`` needed.

Revision ID: 0073
Revises: 0072
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0073"
down_revision: str = "0072"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE wf_run_vaults (
            run_id     text NOT NULL REFERENCES wf_runs(id) ON DELETE CASCADE,
            vault_id   text NOT NULL REFERENCES vaults(id),
            rank       integer NOT NULL,
            account_id text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            PRIMARY KEY (run_id, vault_id)
        )
    """)
    op.execute("ALTER TABLE workflows ADD COLUMN tools jsonb NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE workflows ADD COLUMN mcp_servers jsonb NOT NULL DEFAULT '[]'::jsonb")
    op.execute("ALTER TABLE workflows ADD COLUMN http_servers jsonb NOT NULL DEFAULT '[]'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS http_servers")
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS mcp_servers")
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS tools")
    op.execute("DROP TABLE IF EXISTS wf_run_vaults")
