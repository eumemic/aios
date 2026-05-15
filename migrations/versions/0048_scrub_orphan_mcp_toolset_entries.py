"""Strip orphan ``mcp_toolset`` tool entries that reference removed servers.

Pre-#301 agents (created before the MCP-out cutover in PR 5 of #328) still
carry an ``mcp_toolset`` entry in ``tools`` for a server name that has
since been removed from the same row's ``mcp_servers`` list. Runtime
already ignores these — ``to_openai_tools`` skips entries whose
``mcp_server_name`` doesn't match any active server — but they inflate
operator-facing tool counts and confuse reads of the resource. See #345.

This migration walks ``agents.tools`` and ``agent_versions.tools`` and
removes any element where:

* ``type = 'mcp_toolset'``
* AND ``mcp_server_name`` is not present in the same row's
  ``mcp_servers[*].name`` array.

Only the ``tools`` column is rewritten; ``mcp_servers`` is the source of
truth and untouched. Idempotent — re-running is a no-op once the orphans
are gone.

Revision ID: 0048
Revises: 0047
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0048"
down_revision: str = "0047"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (table_name,) — both tables share the same column shape (`tools` jsonb
# array, `mcp_servers` jsonb array) so the same SQL template covers both.
_TABLES: Sequence[str] = ("agents", "agent_versions")


def _scrub_sql(table: str) -> str:
    # Inline a CTE that names the active server set as a jsonb array of
    # strings, then filter the ``tools`` array's elements to those that
    # are NOT orphan ``mcp_toolset`` entries. ``jsonb_path_query_array``
    # would be cleaner on PG 17+; this form works on PG 14+ (our floor).
    return f"""
    UPDATE {table} a SET tools = COALESCE((
        SELECT jsonb_agg(t)
          FROM jsonb_array_elements(a.tools) t
         WHERE NOT (
            t->>'type' = 'mcp_toolset'
            AND NOT EXISTS (
                SELECT 1
                  FROM jsonb_array_elements(a.mcp_servers) s
                 WHERE s->>'name' = t->>'mcp_server_name'
            )
         )
    ), '[]'::jsonb)
    WHERE EXISTS (
        SELECT 1
          FROM jsonb_array_elements(a.tools) t
         WHERE t->>'type' = 'mcp_toolset'
           AND NOT EXISTS (
                SELECT 1
                  FROM jsonb_array_elements(a.mcp_servers) s
                 WHERE s->>'name' = t->>'mcp_server_name'
           )
    )
    """


def upgrade() -> None:
    for table in _TABLES:
        op.execute(_scrub_sql(table))


def downgrade() -> None:
    # Re-introducing orphans intentionally would require knowing which
    # server names had been removed, which is information this migration
    # discards. The cleanup is forward-only.
    pass
