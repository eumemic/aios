"""Drop deprecated MCP channel_context config.

The runtime injects focal-channel metadata into normal MCP calls based on
session state, not agent toolset config. Strip the old inert field from stored
agent JSON before the Pydantic model stops accepting it.

Revision ID: 0027
Revises: 0026
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0027"
down_revision: str = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _strip_channel_context_sql(table: str) -> str:
    return f"""
        UPDATE {table}
           SET tools = (
               SELECT COALESCE(
                   jsonb_agg(
                       CASE
                           WHEN jsonb_typeof(tool) = 'object'
                           THEN tool - 'channel_context'
                           ELSE tool
                       END
                       ORDER BY ord
                   ),
                   '[]'::jsonb
               )
                 FROM jsonb_array_elements(COALESCE(tools, '[]'::jsonb))
                      WITH ORDINALITY AS t(tool, ord)
           )
    """


def upgrade() -> None:
    op.execute(_strip_channel_context_sql("agents"))
    op.execute(_strip_channel_context_sql("agent_versions"))


def downgrade() -> None:
    # The removed JSON field was inert and cannot be reconstructed.
    pass
