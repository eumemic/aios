"""Drop the retired ``complete_goal``/``fail_goal`` builtin tool names from persisted ``tools`` JSONB.

#1525 (unify-obligations #2, commit 1c6e6d09) removed ``complete_goal``/``fail_goal`` from the
``BuiltinToolType`` Literal + the tool registry, but shipped **no read-tolerance shim and no data
migration**. Long-lived agents whose persisted ``agents.tools`` (and ``agent_versions.tools``)
JSONB still listed those builtins fail ``ToolSpec.model_validate`` on read — a pre-context-build
throw that wedged the affected agent (the live kedalion-ultron agent) into an infinite
reschedule. ``return``/``error`` are general step verbs, not model-listed builtins, so the retired
tools have NO canonical successor → REMOVE, do not remap.

This migration removes every ``{"type": "complete_goal"}`` / ``{"type": "fail_goal"}`` element
from every agent/workflow ``ToolSpec`` surface — ``agents``, ``agent_versions``, ``workflows``,
``workflow_versions``, ``wf_runs`` (the launch-pinned surface), and ``sessions`` (the frozen
child surface, nullable). Order-preserving; a list that was only ``[complete_goal, fail_goal]``
collapses to ``[]``; clean rows are untouched. The model code carries a matching read-tolerance
shim (``_RETIRED_BUILTINS`` + ``load_tool_specs`` in ``models/agents.py``) for the
post-deploy/pre-migrate window and for any future respawn from an unmigrated row; both can be
removed once this has run everywhere.

This is the same "remove a builtin from BuiltinToolType + registry → ship the shim-and-migration
pair" discipline that #1419 (migration 0116) / #1428 (migration 0117) / #1458 (migration 0120)
established; #1525 omitted both halves and this closes the gap. Part of #1516 / #1562.

``downgrade()`` is a no-op: the tools are gone, so there is nothing to restore.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0122"
down_revision: str = "0121"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables with a NOT-NULL ``tools`` JSONB surface column.
_TOOLS_TABLES = ("agents", "agent_versions", "workflows", "workflow_versions", "wf_runs")
# Retired builtin tool ``type`` values (no canonical successor → dropped, not remapped).
_RETIRED_NAMES_SQL = "'complete_goal', 'fail_goal'"


def upgrade() -> None:
    # Drop every retired-builtin element from a tools array, order-preserving.
    op.execute(rf"""
        CREATE FUNCTION _aios_drop_retired_goal_tools(tools jsonb) RETURNS jsonb
        LANGUAGE sql IMMUTABLE AS $fn$
            SELECT coalesce(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
            FROM jsonb_array_elements(tools) WITH ORDINALITY AS arr(elem, ord)
            WHERE elem->>'type' NOT IN ({_RETIRED_NAMES_SQL})
        $fn$
    """)

    for table in _TOOLS_TABLES:
        op.execute(f"""
            UPDATE {table} SET tools = _aios_drop_retired_goal_tools(tools)
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements(tools) e
                WHERE e->>'type' IN ({_RETIRED_NAMES_SQL})
            )
        """)

    # sessions.tools is nullable (only frozen child sessions carry a surface).
    op.execute(f"""
        UPDATE sessions SET tools = _aios_drop_retired_goal_tools(tools)
        WHERE tools IS NOT NULL AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(tools) e
            WHERE e->>'type' IN ({_RETIRED_NAMES_SQL})
        )
    """)

    op.execute("DROP FUNCTION _aios_drop_retired_goal_tools(jsonb)")


def downgrade() -> None:
    # Forward-only: the complete_goal/fail_goal tools are retired, nothing to restore.
    pass
