"""Drop the retired ``cancel`` builtin tool name from persisted ``tools`` JSONB.

#1458 deleted the model-facing ``cancel`` tool (the local in-flight tool-task detach);
``stop_task`` is the sole model cancel verb (the detach was unreachable while the
inference gate held the model, and redundant with per-tool timeouts / ``stop_task`` /
the #752 abandoned-client reaper). ``cancel`` was removed from the ``BuiltinToolType``
Literal, so any row whose ``tools`` JSONB still carries a ``{"type": "cancel"}`` element
would fail ``ToolSpec`` validation on read — and ``to_openai_tools``' ``registry.get``
raises on the now-unregistered name, 500-ing every affected agent.

This migration removes the ``cancel`` element from every agent/workflow ``ToolSpec``
surface — ``agents``, ``agent_versions``, ``workflows``, ``workflow_versions``,
``wf_runs`` (the launch-pinned surface), and ``sessions`` (the frozen child surface,
nullable). Order-preserving; a list that was only ``[cancel]`` collapses to ``[]``.

``downgrade()`` is a no-op: the tool is gone, so there is nothing to restore.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0120"
down_revision: str = "0119"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables with a NOT-NULL ``tools`` JSONB surface column.
_TOOLS_TABLES = ("agents", "agent_versions", "workflows", "workflow_versions", "wf_runs")


def upgrade() -> None:
    # Drop every ``{"type": "cancel"}`` element from a tools array, order-preserving.
    op.execute(r"""
        CREATE FUNCTION _aios_drop_cancel_tool(tools jsonb) RETURNS jsonb
        LANGUAGE sql IMMUTABLE AS $fn$
            SELECT coalesce(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
            FROM jsonb_array_elements(tools) WITH ORDINALITY AS arr(elem, ord)
            WHERE elem->>'type' <> 'cancel'
        $fn$
    """)

    for table in _TOOLS_TABLES:
        op.execute(f"""
            UPDATE {table} SET tools = _aios_drop_cancel_tool(tools)
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements(tools) e WHERE e->>'type' = 'cancel'
            )
        """)

    # sessions.tools is nullable (only frozen child sessions carry a surface).
    op.execute("""
        UPDATE sessions SET tools = _aios_drop_cancel_tool(tools)
        WHERE tools IS NOT NULL AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(tools) e WHERE e->>'type' = 'cancel'
        )
    """)

    op.execute("DROP FUNCTION _aios_drop_cancel_tool(jsonb)")


def downgrade() -> None:
    # Forward-only: the cancel tool is deleted, nothing to restore.
    pass
