"""Normalize the retired ``cancel_run`` builtin tool name in persisted ``tools`` JSONB.

#1428 superseded the ``cancel_run`` MODEL tool with ``stop_task`` (which cancels any awaited
``call_*`` task by its ``tool_call_id`` — a session servicer or a run — not just a run by id).
The ``wf_service.cancel_run`` SERVICE is unaffected; only the model-facing tool name changed.
``cancel_run`` is validated against the ``BuiltinToolType`` Literal on read, so any row whose
``tools`` JSONB still carries it would fail validation post-deploy — and worse, ``to_openai_tools``'
``registry.get`` raises on an unregistered name, 500-ing every affected agent in the window.

This migration rewrites ``cancel_run`` → ``stop_task`` in every table that stores an
agent/workflow ``ToolSpec`` surface — ``agents``, ``agent_versions``, ``workflows``,
``workflow_versions``, ``wf_runs`` (the launch-pinned surface), and ``sessions`` (the frozen
child surface, nullable). The rename is 1:1, so no collapse occurs; the ``DISTINCT ON`` dedup
shape (carried over from 0116) only matters in the unlikely case a row already carried both
``cancel_run`` and ``stop_task`` (it keeps the first, preserving its overrides). The model code
carries a matching read-tolerance shim (``_LEGACY_BUILTIN_RENAMES`` in ``models/agents.py``) for
the post-deploy/pre-migrate window; that entry can be removed once this has run everywhere.

``downgrade()`` is a no-op: the rename is forward-only (``stop_task`` is a strict superset of
``cancel_run``'s capability, so there is nothing meaningful to reverse to).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0117"
down_revision: str = "0116"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables with a NOT-NULL ``tools`` JSONB surface column.
_TOOLS_TABLES = ("agents", "agent_versions", "workflows", "workflow_versions", "wf_runs")
_LEGACY_NAMES_SQL = "'cancel_run'"


def upgrade() -> None:
    # Map + dedupe a tools array: rename the legacy builtin name, then keep one entry per
    # resulting builtin name (custom/mcp_toolset never deduped), original order preserved.
    op.execute(r"""
        CREATE FUNCTION _aios_normalize_legacy_tools(tools jsonb) RETURNS jsonb
        LANGUAGE sql IMMUTABLE AS $fn$
            SELECT coalesce(jsonb_agg(elem ORDER BY ord), '[]'::jsonb)
            FROM (
                SELECT DISTINCT ON (dedup_key) elem, ord
                FROM (
                    SELECT
                        jsonb_set(e, '{type}', to_jsonb(new_type)) AS elem,
                        ord,
                        CASE WHEN new_type IN ('custom', 'mcp_toolset')
                             THEN 'pos:' || ord::text
                             ELSE new_type END AS dedup_key
                    FROM jsonb_array_elements(tools) WITH ORDINALITY AS arr(e, ord)
                    CROSS JOIN LATERAL (
                        SELECT CASE e->>'type'
                            WHEN 'cancel_run' THEN 'stop_task'
                            ELSE e->>'type'
                        END AS new_type
                    ) mapped
                ) m
                ORDER BY dedup_key, ord
            ) deduped;
        $fn$
    """)

    for table in _TOOLS_TABLES:
        op.execute(f"""
            UPDATE {table} SET tools = _aios_normalize_legacy_tools(tools)
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements(tools) e
                WHERE e->>'type' IN ({_LEGACY_NAMES_SQL})
            )
        """)

    # sessions.tools is nullable (only frozen child sessions carry a surface).
    op.execute(f"""
        UPDATE sessions SET tools = _aios_normalize_legacy_tools(tools)
        WHERE tools IS NOT NULL AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(tools) e
            WHERE e->>'type' IN ({_LEGACY_NAMES_SQL})
        )
    """)

    op.execute("DROP FUNCTION _aios_normalize_legacy_tools(jsonb)")


def downgrade() -> None:
    # Forward-only: stop_task is a strict superset of cancel_run, so there is nothing to reverse.
    pass
