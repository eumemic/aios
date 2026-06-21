"""Normalize pre-#1419 builtin tool names in persisted ``tools`` JSONB.

The invocation-kernel epic renamed the model tools ``invoke``→``call_session``,
``invoke_agent``→``call_agent``, ``invoke_workflow``→``call_workflow`` and folded the
two-step ``create_run``/``await_run`` launch pair into ``call_workflow``. Those names are
validated against the ``BuiltinToolType`` Literal on read, so any row whose ``tools``
JSONB still carries a legacy name would fail validation post-deploy (a deploy-breaker for
every agent that exposed workflow-launch/session-invoke to its model).

This migration rewrites the legacy names to canonical in every table that stores an
agent/workflow ``ToolSpec`` surface — ``agents``, ``agent_versions``, ``workflows``,
``workflow_versions``, ``wf_runs`` (the launch-pinned surface), and ``sessions`` (the
frozen child surface, nullable). Multiple legacy tools that collapse to the same name
(``invoke_workflow`` + ``create_run`` → ``call_workflow``) are deduped to a single entry,
first-occurrence-wins (preserving its permission/transport overrides), with original
order otherwise preserved. The model code carries a matching read-tolerance shim
(``_LEGACY_BUILTIN_RENAMES`` in ``models/agents.py``) for the post-deploy/pre-migrate
window; that shim can be removed once this has run everywhere.

``downgrade()`` is a no-op: the rename is forward-only (``call_workflow`` cannot be
un-collapsed back into ``invoke_workflow`` vs ``create_run``/``await_run``).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0116"
down_revision: str = "0115"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables with a NOT-NULL ``tools`` JSONB surface column.
_TOOLS_TABLES = ("agents", "agent_versions", "workflows", "workflow_versions", "wf_runs")
_LEGACY_NAMES_SQL = "'invoke','invoke_agent','invoke_workflow','create_run','await_run'"


def upgrade() -> None:
    # Map + dedupe a tools array: rename legacy builtin names, then keep one entry per
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
                            WHEN 'invoke'          THEN 'call_session'
                            WHEN 'invoke_agent'    THEN 'call_agent'
                            WHEN 'invoke_workflow' THEN 'call_workflow'
                            WHEN 'create_run'      THEN 'call_workflow'
                            WHEN 'await_run'       THEN 'call_workflow'
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
    # Forward-only: call_workflow cannot be un-collapsed into invoke_workflow vs
    # create_run/await_run, so there is nothing to reverse.
    pass
