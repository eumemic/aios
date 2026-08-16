"""Contract the legacy invocation-kernel builtin names.

Re-runs the 0116/0117 canonicalization over every registry-declared tool
surface, including ``connectors.tools_schema`` which those migrations predated.
Rows are stamped at vocabulary epoch 155 and the stale-row indexes advance to
that horizon.  This closes the expand span so the read validator can stop
accepting the retired names.

Revision ID: 0155
Revises: 0154
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0155"
down_revision: str = "0154"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SURFACES: tuple[tuple[str, str], ...] = (
    ("agents", "tools"),
    ("agent_versions", "tools"),
    ("workflows", "tools"),
    ("workflow_versions", "tools"),
    ("wf_runs", "tools"),
    ("sessions", "tools"),
    ("connectors", "tools_schema"),
)
_LEGACY_NAMES_SQL = (
    "'invoke','invoke_agent','invoke_workflow','create_run','await_run','cancel_run'"
)
_EPOCH_HORIZON = 155


def upgrade() -> None:
    op.execute(r"""
        CREATE FUNCTION _aios_contract_legacy_tools(tools jsonb) RETURNS jsonb
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
                            WHEN 'cancel_run'      THEN 'stop_task'
                            ELSE e->>'type'
                        END AS new_type
                    ) mapped
                ) mapped_elements
                ORDER BY dedup_key, ord
            ) deduped;
        $fn$
    """)

    for table, column in _SURFACES:
        # ``workflow_versions`` is immutable at runtime. Temporarily suspend its
        # guard for this migration-owned canonicalization, as migration 0154 does
        # for its account-id rewrite.
        immutable = table == "workflow_versions"
        if immutable:
            op.execute("ALTER TABLE workflow_versions DISABLE TRIGGER workflow_versions_no_update")
        try:
            op.execute(f"""
                UPDATE {table}
                SET {column} = CASE
                        WHEN {column} IS NULL THEN NULL
                        WHEN EXISTS (
                            SELECT 1 FROM jsonb_array_elements({column}) e
                            WHERE e->>'type' IN ({_LEGACY_NAMES_SQL})
                        ) THEN _aios_contract_legacy_tools({column})
                        ELSE {column}
                    END,
                    tools_vocab_epoch = {_EPOCH_HORIZON}
            """)
        finally:
            if immutable:
                op.execute(
                    "ALTER TABLE workflow_versions ENABLE TRIGGER workflow_versions_no_update"
                )
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tools_vocab_epoch_stale")
        op.execute(
            f"CREATE INDEX ix_{table}_tools_vocab_epoch_stale "
            f"ON {table} (tools_vocab_epoch) "
            f"WHERE tools_vocab_epoch < {_EPOCH_HORIZON}"
        )

    op.execute("DROP FUNCTION _aios_contract_legacy_tools(jsonb)")


def downgrade() -> None:
    # The many-to-one renames are forward-only. Restore the previous epoch index
    # horizon without attempting to reconstruct retired vocabulary.
    for table, _column in _SURFACES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tools_vocab_epoch_stale")
        op.execute(
            f"CREATE INDEX ix_{table}_tools_vocab_epoch_stale "
            f"ON {table} (tools_vocab_epoch) WHERE tools_vocab_epoch < 122"
        )
