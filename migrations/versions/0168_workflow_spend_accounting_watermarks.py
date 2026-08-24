"""Durable provenance for historical workflow-spend reconciliation.

Revision ID: 0168
Revises: 0166
"""

from __future__ import annotations

from alembic import op

revision = "0168"
down_revision = "0166"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE workflow_spend_accounting_watermarks (
            account_id text PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
            accounted_run_cost_microusd bigint NOT NULL
                CHECK (accounted_run_cost_microusd >= 0),
            last_observed_run_cost_microusd bigint
                CHECK (last_observed_run_cost_microusd >= 0),
            last_applied_delta_microusd bigint
                CHECK (last_applied_delta_microusd >= 0),
            reconciled_at timestamptz
        )
    """)
    op.execute(r"""
        COMMENT ON TABLE workflow_spend_accounting_watermarks IS
        'Operator-declared and migration-maintained provenance for the amount '
        'of cumulative wf_runs.call_llm_cost_microusd already represented in '
        'accounts.spent_microusd'
    """)
    op.execute(r"""
        DO $$
        BEGIN
            IF to_regclass('_aios_0168_workflow_spend_watermarks_archive') IS NOT NULL THEN
                IF EXISTS (
                    SELECT 1
                      FROM _aios_0168_workflow_spend_watermarks_archive archived
                      LEFT JOIN accounts a ON a.id = archived.account_id
                     WHERE a.id IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'cannot restore archived 0168 workflow spend watermark: an account is missing';
                END IF;
                INSERT INTO workflow_spend_accounting_watermarks (
                    account_id,
                    accounted_run_cost_microusd,
                    last_observed_run_cost_microusd,
                    last_applied_delta_microusd,
                    reconciled_at
                )
                SELECT account_id,
                       accounted_run_cost_microusd,
                       last_observed_run_cost_microusd,
                       last_applied_delta_microusd,
                       reconciled_at
                  FROM _aios_0168_workflow_spend_watermarks_archive;
                DROP TABLE _aios_0168_workflow_spend_watermarks_archive;
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("LOCK TABLE accounts, workflow_spend_accounting_watermarks IN ACCESS EXCLUSIVE MODE")
    op.execute(r"""
        CREATE TABLE _aios_0168_workflow_spend_watermarks_archive AS
        SELECT account_id,
               accounted_run_cost_microusd,
               last_observed_run_cost_microusd,
               last_applied_delta_microusd,
               reconciled_at
          FROM workflow_spend_accounting_watermarks
    """)
    op.execute("DROP TABLE workflow_spend_accounting_watermarks")
