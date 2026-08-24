"""Creation-edge inference accounting and rolling usage ledger.

Revision ID: 0168
Revises: 0167
"""

from __future__ import annotations

from alembic import op

revision = "0168"
down_revision = "0167"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This revision takes ACCESS EXCLUSIVE locks as it adds columns. Bound how
    # long deploy waits for old writers and how long the whole repair statement
    # may run; PostgreSQL rolls the revision back atomically on either timeout.
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '5min'")
    op.execute("LOCK TABLE workflow_spend_accounting_watermarks IN ACCESS EXCLUSIVE MODE")

    # One immutable accounting parent per node.  Two nullable typed FKs avoid a
    # polymorphic id with no referential integrity; the XOR check makes the
    # creation graph single-parented (roots leave both NULL).
    for table in ("sessions", "wf_runs"):
        op.execute(f"ALTER TABLE {table} ADD COLUMN creator_session_id text")
        op.execute(f"ALTER TABLE {table} ADD COLUMN creator_run_id text")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {table}_one_creator_ck CHECK ("
            "creator_session_id IS NULL OR creator_run_id IS NULL)"
        )

    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_creator_not_self_ck "
        "CHECK (creator_session_id IS NULL OR creator_session_id <> id)"
    )
    op.execute(
        "ALTER TABLE wf_runs ADD CONSTRAINT wf_runs_creator_not_self_ck "
        "CHECK (creator_run_id IS NULL OR creator_run_id <> id)"
    )

    # Existing workflow agent() children already have an unambiguous creation
    # edge. Backfill those before considering explicit resource provenance.
    op.execute("UPDATE sessions SET creator_run_id = parent_run_id WHERE parent_run_id IS NOT NULL")
    # Soft resource provenance predates this migration and records who created
    # the resource, rather than who later invoked it. It is the only safe legacy
    # session -> session backfill. In particular, archive_when_idle plus a first
    # request_opened event is NOT creation evidence: the public API can create a
    # self-archiving root which call_session invokes later. Ambiguous historical
    # rows intentionally remain roots instead of transferring their spend to an
    # invocation peer. New call_agent writes creator_session_id at insert time.
    op.execute(r"""
        UPDATE sessions s
           SET creator_session_id = s.created_by_ref
         WHERE s.creator_run_id IS NULL
           AND s.creator_session_id IS NULL
           AND s.created_by_type = 'session_actor'
           AND s.created_by_ref <> s.id
           AND EXISTS (
               SELECT 1 FROM sessions creator
                WHERE creator.id = s.created_by_ref
                  AND creator.account_id = s.account_id
           )
    """)

    # A run is always newly created.  Its trusted caller is therefore the best
    # historical creation edge.  Detached launches fall back to their launcher
    # session; a bare nested run falls back to parent_run_id.
    op.execute(r"""
        UPDATE wf_runs r
           SET creator_session_id = r.caller->>'id'
         WHERE r.caller->>'kind' = 'session'
           AND EXISTS (
               SELECT 1 FROM sessions s
                WHERE s.id = r.caller->>'id' AND s.account_id = r.account_id
           )
    """)

    # Creator ids are tenant-bearing edges. Composite FKs make a cross-account
    # ownership edge impossible even if a future internal writer forgets a
    # service-layer scope check. PostgreSQL's column-list SET NULL preserves the
    # child's non-null account_id when a creator is hard-deleted.
    for table in ("sessions", "wf_runs"):
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {table}_creator_session_account_fk "
            "FOREIGN KEY (creator_session_id, account_id) "
            "REFERENCES sessions(id, account_id) "
            "ON DELETE SET NULL (creator_session_id) NOT VALID"
        )
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {table}_creator_run_account_fk "
            "FOREIGN KEY (creator_run_id, account_id) "
            "REFERENCES wf_runs(id, account_id) "
            "ON DELETE SET NULL (creator_run_id) NOT VALID"
        )
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {table}_creator_session_account_fk")
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {table}_creator_run_account_fk")
    op.execute(r"""
        UPDATE wf_runs r
           SET creator_run_id = r.caller->>'id'
         WHERE r.creator_session_id IS NULL
           AND r.caller->>'kind' = 'run'
           AND r.caller->>'id' <> r.id
           AND EXISTS (
               SELECT 1 FROM wf_runs parent
                WHERE parent.id = r.caller->>'id' AND parent.account_id = r.account_id
           )
    """)
    op.execute(r"""
        UPDATE wf_runs r
           SET creator_session_id = r.launcher_session_id
         WHERE r.creator_session_id IS NULL
           AND r.creator_run_id IS NULL
           AND r.launcher_session_id IS NOT NULL
    """)
    op.execute(r"""
        UPDATE wf_runs r
           SET creator_run_id = r.parent_run_id
         WHERE r.creator_session_id IS NULL
           AND r.creator_run_id IS NULL
           AND r.parent_run_id IS NOT NULL
           AND r.parent_run_id <> r.id
    """)

    for table in ("sessions", "wf_runs"):
        op.execute(
            f"CREATE INDEX {table}_creator_session_idx ON {table} "
            "(account_id, creator_session_id) WHERE creator_session_id IS NOT NULL"
        )
        op.execute(
            f"CREATE INDEX {table}_creator_run_idx ON {table} "
            "(account_id, creator_run_id) WHERE creator_run_id IS NOT NULL"
        )

    # Raw call_llm() already had a run-level cost meter. Add its token peers.
    # Existing runs start explicitly incomplete: journals may be pruned or
    # malformed, so zero is not evidence of a measured zero. New runs default
    # complete because every post-migration writer persists all four counters.
    for column in (
        "call_llm_input_tokens",
        "call_llm_output_tokens",
        "call_llm_cache_read_input_tokens",
        "call_llm_cache_creation_input_tokens",
    ):
        op.execute(
            f"ALTER TABLE wf_runs ADD COLUMN {column} bigint NOT NULL DEFAULT 0 "
            f"CHECK ({column} >= 0)"
        )
    op.execute(
        "ALTER TABLE wf_runs ADD COLUMN call_llm_tokens_complete boolean NOT NULL DEFAULT FALSE"
    )
    op.execute("ALTER TABLE wf_runs ALTER COLUMN call_llm_tokens_complete SET DEFAULT TRUE")
    op.execute(r"""
        DO $$
        BEGIN
            IF to_regclass('_aios_0168_wf_run_usage_archive') IS NOT NULL THEN
                IF EXISTS (
                    SELECT 1
                      FROM _aios_0168_wf_run_usage_archive archived
                      LEFT JOIN wf_runs r
                        ON r.id = archived.run_id
                       AND r.account_id = archived.account_id
                     WHERE r.id IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'cannot restore archived 0168 run usage: a workflow run is missing';
                END IF;
                UPDATE wf_runs r
                   SET call_llm_input_tokens = a.call_llm_input_tokens,
                       call_llm_output_tokens = a.call_llm_output_tokens,
                       call_llm_cache_read_input_tokens = a.call_llm_cache_read_input_tokens,
                       call_llm_cache_creation_input_tokens = a.call_llm_cache_creation_input_tokens,
                       call_llm_tokens_complete = a.call_llm_tokens_complete
                  FROM _aios_0168_wf_run_usage_archive a
                 WHERE r.id = a.run_id
                   AND r.account_id = a.account_id;
                DROP TABLE _aios_0168_wf_run_usage_archive;
            END IF;
        END
        $$
    """)

    # Make the database own the canonical account projection. The trigger is
    # deliberately installed before historical reconciliation: an old writer
    # blocked by this migration's table lock resumes after COMMIT and still
    # dual-writes its delta exactly once. New application writers update only
    # the run meter, so old/new cutover cannot double-charge the account.
    op.execute(r"""
        CREATE FUNCTION _aios_charge_workflow_account_spend() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            delta bigint;
        BEGIN
            delta := NEW.call_llm_cost_microusd - OLD.call_llm_cost_microusd;
            IF delta < 0 THEN
                RAISE EXCEPTION 'wf_runs.call_llm_cost_microusd is append-only';
            END IF;
            IF delta > 0 THEN
                UPDATE accounts
                   SET spent_microusd = spent_microusd + delta
                 WHERE id = NEW.account_id;
                INSERT INTO workflow_spend_accounting_watermarks AS watermark (
                    account_id,
                    accounted_run_cost_microusd,
                    last_observed_run_cost_microusd,
                    last_applied_delta_microusd,
                    reconciled_at
                ) VALUES (NEW.account_id, delta, delta, delta, now())
                ON CONFLICT (account_id) DO UPDATE
                    SET accounted_run_cost_microusd =
                            watermark.accounted_run_cost_microusd + delta,
                        last_observed_run_cost_microusd =
                            watermark.accounted_run_cost_microusd + delta,
                        last_applied_delta_microusd = delta,
                        reconciled_at = now();
            END IF;
            RETURN NEW;
        END
        $$
    """)
    op.execute(r"""
        CREATE TRIGGER wf_runs_charge_account_spend_trg
        AFTER UPDATE OF call_llm_cost_microusd ON wf_runs
        FOR EACH ROW
        WHEN (NEW.call_llm_cost_microusd <> OLD.call_llm_cost_microusd)
        EXECUTE FUNCTION _aios_charge_workflow_account_spend()
    """)

    # Aggregate equality cannot prove which historical workflow charges were
    # already counted: unrelated/manual spend and deleted sessions can produce
    # the same scalar. Revision 0167 therefore requires an operator-declared
    # watermark for every account with retained workflow cost. Reconcile only
    # the unaccounted delta, then advance the watermark atomically. The trigger
    # above maintains the same provenance for every post-migration charge.
    op.execute(r"""
        DO $$
        BEGIN
            IF EXISTS (
                WITH run_meter AS (
                    SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total
                      FROM wf_runs GROUP BY account_id
                )
                SELECT 1
                  FROM run_meter r
                  LEFT JOIN workflow_spend_accounting_watermarks watermark
                    ON watermark.account_id = r.account_id
                 WHERE r.total > 0 AND watermark.account_id IS NULL
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'missing workflow spend accounting watermark before 0168',
                    HINT = 'at revision 0167, explicitly record how much retained workflow cost is already represented in each account meter';
            END IF;
            IF EXISTS (
                WITH run_meter AS (
                    SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total
                      FROM wf_runs GROUP BY account_id
                )
                SELECT 1
                  FROM workflow_spend_accounting_watermarks watermark
                  LEFT JOIN run_meter r ON r.account_id = watermark.account_id
                 WHERE watermark.accounted_run_cost_microusd > COALESCE(r.total, 0)
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'workflow spend accounting watermark exceeds retained run cost',
                    HINT = 'correct the operator-declared watermark at revision 0167 before retrying 0168';
            END IF;
        END
        $$
    """)
    op.execute(r"""
        UPDATE accounts a
           SET spent_microusd = a.spent_microusd
               + r.total - watermark.accounted_run_cost_microusd
          FROM (
               SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total
                 FROM wf_runs GROUP BY account_id
          ) r
          JOIN workflow_spend_accounting_watermarks watermark
            ON watermark.account_id = r.account_id
         WHERE a.id = r.account_id
           AND r.total > 0
    """)
    op.execute(r"""
        UPDATE workflow_spend_accounting_watermarks watermark
           SET last_applied_delta_microusd =
                   r.total - watermark.accounted_run_cost_microusd,
               accounted_run_cost_microusd = r.total,
               last_observed_run_cost_microusd = r.total,
               reconciled_at = now()
          FROM (
               SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total
                 FROM wf_runs GROUP BY account_id
          ) r
         WHERE watermark.account_id = r.account_id
    """)

    # Rates need deltas, not cumulative rows. Per-account coverage makes a
    # partial first window explicit and follows account lifecycle naturally.
    op.execute(
        "ALTER TABLE accounts ADD COLUMN usage_ledger_started_at timestamptz NOT NULL DEFAULT now()"
    )
    op.execute(r"""
        DO $$
        BEGIN
            IF to_regclass('_aios_0168_account_usage_archive') IS NOT NULL THEN
                IF EXISTS (
                    SELECT 1
                      FROM _aios_0168_account_usage_archive archived
                      LEFT JOIN accounts a ON a.id = archived.account_id
                     WHERE a.id IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'cannot restore archived 0168 coverage: an account is missing';
                END IF;
                UPDATE accounts a
                   SET usage_ledger_started_at = archived.usage_ledger_started_at
                  FROM _aios_0168_account_usage_archive archived
                 WHERE a.id = archived.account_id;
                DROP TABLE _aios_0168_account_usage_archive;
            END IF;
        END
        $$
    """)
    op.execute(r"""
        CREATE TABLE inference_usage_ledger (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            account_id text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            session_id text REFERENCES sessions(id) ON DELETE CASCADE,
            run_id text REFERENCES wf_runs(id) ON DELETE CASCADE,
            input_tokens bigint NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
            output_tokens bigint NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
            cache_read_input_tokens bigint NOT NULL DEFAULT 0
                CHECK (cache_read_input_tokens >= 0),
            cache_creation_input_tokens bigint NOT NULL DEFAULT 0
                CHECK (cache_creation_input_tokens >= 0),
            cost_microusd bigint NOT NULL DEFAULT 0 CHECK (cost_microusd >= 0),
            occurred_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT inference_usage_ledger_one_node_ck CHECK (
                (session_id IS NOT NULL)::integer + (run_id IS NOT NULL)::integer = 1
            )
        )
    """)
    op.execute(r"""
        DO $$
        BEGIN
            IF to_regclass('_aios_0168_inference_usage_ledger_archive') IS NOT NULL THEN
                INSERT INTO inference_usage_ledger (
                    id,
                    account_id,
                    session_id,
                    run_id,
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    cost_microusd,
                    occurred_at
                )
                OVERRIDING SYSTEM VALUE
                SELECT id,
                       account_id,
                       session_id,
                       run_id,
                       input_tokens,
                       output_tokens,
                       cache_read_input_tokens,
                       cache_creation_input_tokens,
                       cost_microusd,
                       occurred_at
                  FROM _aios_0168_inference_usage_ledger_archive;
                PERFORM setval(
                    pg_get_serial_sequence('inference_usage_ledger', 'id'),
                    COALESCE((SELECT MAX(id) FROM inference_usage_ledger), 1),
                    EXISTS (SELECT 1 FROM inference_usage_ledger)
                );
                DROP TABLE _aios_0168_inference_usage_ledger_archive;
            END IF;
        END
        $$
    """)
    op.execute(
        "CREATE INDEX inference_usage_ledger_session_window_idx "
        "ON inference_usage_ledger (session_id, occurred_at DESC) "
        "WHERE session_id IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX inference_usage_ledger_run_window_idx "
        "ON inference_usage_ledger (run_id, occurred_at DESC) WHERE run_id IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX inference_usage_ledger_account_window_idx "
        "ON inference_usage_ledger (account_id, occurred_at DESC)"
    )


def downgrade() -> None:
    # Older application revisions do not know about the rolling ledger or
    # token-completeness flag. Archive that evidence instead of deleting it;
    # a later re-upgrade restores it before accepting new writes. Account spend
    # remains at its canonical value rather than attempting a lossy subtraction.
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '5min'")
    op.execute("LOCK TABLE wf_runs, accounts, inference_usage_ledger IN ACCESS EXCLUSIVE MODE")
    op.execute(r"""
        CREATE TABLE _aios_0168_wf_run_usage_archive AS
        SELECT id AS run_id,
               account_id,
               call_llm_input_tokens,
               call_llm_output_tokens,
               call_llm_cache_read_input_tokens,
               call_llm_cache_creation_input_tokens,
               call_llm_tokens_complete
          FROM wf_runs
    """)
    op.execute(r"""
        CREATE TABLE _aios_0168_account_usage_archive AS
        SELECT id AS account_id, usage_ledger_started_at
          FROM accounts
    """)
    op.execute(r"""
        CREATE TABLE _aios_0168_inference_usage_ledger_archive AS
        SELECT id,
               account_id,
               session_id,
               run_id,
               input_tokens,
               output_tokens,
               cache_read_input_tokens,
               cache_creation_input_tokens,
               cost_microusd,
               occurred_at
          FROM inference_usage_ledger
    """)
    op.execute("DROP TABLE inference_usage_ledger")
    op.execute("DROP TRIGGER wf_runs_charge_account_spend_trg ON wf_runs")
    op.execute("DROP FUNCTION _aios_charge_workflow_account_spend()")
    op.execute("ALTER TABLE accounts DROP COLUMN usage_ledger_started_at")
    op.execute("ALTER TABLE wf_runs DROP COLUMN call_llm_tokens_complete")
    for column in reversed(
        (
            "call_llm_input_tokens",
            "call_llm_output_tokens",
            "call_llm_cache_read_input_tokens",
            "call_llm_cache_creation_input_tokens",
        )
    ):
        op.execute(f"ALTER TABLE wf_runs DROP COLUMN {column}")
    for table in reversed(("sessions", "wf_runs")):
        op.execute(f"DROP INDEX {table}_creator_run_idx")
        op.execute(f"DROP INDEX {table}_creator_session_idx")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_creator_run_account_fk")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_creator_session_account_fk")
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT wf_runs_creator_not_self_ck")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_creator_not_self_ck")
    for table in reversed(("sessions", "wf_runs")):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_one_creator_ck")
        op.execute(f"ALTER TABLE {table} DROP COLUMN creator_run_id")
        op.execute(f"ALTER TABLE {table} DROP COLUMN creator_session_id")
