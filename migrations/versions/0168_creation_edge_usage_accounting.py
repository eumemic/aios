"""Creation-edge inference accounting and rolling usage ledger.

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
    # edge.  Backfill those before considering session caller provenance.
    op.execute("UPDATE sessions SET creator_run_id = parent_run_id WHERE parent_run_id IS NOT NULL")
    # Historical call_agent children are ephemeral (archive_when_idle) fresh
    # sessions whose first request_opened frame names the creating session.
    # Existing-session call_session targets are not ephemeral, so they are
    # intentionally excluded: invocation never changes accounting ownership.
    op.execute(r"""
        UPDATE sessions s
           SET creator_session_id = (
               SELECT e.data->'caller'->>'id'
                 FROM events e
                 JOIN sessions creator
                   ON creator.id = e.data->'caller'->>'id'
                  AND creator.account_id = s.account_id
                WHERE e.session_id = s.id
                  AND e.account_id = s.account_id
                  AND e.kind = 'lifecycle'
                  AND e.data->>'event' = 'request_opened'
                  AND e.data->'caller'->>'kind' = 'session'
                  AND e.data->'caller'->>'id' <> s.id
                ORDER BY e.seq
                LIMIT 1
           )
         WHERE s.creator_run_id IS NULL
           AND s.archive_when_idle = TRUE
           AND EXISTS (
               SELECT 1
                 FROM events e
                 JOIN sessions creator
                   ON creator.id = e.data->'caller'->>'id'
                  AND creator.account_id = s.account_id
                WHERE e.session_id = s.id
                  AND e.account_id = s.account_id
                  AND e.kind = 'lifecycle'
                  AND e.data->>'event' = 'request_opened'
                  AND e.data->'caller'->>'kind' = 'session'
                  AND e.data->'caller'->>'id' <> s.id
           )
    """)
    # Soft resource provenance predates this migration and is a useful fallback
    # for any session-created resource not covered by the invocation substrate.
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

    # Raw call_llm() already had a run-level cost meter.  Add its token peers so
    # a workflow run's own inference is the same complete vector as a session's.
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

    # Recover historical raw-turn tokens wherever the durable run journal is
    # still present. The call_started row identifies call_llm unambiguously;
    # the unique (run_id, call_key, type) memo guarantees one matching result.
    # Pruned journals cannot supply tokens, so cost remains the exact historical
    # source of truth for those older runs.
    op.execute(r"""
        UPDATE wf_runs r
           SET call_llm_input_tokens = history.input_tokens,
               call_llm_output_tokens = history.output_tokens,
               call_llm_cache_read_input_tokens = history.cache_read_input_tokens,
               call_llm_cache_creation_input_tokens = history.cache_creation_input_tokens
          FROM (
               SELECT started.run_id,
                      SUM(CASE WHEN jsonb_typeof(done.payload->'result'->'usage'->'input_tokens')
                                         = 'number'
                               THEN (done.payload->'result'->'usage'->>'input_tokens')::bigint
                               ELSE 0 END)::bigint AS input_tokens,
                      SUM(CASE WHEN jsonb_typeof(done.payload->'result'->'usage'->'output_tokens')
                                         = 'number'
                               THEN (done.payload->'result'->'usage'->>'output_tokens')::bigint
                               ELSE 0 END)::bigint AS output_tokens,
                      SUM(CASE WHEN jsonb_typeof(
                                             done.payload->'result'->'usage'
                                                 ->'cache_read_input_tokens'
                                         ) = 'number'
                               THEN (done.payload->'result'->'usage'
                                         ->>'cache_read_input_tokens')::bigint
                               ELSE 0 END)::bigint AS cache_read_input_tokens,
                      SUM(CASE WHEN jsonb_typeof(
                                             done.payload->'result'->'usage'
                                                 ->'cache_creation_input_tokens'
                                         ) = 'number'
                               THEN (done.payload->'result'->'usage'
                                         ->>'cache_creation_input_tokens')::bigint
                               ELSE 0 END)::bigint AS cache_creation_input_tokens
                 FROM wf_run_events started
                 JOIN wf_run_events done
                   ON done.run_id = started.run_id
                  AND done.call_key = started.call_key
                  AND done.type = 'call_result'
                WHERE started.type = 'call_started'
                  AND started.payload->>'capability' = 'call_llm'
                GROUP BY started.run_id
          ) history
         WHERE r.id = history.run_id
    """)

    # Repair the original workflows-as-models accounting omission: historical
    # raw call_llm cost lived only on wf_runs, while every public/account limit
    # read trusts accounts.spent_microusd. Future charges dual-write atomically.
    op.execute(r"""
        UPDATE accounts a
           SET spent_microusd = a.spent_microusd + raw.total_microusd
          FROM (
               SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total_microusd
                 FROM wf_runs
                GROUP BY account_id
          ) raw
         WHERE a.id = raw.account_id
    """)

    # Rates need deltas, not cumulative rows. Per-account coverage makes a
    # partial first window explicit and follows account lifecycle naturally.
    op.execute(
        "ALTER TABLE accounts ADD COLUMN usage_ledger_started_at timestamptz NOT NULL DEFAULT now()"
    )
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
    op.execute("DROP TABLE inference_usage_ledger")
    # ``IF EXISTS`` also makes local iteration safe for databases that briefly
    # ran the pre-review singleton coverage design of this unreleased revision.
    op.execute("DROP TABLE IF EXISTS inference_usage_ledger_state")
    op.execute("ALTER TABLE accounts DROP COLUMN IF EXISTS usage_ledger_started_at")
    op.execute(r"""
        UPDATE accounts a
           SET spent_microusd = GREATEST(0, a.spent_microusd - raw.total_microusd)
          FROM (
               SELECT account_id, SUM(call_llm_cost_microusd)::bigint AS total_microusd
                 FROM wf_runs
                GROUP BY account_id
          ) raw
         WHERE a.id = raw.account_id
    """)
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
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_creator_run_account_fk")
        op.execute(
            f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_creator_session_account_fk"
        )
    op.execute("ALTER TABLE wf_runs DROP CONSTRAINT wf_runs_creator_not_self_ck")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_creator_not_self_ck")
    for table in reversed(("sessions", "wf_runs")):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_one_creator_ck")
        op.execute(f"ALTER TABLE {table} DROP COLUMN creator_run_id")
        op.execute(f"ALTER TABLE {table} DROP COLUMN creator_session_id")
