"""Add durable sequenced connection discovery ledger + prune horizons.

``connection_changes`` is the sequenced ledger the discovery SSE replays.
Writers serialize per ``(account_id, connector)`` stream via a transaction-
scoped advisory lock (see ``queries.insert_connection_change``), which makes
``seq`` order equal commit order *within a stream* — the property the
``fresh``/``tail`` cursors rely on to never skip a committed change.

``connection_change_horizons`` is the durable per-stream pruning watermark.
``pruned_through_seq`` means "every ledger row with ``seq <= this`` may have
been deleted"; a ``tail`` cursor below it gets a ``reset`` instead of a
silently incomplete replay.  Rows are written by the pruner (#1909) in the
same transaction as its DELETE; absence of a row means "never pruned" (0).
A derived ``MIN(seq)`` floor cannot provide this: it fails open the moment
retention empties the table, and it is global where the cursor is per-stream.

Both ``account_id`` foreign keys are ``ON DELETE CASCADE``.  These are
audit/retention rows, not resources: with a plain (RESTRICT-by-default) FK
the compliance hard-delete path (``queries.hard_delete_account`` ->
``services.accounts.purge_account``) would be permanently blocked for any
account that ever attached a connection, since ledger rows outlive the
archived connections they describe and a horizon row *survives the deletes
it describes* by design — so it would block the purge forever, even after
retention emptied the ledger.  This matches the cascading transient/audit
tables the hard-delete docstring already calls out (``oauth_flows`` 0061,
the workflow tables 0064, ``wf_run_vaults`` 0073, ``trigger_runs`` 0086).

The revision follows 0158 so the migration graph remains a single linear
upgrade path after outbound tool reservations landed on the base branch.

Revision ID: 0159
Revises: 0158
"""

from alembic import op

revision = "0159"
down_revision = "0158"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE connection_changes (
            seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            connector TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('added', 'removed')),
            connection_id TEXT NOT NULL,
            external_account_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX connection_changes_discovery_idx
        ON connection_changes (account_id, connector, seq)
    """)
    op.execute("""
        CREATE TABLE connection_change_horizons (
            account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            connector TEXT NOT NULL,
            pruned_through_seq BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (account_id, connector)
        )
    """)


def downgrade() -> None:
    op.drop_table("connection_change_horizons")
    op.drop_table("connection_changes")
