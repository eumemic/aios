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

Revision ID: 0156
Revises: 0155
"""

from alembic import op

revision = "0156"
down_revision = "0155"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE connection_changes (
            seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
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
            account_id TEXT NOT NULL REFERENCES accounts(id),
            connector TEXT NOT NULL,
            pruned_through_seq BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (account_id, connector)
        )
    """)


def downgrade() -> None:
    op.drop_table("connection_change_horizons")
    op.drop_table("connection_changes")
