"""Add image-aware token baseline v2 state.

Revision ID: 0161
Revises: 0158
"""

from alembic import op

revision = "0161"
# Re-parented 0158 -> 0159 (seat, 2026-08-13). 0159_connection_changes landed on master
# after this migration was authored, and BOTH declared down_revision="0158" -- a branched
# alembic history with two heads, which fails every DB-touching check (unit included, which
# is what distinguished this from the known-flaky e2e shard). A git rebase moves the FILE
# but does not RE-PARENT a migration; that has to be done deliberately.
# Verified before editing: master has exactly ONE head, 0159.
down_revision = "0159"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # lock_timeout on the ``events`` ALTER (review finding, #2050).  PG11+
    # fast-default means adding a DEFAULTed column does NOT rewrite the table,
    # so this is a catalog-only change -- but it still needs ACCESS EXCLUSIVE,
    # and ``events`` is one of the two tables holding ~87% of the DB with
    # appends landing on it continuously.  Without a timeout the ALTER queues
    # behind any live transaction AND every subsequent append queues behind the
    # ALTER: a single long-running reader turns a catalog tweak into a
    # fleet-wide append stall.  Failing fast and retrying is strictly better
    # than an unbounded convoy.  Scoped to this transaction via SET LOCAL.
    op.execute("SET LOCAL lock_timeout = '3s'")
    op.execute(
        "ALTER TABLE events "
        "ADD COLUMN cumulative_image_mass bigint NOT NULL DEFAULT 0, "
        "ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1"
    )
    op.execute("ALTER TABLE sessions ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1")


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP COLUMN token_baseline_v")
    op.execute("ALTER TABLE events DROP COLUMN token_baseline_v, DROP COLUMN cumulative_image_mass")
