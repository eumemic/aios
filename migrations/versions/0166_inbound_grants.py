"""Runtime inbound approval grants ledger.

Revision ID: 0166
Revises: 0165
"""

from alembic import op

# Renumbered 0164 -> 0166 and re-parented onto 0165 (fix round 2, 2026-08-17).
#
# History of this file's breakage, because the shape matters:
#   * Authored when master's head was 0159, so it was written as rev 0164 /
#     parent 0159. Master then advanced 0159 -> 0161 -> 0162 -> 0163 -> 0165.
#     A git rebase moves the FILE but does NOT re-parent a migration, so after
#     the rebase 0164 (parent 0159) and 0165 (parent 0163) were both heads:
#     a genuine DOUBLE HEAD.
#   * Fix round 1 re-parented 0164 onto 0165. That removed the double head and
#     the on-disk ladder really was linear -- but it left the ladder
#     NON-MONOTONIC: the head (0164) sorted BELOW an ancestor (0165).
#
# Non-monotonic ids are not cosmetic here. The retirement generator
# (aios.retirements.migration_gen) extends the ladder with next_revisions(head),
# i.e. head+1. With head=0164 it emitted 0165/0166/0167 -- and 0165 ALREADY
# EXISTED. Two files then claimed id 0165 ("Revision 0165 is present more than
# once"), alembic merged them into one node inheriting both parents, and
# 0164 -> 0165 -> 0164 closed a CYCLE, surfacing as
# CycleDetected(0164, 0165, 0166, 0167).
#
# So the real invariant is stronger than "one head": the head must also be the
# numerically HIGHEST id, or the generator's head+1 collides with an existing
# revision. Renumbering to 0166 restores that. Safe to renumber: this branch is
# unmerged and unapplied, no database is stamped 0164, and no code or test
# referenced the id (grep-verified) -- alembic ids are opaque strings.
revision = "0166"
down_revision = "0165"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE inbound_grants (
            id text PRIMARY KEY DEFAULT gen_random_uuid()::text,
            account_id text NOT NULL REFERENCES accounts(id),
            connection_id text NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            chat_id text NOT NULL,
            status text NOT NULL CHECK (status IN ('pending', 'active', 'revoked')),
            approved_by text REFERENCES accounts(id),
            approved_at timestamptz,
            approved_via_channel text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE UNIQUE INDEX inbound_grants_live_uniq
            ON inbound_grants (connection_id, chat_id) WHERE status <> 'revoked';
        CREATE INDEX inbound_grants_pending_gc_idx
            ON inbound_grants (created_at) WHERE status = 'pending';
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE inbound_grants")
