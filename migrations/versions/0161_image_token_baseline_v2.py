"""Add image-aware token baseline v2 state.

Revision ID: 0161
Revises: 0158
"""

import time

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

_MAX_ATTEMPTS = 5
_LOCK_TIMEOUT = "3s"
_RETRY_SLEEP_SECONDS = 1.0


def _add_events_columns_with_retry() -> None:
    """Acquire the hot events-table lock with bounded retry, then fail hard."""
    bind = op.get_bind()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            # A failed statement aborts its transaction. A savepoint makes each
            # attempt independently recoverable while Alembic keeps ownership of
            # the outer migration transaction.
            with bind.begin_nested():
                bind.exec_driver_sql(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
                bind.exec_driver_sql(
                    "ALTER TABLE events "
                    "ADD COLUMN cumulative_image_mass bigint NOT NULL DEFAULT 0, "
                    "ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1"
                )
            return
        except Exception:
            if attempt == _MAX_ATTEMPTS:
                raise
            time.sleep(_RETRY_SLEEP_SECONDS)


def upgrade() -> None:
    # PG11+ fast-default makes this catalog-only, but ACCESS EXCLUSIVE can still
    # queue every append behind a long reader. Bound each wait, retry transient
    # contention, and propagate the final failure so deployment cannot go green.
    _add_events_columns_with_retry()
    op.execute("ALTER TABLE sessions ADD COLUMN token_baseline_v smallint NOT NULL DEFAULT 1")


def downgrade() -> None:
    # Operational rollback is image-only: migrations are never auto-reverted.
    # This remains for deliberate operator recovery and Alembic completeness.
    op.execute("ALTER TABLE sessions DROP COLUMN token_baseline_v")
    op.execute("ALTER TABLE events DROP COLUMN token_baseline_v, DROP COLUMN cumulative_image_mass")
