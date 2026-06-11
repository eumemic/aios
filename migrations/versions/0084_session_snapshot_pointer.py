"""The DB snapshot pointer for durable session sandboxes (#…).

Under durable persistence the daemon/store remains the source of truth for
snapshot *content and lineage*, but horizontal scale (a second worker, host
replacement/rebalancing — a stated requirement) needs the DB to answer the
one question no daemon probe can: *where does this session's snapshot live?*
A worker resuming a session whose snapshot another daemon produced is blind
without it. So ``sessions`` carries a reconciled routing pointer:

* ``snapshot_ref``  — artifact identity: a deterministic, host-independent
  name (the local snapshot tag in v1). NULL = no snapshot.
* ``snapshot_host`` — owning worker/daemon/host id. NULL = none. Equals
  ``settings.instance_id`` in v1 (one worker), kept conceptually distinct
  from the deployment namespace that derives ``snapshot_ref`` so a future
  multi-host deployment never changes a session's ref on handoff.
* ``snapshot_bytes`` — last-known unique size; reporting only, NEVER an
  enforcement input (enforcement always derives from the owning store's
  live enumeration; this column can be one generation stale in a crash
  window).
* ``snapshot_updated_at`` — last pointer write (the compare-and-swap key on
  multi-host shapes; vacuous single-host).

The pointer is written inside the snapshot critical section (after commit
success, before ``rm``, under the per-session lock) and reconciled against
store truth by the GC tick, so it never claims a snapshot that wasn't
committed. All four columns are internal — excluded from the session wire
shape, exactly as the dead ``container_id`` was — so there is no
openapi/SDK churn from this migration.

The same migration **drops the dead ``container_id``** (migration 0001):
zero SQL reads/writes anywhere, superseded by the snapshot pointer.

``sessions`` is not the hot ``events`` table, and every column here is
nullable with NULL = "no snapshot", so this is metadata-only DDL — no
backfill, no table rewrite.

Revision ID: 0084
Revises: 0083
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0084"
down_revision: str = "0083"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE sessions
            ADD COLUMN snapshot_ref        text,
            ADD COLUMN snapshot_host       text,
            ADD COLUMN snapshot_bytes      bigint,
            ADD COLUMN snapshot_updated_at timestamptz;
        """
    )
    op.execute("ALTER TABLE sessions DROP COLUMN container_id;")


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE sessions
            DROP COLUMN snapshot_ref,
            DROP COLUMN snapshot_host,
            DROP COLUMN snapshot_bytes,
            DROP COLUMN snapshot_updated_at;
        """
    )
    op.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS container_id text;")
