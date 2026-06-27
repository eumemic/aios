"""Per-row ``tools_vocab_epoch`` stamp + per-surface partial index (raw-restore belt).

Part of the pattern-retirement epic (#1572); the MECHANISM layer behind the
boot-admission gate. Adds a ``tools_vocab_epoch smallint`` stamp to all SEVEN
persisted tool-surface tables — ``agents``, ``agent_versions``, ``workflows``,
``workflow_versions``, ``wf_runs``, ``sessions``, ``connectors`` — the exact
surface set declared as data on the retirement registry (``TOOL_SURFACES``).

The stamp is the **raw-restore belt**: it records the most recent backfill (the
data migration that rewrites persisted ``ToolSpec`` rows to canonical
vocabulary) a row's blob was known to satisfy, and it travels INSIDE a
``pg_restore``/volume-swap snapshot. A chain-bypassing restore of an old DB
therefore self-describes its staleness — its rows read ``epoch <
latest-backfill-rev`` — even when it bypasses ``aios migrate`` entirely. It is a
belt, not the primary authority: the content-predicate boot scan is the
authority; the stamp lets that scan fast-path to ``MIN(epoch) >= backfill_rev``
per table instead of re-scanning every blob.

**Default 0, intentionally.** Every existing row (and every row in an old raw
snapshot) gets ``tools_vocab_epoch = 0``, which is ``< latest-backfill-rev`` for
any real backfill, so it self-describes as stale and is admitted only after the
boot gate / re-upcast hook brings it current. New write paths stamp the current
epoch (:data:`aios.retirements.epoch.TOOLS_VOCAB_EPOCH`) explicitly so a fresh
row is born current. The column DEFAULT is left at 0 so a direct ``INSERT`` that
does not name the column (a raw restore, a future write path that forgets it)
still produces a *stale* row rather than a silently-current one — fail safe.

**Partial index per surface.** Each table gets a partial index over the stale
rows (``WHERE tools_vocab_epoch < <epoch>``) so the boot residue scan is
O(index) not O(table): boot must be fast. The predicate's threshold is the epoch
of this migration's backfill horizon — pinned as a literal below so the index
definition is reproducible regardless of how the registry evolves later.

``downgrade()`` drops the indexes and the column.

Revision ID: 0124
Revises: 0123
Create Date: 2026-06-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0124"
down_revision: str = "0123"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The SEVEN persisted tool surfaces (table -> nullable jsonb surface column).
# This is the same surface set declared as data on the retirement registry
# (``aios.retirements.registry.TOOL_SURFACES``); kept as a literal here because a
# migration must be self-contained and frozen at author time.
_SURFACE_TABLES: tuple[str, ...] = (
    "agents",
    "agent_versions",
    "workflows",
    "workflow_versions",
    "wf_runs",
    "sessions",
    "connectors",
)

# The epoch horizon this migration indexes against: the integer epoch of the
# latest declared ``tool_surface`` backfill at author time (rev 0122 →
# ``aios.retirements.epoch.TOOLS_VOCAB_EPOCH``). Pinned as a literal so the
# partial-index predicate is reproducible: a later backfill ships its own
# migration that re-creates the index against its newer horizon.
_EPOCH_HORIZON = 122


def upgrade() -> None:
    for table in _SURFACE_TABLES:
        # Additive: NOT NULL with a server default so every existing row gets 0
        # (stale) without a rewrite-blocking backfill. The default is RETAINED
        # (unlike the host-semantics epoch in 0092): a write path / raw restore
        # that omits the column must still produce a stale row, never a
        # silently-current one.
        op.execute(
            f"ALTER TABLE {table} "
            "ADD COLUMN tools_vocab_epoch smallint NOT NULL DEFAULT 0"
        )
        # Partial index over the stale residue only: the boot scan reads this to
        # locate rows needing re-upcast in O(index), and can fast-path a table
        # to "clean" when the index is empty (MIN(epoch) >= horizon).
        op.execute(
            f"CREATE INDEX ix_{table}_tools_vocab_epoch_stale "
            f"ON {table} (tools_vocab_epoch) "
            f"WHERE tools_vocab_epoch < {_EPOCH_HORIZON}"
        )


def downgrade() -> None:
    for table in _SURFACE_TABLES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tools_vocab_epoch_stale")
        op.execute(f"ALTER TABLE {table} DROP COLUMN tools_vocab_epoch")
