"""Residue-events table (#1328) — the de-Goodharted detection-residue gauge.

A fresh append-only ``residue_events`` table, the durable substrate behind
Plane D of the substrate-different-verdict invariant (epic #1330, roadmap item
9). Rows are CLASSIFICATIONS the ops-agent writes; the DENOMINATOR is never
stored (it is computed live from uncorrelated substrates — ``wf_runs`` and the
GitHub merged-PR set — in the render path), so the gauge can never be
maker==checker. The table follows the ``wf_runs`` shape (prefixed-ULID id,
``account_id``, ``created_at timestamptz default now()``) and the small-table
idiom of ``0086``/``0108``.

LOAD-BEARING INVARIANTS, made STRUCTURAL (not convention):

- **Append-only.** A ``BEFORE UPDATE OR DELETE`` trigger ``RAISE EXCEPTION`` —
  the never-delete stance turned into a constraint the query layer cannot
  bypass. There is NO update/delete path in the query module.
- **Axis segregation (the de-Goodhart).** ``axis smallint CHECK (axis IN (1,2))``
  is a column-level invariant; the two axes (axis-1 = machine-found mechanical
  waste; axis-2 = class migration) are STRUCTURALLY distinct and the query/render
  layer never ``SUM``s across them (enforced additionally by a grep unit test).
- **Frozen finder vocabulary.** ``finder CHECK (finder IN (...4 values))`` — a
  new finder is a deliberate migration, not a typo.
- **kind_source provenance.** ``kind_source CHECK (kind_source IN (...3))`` so a
  ``manual``/``other`` row is distinguishable from a stamped-at-source one.
- **OPEN residue_kind enum.** ``residue_kind`` is deliberately NOT CHECK-frozen:
  the four human-in-loop kinds PLUS the ``other`` sentinel. ``other``-bucket
  growth is a render-side ALARM, not a constraint.
- **At-least-once idempotency.** ``UNIQUE(account_id, idempotency_key)`` +
  ON CONFLICT DO NOTHING (the ``reference-workflow-crash-contract-at-least-once``
  generalized to every ingest node) so a gate-resume replay or an observer
  re-scan inserts ONE row, never a double-count.

``downgrade()`` fails HARD if any ``residue_events`` row exists (the 0108
unrepresentable-row / never-reconstruct stance) and then drops the trigger,
function and table.

Revision ID: 0109
Revises: 0108
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0109"
down_revision: str = "0108"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ─── frozen vocabularies (the CHECK predicates; pinned byte-for-byte by the
#     0108-idiom predicate test so a silent vocabulary drift is caught) ─────────

# The segregated axis: 1 = machine-found mechanical waste (observer/axis-1),
# 2 = class migration (gate-resolve/axis-2). NEVER summed.
AXIS_CHECK = "axis IN (1, 2)"

# The ledger's frozen finder vocabulary — a new finder is a deliberate migration.
FINDER_CHECK = (
    "finder IN ('internal-armed-check', 'external-world', 'chairman', 'seat-incidental')"
)

# Provenance of residue_kind: stamped at the gate-resolve source, written by the
# observer, or a manual classification. Distinguishes a stamped-at-source row
# from an ``other``/manual one.
KIND_SOURCE_CHECK = "kind_source IN ('gate-resolve-payload', 'observer', 'manual')"

# The append-only guard: a trigger function that RAISEs on any UPDATE or DELETE.
APPEND_ONLY_FN = "residue_events_append_only"
APPEND_ONLY_TRIGGER = "residue_events_no_update_delete"


def upgrade() -> None:
    # 1. The table — wf_runs shape (prefixed-ULID id, account_id, created_at).
    op.execute(
        f"""
        CREATE TABLE residue_events (
            id                  text PRIMARY KEY,
            account_id          text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            created_at          timestamptz NOT NULL DEFAULT now(),
            axis                smallint NOT NULL CHECK ({AXIS_CHECK}),
            finder              text NOT NULL CHECK ({FINDER_CHECK}),
            residue_kind        text NOT NULL,
            kind_source         text NOT NULL CHECK ({KIND_SOURCE_CHECK}),
            signature           jsonb NOT NULL,
            source_run_id       text,
            source_gate_nonce   text,
            issue_ref           text,
            class_closed_issue  text,
            idempotency_key     text NOT NULL,
            UNIQUE (account_id, idempotency_key)
        )
        """
    )
    # Render-path reads pull each axis SEPARATELY over a created_at window, so an
    # (account_id, axis, created_at) index serves both axis-scoped scans.
    op.execute(
        "CREATE INDEX residue_events_axis_idx "
        "ON residue_events (account_id, axis, created_at DESC)"
    )

    # 2. Append-only made structural: a BEFORE UPDATE OR DELETE trigger that
    #    RAISEs. never-delete is not a convention the query layer can drift from;
    #    it is a constraint Postgres enforces against every writer.
    op.execute(
        f"""
        CREATE FUNCTION {APPEND_ONLY_FN}() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'residue_events is append-only: % is forbidden (never-delete, #1328)',
                TG_OP;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {APPEND_ONLY_TRIGGER}
            BEFORE UPDATE OR DELETE ON residue_events
            FOR EACH ROW EXECUTE FUNCTION {APPEND_ONLY_FN}()
        """
    )


def downgrade() -> None:
    # Fail hard on any existing row — a classification is a never-reconstruct
    # fact (the 0108 unrepresentable-row stance). A residue row deleted on
    # downgrade would silently erase a recorded human-in-loop event.
    n = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM residue_events"))
        .scalar()
    )
    if n:
        raise RuntimeError(
            f"cannot downgrade: {n} residue_events row(s) — the residue ledger is "
            "append-only and never reconstructed (#1328)."
        )

    # The trigger guards UPDATE/DELETE of ROWS; DROP TRIGGER / DROP TABLE are DDL
    # and are not intercepted, so teardown is clean on an empty table.
    op.execute(f"DROP TRIGGER IF EXISTS {APPEND_ONLY_TRIGGER} ON residue_events")
    op.execute(f"DROP FUNCTION IF EXISTS {APPEND_ONLY_FN}()")
    op.execute("DROP TABLE IF EXISTS residue_events")
