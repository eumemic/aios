"""Triggers: ``external_event`` source (#1281) — authenticated inbound webhook.

Additive, capped-table, zero row rewrites — the growth rule's "a new behavior
is a new ``source`` *kind*, never a flag" applied verbatim (the 0086 slice-2
idiom). external_event is a THIRD reactive source: like run_completion it is
unschedulable by the tick (``next_fire`` permanently NULL), but it fires from a
new authenticated HTTP ingress rather than a run-completion transaction.

- ``triggers.ingest_token_hash text`` — the SHA-256-at-rest of the per-trigger
  ingest secret (``aios_evt_…``). NULL for every non-external_event row; the
  catalog-only ADD COLUMN is a zero-rewrite.
- ``triggers_source_spec_shape`` swap: the 0086 predicate gains a
  ``WHEN 'external_event'`` branch ({}-spec: no schedule/fire_at/workflow_id);
  every pre-existing branch stays BYTE-IDENTICAL (0086's verbatim predicate is
  embedded as ``*_0086`` for ``downgrade()``).
- ``triggers_run_completion_no_next_fire`` → renamed
  ``triggers_reactive_no_next_fire`` and broadened to cover external_event:
  ``source NOT IN ('run_completion','external_event') OR next_fire IS NULL``.
- ``triggers_ingest_token_iff_external_event`` — the
  ``environment_id_iff_workflow`` idiom: ``(source = 'external_event') =
  (ingest_token_hash IS NOT NULL)``. Existing rows pass instantly (both sides
  false).
- ``triggers_ingest_token_hash`` UNIQUE partial index — the resolver lookup
  index AND the double-mint collision guard.
- ``trigger_runs.trigger_context`` CHECK extended to include
  ``'external_event'`` (DROP + ADD; Postgres auto-named the 0086 inline CHECK
  ``trigger_runs_trigger_context_check``).

``downgrade()`` fails hard on any ``source='external_event'`` row (the 0086
unrepresentable-row stance) and reverses every change. No cross-migration import
(migrations load under synthetic module names) — the 0086 predicates are
embedded verbatim.

Revision ID: 0108
Revises: 0107
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0108"
down_revision: str = "0107"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ─── source-spec shape: 0086 predicate + the new external_event branch ────────
# Outer COALESCE(…, false) is load-bearing (the 0086 idiom): ``->`` on an absent
# key returns SQL NULL, and a NULL CHECK passes, so a row missing a required key
# would insert undetected without it.
SOURCE_SPEC_PREDICATE = """COALESCE((
    CASE source
        WHEN 'cron' THEN
            jsonb_typeof(source_spec -> 'schedule') = 'string'
            AND NOT (source_spec ? 'fire_at')
        WHEN 'one_shot' THEN
            jsonb_typeof(source_spec -> 'fire_at') = 'string'
            AND NOT (source_spec ? 'schedule')
        WHEN 'run_completion' THEN
            jsonb_typeof(source_spec -> 'workflow_id') = 'string'
            AND jsonb_typeof(source_spec -> 'statuses') = 'array'
            AND NOT (source_spec ? 'schedule')
            AND NOT (source_spec ? 'fire_at')
        WHEN 'external_event' THEN
            NOT (source_spec ? 'schedule')
            AND NOT (source_spec ? 'fire_at')
            AND NOT (source_spec ? 'workflow_id')
        ELSE false
    END
), false)"""

# Verbatim 0086 predicate, for downgrade only.
SOURCE_SPEC_PREDICATE_0086 = """COALESCE((
    CASE source
        WHEN 'cron' THEN
            jsonb_typeof(source_spec -> 'schedule') = 'string'
            AND NOT (source_spec ? 'fire_at')
        WHEN 'one_shot' THEN
            jsonb_typeof(source_spec -> 'fire_at') = 'string'
            AND NOT (source_spec ? 'schedule')
        WHEN 'run_completion' THEN
            jsonb_typeof(source_spec -> 'workflow_id') = 'string'
            AND jsonb_typeof(source_spec -> 'statuses') = 'array'
            AND NOT (source_spec ? 'schedule')
            AND NOT (source_spec ? 'fire_at')
        ELSE false
    END
), false)"""

# Broadened reactive-no-next-fire (run_completion AND external_event), and the
# verbatim 0086 form for downgrade.
REACTIVE_NO_NEXT_FIRE_PREDICATE = (
    "source NOT IN ('run_completion', 'external_event') OR next_fire IS NULL"
)
RUN_COMPLETION_NO_NEXT_FIRE_PREDICATE_0086 = "source <> 'run_completion' OR next_fire IS NULL"

# The ingest_token_hash iff constraint (the 0086 environment_id_iff_workflow
# idiom): external_event rows MUST carry a hash; every other kind MUST NOT.
INGEST_TOKEN_IFF_EXTERNAL_EVENT_PREDICATE = (
    "(source = 'external_event') = (ingest_token_hash IS NOT NULL)"
)


def upgrade() -> None:
    # 1. Catalog-only nullable column — zero row rewrites.
    op.execute("ALTER TABLE triggers ADD COLUMN ingest_token_hash text")

    # 2. Validating SELECT (shares predicates with ADD CONSTRAINT). No backfill;
    #    every pre-existing row satisfies the byte-identical old branches and
    #    has a NULL hash on a non-external_event source.
    bad = (
        op.get_bind()
        .execute(
            sa.text(f"""
                SELECT id, source, source_spec, ingest_token_hash, next_fire
                FROM triggers
                WHERE NOT {SOURCE_SPEC_PREDICATE}
                   OR NOT ({REACTIVE_NO_NEXT_FIRE_PREDICATE})
                   OR NOT ({INGEST_TOKEN_IFF_EXTERNAL_EVENT_PREDICATE})
                LIMIT 20
            """)
        )
        .fetchall()
    )
    if bad:
        raise RuntimeError(f"existing trigger rows violate the external_event shape: {bad!r}")

    # 3. Shape-CHECK swap (DROP + plain ADD on the capped table — one read-only
    #    validation scan, zero rewrites).
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape "
        f"CHECK ({SOURCE_SPEC_PREDICATE})"
    )

    # 4. Broaden + rename the reactive-no-next-fire guard.
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_run_completion_no_next_fire")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_reactive_no_next_fire "
        f"CHECK ({REACTIVE_NO_NEXT_FIRE_PREDICATE})"
    )

    # 5. The ingest-token iff constraint. Existing rows pass instantly (source
    #    non-external_event ⇒ left false; hash NULL ⇒ right false).
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_ingest_token_iff_external_event "
        f"CHECK ({INGEST_TOKEN_IFF_EXTERNAL_EVENT_PREDICATE})"
    )

    # 6. Unique partial index: the resolver lookup index + double-mint guard.
    op.execute(
        "CREATE UNIQUE INDEX triggers_ingest_token_hash "
        "ON triggers (ingest_token_hash) WHERE ingest_token_hash IS NOT NULL"
    )

    # 7. Extend the trigger_runs.trigger_context CHECK (0086's inline CHECK was
    #    auto-named ``trigger_runs_trigger_context_check`` by Postgres).
    op.execute("ALTER TABLE trigger_runs DROP CONSTRAINT trigger_runs_trigger_context_check")
    op.execute(
        "ALTER TABLE trigger_runs ADD CONSTRAINT trigger_runs_trigger_context_check "
        "CHECK (trigger_context IN ('cron', 'one_shot', 'run_completion', 'external_event'))"
    )


def downgrade() -> None:
    # Fail hard on any external_event row — unrepresentable under the prior
    # predicates and not reconstructible (the 0086 unrepresentable-row stance).
    n = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM triggers WHERE source = 'external_event'"))
        .scalar()
    )
    if n:
        raise RuntimeError(f"cannot downgrade: {n} external_event trigger rows")

    op.execute("ALTER TABLE trigger_runs DROP CONSTRAINT trigger_runs_trigger_context_check")
    op.execute(
        "ALTER TABLE trigger_runs ADD CONSTRAINT trigger_runs_trigger_context_check "
        "CHECK (trigger_context IN ('cron', 'one_shot', 'run_completion'))"
    )

    op.execute("DROP INDEX triggers_ingest_token_hash")
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_ingest_token_iff_external_event")

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_reactive_no_next_fire")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_run_completion_no_next_fire "
        f"CHECK ({RUN_COMPLETION_NO_NEXT_FIRE_PREDICATE_0086})"
    )

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape "
        f"CHECK ({SOURCE_SPEC_PREDICATE_0086})"
    )

    # ingest_token_hash is NULL on every remaining (non-external_event) row, so
    # the column drops cleanly.
    op.execute("ALTER TABLE triggers DROP COLUMN ingest_token_hash")
