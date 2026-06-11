"""Triggers slice 2 (#819): ``workflow`` action, ``run_completion`` source, ``trigger_runs``.

Extends the slice-1 envelope (migration 0083, design contract 0818) ADDITIVELY —
zero rewrites of existing trigger rows:

- ``triggers.environment_id`` — nullable FK → ``environments(id)`` (bare
  REFERENCES, no ON DELETE — the 0067 ``wf_runs.environment_id`` precedent).
  Set iff the action kind is ``workflow``; the contract's §1.1 FK exception
  ("a per-kind presence/absence conjunct in the action-CHECK swap") is honored
  by the SIBLING constraint ``triggers_environment_id_iff_workflow`` rather
  than conjuncts inside the action CASE — same migration, zero rewrites, both
  directions in one expression — so the existing CHECK branches stay
  byte-identical (see the asymmetry note below).
- Shape-CHECK swap: ``triggers_source_spec_shape`` gains the ``run_completion``
  branch (watched ``workflow_id`` + a ``statuses`` array, both required —
  first-shipped shape, so they join the CHECK); ``triggers_action_shape``
  gains the ``workflow`` branch (``workflow_id`` string; the §1.1
  required-but-nullable ``workflow_version`` idiom; ``input_template``
  presence-only — every JSON type incl. json null is legal; ``vault_ids``
  array). The cron/one_shot/sandbox_command/wake_owner branches are
  BYTE-IDENTICAL to 0083's (enforced by a unit test); they deliberately do NOT
  gain ``NOT (action ? 'workflow_id')``-style exclusions of the new kind's
  keys — the key set is contractually open and ``extra="forbid"`` write models
  make stray keys unreachable.
- ``triggers_run_completion_no_next_fire`` — DB-enforces the §3 invariant that
  reactive rows are unschedulable by the scheduler tick. A service-layer slip
  here would not be one mis-fire but a tick-speed hot re-claim runaway
  (``next_fire`` never advances for non-cron sources and ok-status fires never
  trip auto-disable), hence the constraint.
- ``trigger_runs`` — one audit row per fire AND the per-event claim/dedup
  carrier for run_completion dispatch (contract 0818 §9 D3). ``trigger_id`` /
  ``owner_session_id`` are PLAIN ids, deliberately NO FK: one-shot triggers
  DELETE-BEFORE-FIRE and sessions CASCADE-delete triggers — an enforcing FK
  would eat or block the audit, which must outlive its trigger. ``account_id``
  IS an FK (ON DELETE CASCADE): nothing else reaps these rows. 'manual' stays
  vocabulary-only (not in the context CHECK) until a fire-now endpoint wires it.

The validating SELECT mirrors 0083's craft: it shares the predicate constants
with ADD CONSTRAINT so the two cannot drift, and names offending rows instead
of aborting on the first. With no backfill in this migration it is purely
diagnostic — every pre-existing row satisfies the old branches, which are
byte-identical inside the new predicates.

The ``*_0083`` constants are the verbatim previous predicates, embedded for
``downgrade()`` — migrations load under synthetic module names and cannot
import each other (the 0083 ``_OLD_NOTIFY_FN`` pattern). No ``@dataclass``
anywhere (alembic synthetic-module crash). ``notify_scheduled_tasks_due()`` is
NOT rewritten: its UPDATE gate compares ``source``/``source_spec``/``enabled``/
the running_since-clear edge value-generically, so the new kinds flow through;
a ``run_completion`` INSERT costs one harmless scheduler MIN recompute.

Revision ID: 0085
Revises: 0084
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0085"
down_revision: str = "0084"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Shared shape predicates — defined ONCE, interpolated into both ADD CONSTRAINT
# and the validating SELECT (0083 idiom). The outer COALESCE(…, false) is
# load-bearing: ``->`` on an absent key returns SQL NULL, a NULL CHECK passes,
# so without it a row missing a required key would insert undetected.
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
        ELSE false
    END
), false)"""

ACTION_PREDICATE = """COALESCE((
    CASE action ->> 'kind'
        WHEN 'sandbox_command' THEN
            jsonb_typeof(action -> 'command') = 'string'
            AND jsonb_typeof(action -> 'timeout_seconds') = 'number'
            AND jsonb_typeof(action -> 'max_output_bytes') = 'number'
            AND NOT (action ? 'content')
        WHEN 'wake_owner' THEN
            jsonb_typeof(action -> 'content') = 'string'
            AND NOT (action ? 'command')
        WHEN 'workflow' THEN
            jsonb_typeof(action -> 'workflow_id') = 'string'
            AND (action ? 'workflow_version')
            AND jsonb_typeof(action -> 'workflow_version') IN ('number', 'null')
            AND (action ? 'input_template')
            AND jsonb_typeof(action -> 'vault_ids') = 'array'
            AND NOT (action ? 'environment_id')
            AND NOT (action ? 'command')
            AND NOT (action ? 'content')
        ELSE false
    END
), false)"""

# Contract 0818 §1.1's per-kind presence/absence conjunct for the FK column, as
# its own constraint so the action predicate's existing branches stay
# byte-identical. Boolean equality covers both directions: workflow rows MUST
# carry environment_id; every other kind MUST NOT. (A missing 'kind' makes this
# NULL → CHECK-satisfied, but triggers_action_shape's ELSE false already
# rejects that row.) Slice 4's spawn_session swaps this to
# ((action ->> 'kind') IN ('workflow','spawn_session')) = (environment_id IS NOT NULL).
ENVIRONMENT_ID_IFF_WORKFLOW_PREDICATE = (
    "(action ->> 'kind' = 'workflow') = (environment_id IS NOT NULL)"
)

# §3 of the slice-1 contract makes a NULL next_fire row unschedulable by the
# tick BY PREDICATE; this makes the converse unrepresentable — a run_completion
# row can never carry a next_fire and so can never be tick-claimed.
RUN_COMPLETION_NO_NEXT_FIRE_PREDICATE = "source <> 'run_completion' OR next_fire IS NULL"

# Verbatim 0083 predicates, for downgrade only.
SOURCE_SPEC_PREDICATE_0083 = """COALESCE((
    CASE source
        WHEN 'cron' THEN
            jsonb_typeof(source_spec -> 'schedule') = 'string'
            AND NOT (source_spec ? 'fire_at')
        WHEN 'one_shot' THEN
            jsonb_typeof(source_spec -> 'fire_at') = 'string'
            AND NOT (source_spec ? 'schedule')
        ELSE false
    END
), false)"""

ACTION_PREDICATE_0083 = """COALESCE((
    CASE action ->> 'kind'
        WHEN 'sandbox_command' THEN
            jsonb_typeof(action -> 'command') = 'string'
            AND jsonb_typeof(action -> 'timeout_seconds') = 'number'
            AND jsonb_typeof(action -> 'max_output_bytes') = 'number'
            AND NOT (action ? 'content')
        WHEN 'wake_owner' THEN
            jsonb_typeof(action -> 'content') = 'string'
            AND NOT (action ? 'command')
        ELSE false
    END
), false)"""


def upgrade() -> None:
    # 1. Nullable FK column (0067 precedent: bare REFERENCES, no ON DELETE, no
    #    index — environments are archive-only today, so the RESTRICT is latent
    #    unless a hard-delete path ever ships). ADD COLUMN with no default is
    #    catalog-only: zero row rewrites.
    op.execute("ALTER TABLE triggers ADD COLUMN environment_id text REFERENCES environments(id)")

    # 2. Validating SELECT (shares the predicate constants with ADD CONSTRAINT)
    #    — diagnostic: names offending rows instead of aborting on the first.
    #    No backfill happens in this migration, so this is expected to be empty;
    #    every pre-existing row satisfies the byte-identical old branches.
    bad = (
        op.get_bind()
        .execute(
            sa.text(f"""
                SELECT id, source, source_spec, action
                FROM triggers
                WHERE NOT {SOURCE_SPEC_PREDICATE}
                   OR NOT {ACTION_PREDICATE}
                   OR NOT ({ENVIRONMENT_ID_IFF_WORKFLOW_PREDICATE})
                   OR NOT ({RUN_COMPLETION_NO_NEXT_FIRE_PREDICATE})
                LIMIT 20
            """)
        )
        .fetchall()
    )
    if bad:
        raise RuntimeError(f"existing trigger rows violate the slice-2 shape contract: {bad!r}")

    # 3. Shape-CHECK swap: DROP + plain ADD (0083 precedent). One read-only
    #    validation scan each, zero rewrites; alembic holds the lock to commit
    #    either way, so NOT VALID + VALIDATE buys nothing on this capped table.
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape "
        f"CHECK ({SOURCE_SPEC_PREDICATE})"
    )
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape CHECK ({ACTION_PREDICATE})"
    )

    # 4. The two new sibling constraints. Existing rows pass instantly:
    #    kind ∈ {sandbox_command, wake_owner} (left false) and environment_id
    #    NULL (right false) → false = false → true; no row has
    #    source='run_completion' yet.
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_environment_id_iff_workflow "
        f"CHECK ({ENVIRONMENT_ID_IFF_WORKFLOW_PREDICATE})"
    )
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_run_completion_no_next_fire "
        f"CHECK ({RUN_COMPLETION_NO_NEXT_FIRE_PREDICATE})"
    )

    # 5. The per-fire audit table — also the run_completion dispatch carrier.
    op.execute("""
        CREATE TABLE trigger_runs (
            id               text PRIMARY KEY,
            trigger_id       text NOT NULL,
            account_id       text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            owner_session_id text NOT NULL,
            trigger_name     text NOT NULL,
            trigger_context  text NOT NULL
                CHECK (trigger_context IN ('cron', 'one_shot', 'run_completion')),
            event            jsonb,
            status           text NOT NULL
                CHECK (status IN ('pending', 'running', 'ok', 'error', 'timeout', 'skipped')),
            result_id        text,
            error_summary    text,
            created_at       timestamptz NOT NULL DEFAULT now(),
            started_at       timestamptz,
            finished_at      timestamptz
        )
    """)
    op.execute("CREATE INDEX trigger_runs_by_trigger ON trigger_runs (trigger_id, created_at DESC)")
    # The sweep's scan set: pending = lost defer (re-defer); running = crashed
    # mid-fire (counted + warned, deliberately never retried).
    op.execute(
        "CREATE INDEX trigger_runs_unfinished ON trigger_runs (created_at) "
        "WHERE status IN ('pending', 'running')"
    )

    # 6. The completion matcher's index (account-scoped expression match on the
    #    watched workflow).
    op.execute(
        "CREATE INDEX triggers_run_completion_watch "
        "ON triggers (account_id, (source_spec ->> 'workflow_id')) "
        "WHERE source = 'run_completion'"
    )


def downgrade() -> None:
    # Fail hard on any slice-2 row — unrepresentable under the 0083 predicates
    # and not reconstructible (0083's wake_owner stance). Under the iff
    # constraint, environment_id NOT NULL ⇒ kind='workflow', so this one count
    # subsumes the column check.
    n = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM triggers "
                "WHERE source = 'run_completion' OR action ->> 'kind' = 'workflow'"
            )
        )
        .scalar()
    )
    if n:
        raise RuntimeError(f"cannot downgrade: {n} run_completion/workflow trigger rows")

    op.execute("DROP INDEX triggers_run_completion_watch")
    op.execute("DROP TABLE trigger_runs")

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_run_completion_no_next_fire")
    # Explicit for symmetry with upgrade(); Postgres would auto-drop this table
    # constraint with the column (table constraints involving a dropped column
    # are dropped automatically — CASCADE is only needed for outside deps).
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_environment_id_iff_workflow")
    op.execute("ALTER TABLE triggers DROP COLUMN environment_id")

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape "
        f"CHECK ({SOURCE_SPEC_PREDICATE_0083})"
    )
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape "
        f"CHECK ({ACTION_PREDICATE_0083})"
    )
