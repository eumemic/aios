"""Triggers: ``wake_session`` action kind (#1280) — explicit-target async wake.

Additive CHECK swap, zero row rewrites — the growth rule's "a new behavior is a
new action *kind*, never a flag" applied verbatim. Follows the 0086 idiom
(slice-2 ``workflow`` action): the ``triggers_action_shape`` predicate gains one
new ``WHEN 'wake_session'`` branch; every pre-existing branch stays
BYTE-IDENTICAL (a unit test pins this — see
tests/unit/test_migration_0086_predicates.py extended for 0107).

- ``triggers_action_shape`` swap (DROP + plain ADD on the capped table): the
  new branch asserts ``target_session_id`` and ``content`` are strings and the
  row carries no ``command`` key. ``wake_session`` rows carry NO
  ``environment_id`` (left-side false = right-side false), so the sibling
  ``triggers_environment_id_iff_workflow`` constraint is UNCHANGED.
- A validating SELECT (names offending rows; diagnostic — no backfill here,
  every pre-existing row satisfies the byte-identical old branches).
- ``downgrade()`` fails hard on any ``action ->> 'kind' = 'wake_session'`` row
  (unrepresentable + not reconstructible under the prior predicate, the 0086
  stance) and restores the prior ``ACTION_PREDICATE`` verbatim-embedded (no
  cross-migration import — migrations load under synthetic module names and
  cannot import each other).

Revision ID: 0107
Revises: 0106
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0107"
down_revision: str = "0106"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The new predicate: 0086's ACTION_PREDICATE with one ``WHEN 'wake_session'``
# branch spliced in before ``ELSE false``. The four pre-existing branches are
# byte-identical to 0086's (the byte-identity unit test enforces this).
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
        WHEN 'wake_session' THEN
            jsonb_typeof(action -> 'target_session_id') = 'string'
            AND jsonb_typeof(action -> 'content') = 'string'
            AND NOT (action ? 'command')
        ELSE false
    END
), false)"""

# Verbatim 0086 ACTION_PREDICATE (the head this migration extends), embedded for
# downgrade only — no cross-migration import.
ACTION_PREDICATE_0086 = """COALESCE((
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


def upgrade() -> None:
    # Validating SELECT (shares ACTION_PREDICATE with ADD CONSTRAINT) — names
    # offending rows instead of aborting on the first. No backfill here, so this
    # is expected empty: every pre-existing row satisfies the byte-identical old
    # branches inside the new predicate.
    bad = (
        op.get_bind()
        .execute(
            sa.text(f"""
                SELECT id, action
                FROM triggers
                WHERE NOT {ACTION_PREDICATE}
                LIMIT 20
            """)
        )
        .fetchall()
    )
    if bad:
        raise RuntimeError(
            f"existing trigger rows violate the wake_session action shape contract: {bad!r}"
        )

    # DROP + plain ADD (0086 precedent on the capped table). One read-only
    # validation scan; alembic holds the lock to commit either way, so
    # NOT VALID + VALIDATE buys nothing here.
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape CHECK ({ACTION_PREDICATE})"
    )


def downgrade() -> None:
    # Fail hard on any wake_session row — unrepresentable under the prior
    # predicate and not reconstructible (the 0086 wake_owner/workflow stance).
    n = (
        op.get_bind()
        .execute(
            sa.text("SELECT count(*) FROM triggers WHERE action ->> 'kind' = 'wake_session'")
        )
        .scalar()
    )
    if n:
        raise RuntimeError(f"cannot downgrade: {n} wake_session trigger rows")

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape CHECK ({ACTION_PREDICATE_0086})"
    )
