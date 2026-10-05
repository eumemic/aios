"""Operator-owned triggers (#2473).

Revision ID: 0190
Revises: 0189

Every trigger so far belongs to a session: ``owner_session_id`` is NOT NULL and
every fire launches its run with that session's authority. A trigger an
operator owns has no session, and its workflow runs are operator runs (the
#2476 weekly eval monitor needs that, since replay is operator-only).

- ``triggers.owner_kind`` (``session`` | ``operator``) says which kind of owner
  a row has. It is the kind, never "a NULL owner means operator": identity read
  off a nullable id is what #2467 had to undo. The constant default makes the
  add catalog-only, and an application image from before this revision inserts
  only session triggers, which the default describes correctly.
- ``triggers.owner_session_id`` becomes nullable. Its FK keeps ON DELETE
  CASCADE, so deleting a session still deletes its triggers and never nulls
  them.
- ``triggers_owner_kind_shape`` ties the kind to the row: a session trigger
  has an owner session; an operator trigger has none, fires on a timer
  (``cron`` | ``one_shot``), launches a workflow with a numeric
  ``budget_usd``, binds an environment, and holds no ingest token. An event
  source would let whoever causes the event start an operator run. The
  predicate is COALESCE-wrapped (the 0083 lesson: a NULL CHECK passes) and is
  shared with the validating SELECT that runs before the constraint.
- ``triggers_operator_name`` makes operator trigger names unique per account.
  The existing ``UNIQUE (owner_session_id, name)`` treats NULLs as distinct.
- ``trigger_runs.owner_session_id`` becomes nullable: a NULL is an operator
  trigger's fire. The column is a plain id with no FK, so nothing else ever
  nulls it.
- ``triggers.account_id`` gets the FK it never had, ON DELETE CASCADE like
  ``trigger_runs`` (0086). Until now a trigger was reachable only through a
  session, and sessions RESTRICT an account purge; an operator trigger would
  otherwise outlive its purged account and keep firing. Added NOT VALID and
  then validated, after a SELECT that names any orphan.

An old image never sees an operator row: every claim and read query it has
inner-joins ``sessions``. Under an application rollback operator triggers
pause until the roll-forward.

No ``@dataclass`` here (alembic's synthetic-module load crashes on it).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0190"
down_revision: str | None = "0189"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OWNER_KIND_SHAPE_PREDICATE = """COALESCE((
    CASE owner_kind
        WHEN 'session' THEN owner_session_id IS NOT NULL
        WHEN 'operator' THEN
            owner_session_id IS NULL
            AND source IN ('cron', 'one_shot')
            AND action ->> 'kind' = 'workflow'
            AND jsonb_typeof(action -> 'budget_usd') = 'number'
            AND ingest_token_hash IS NULL
            AND environment_id IS NOT NULL
        ELSE false
    END
), false)"""


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE triggers ADD COLUMN owner_kind text NOT NULL DEFAULT 'session'")
    op.execute("ALTER TABLE triggers ALTER COLUMN owner_session_id DROP NOT NULL")

    bind = op.get_bind()
    bad = bind.execute(
        sa.text(f"""
            SELECT id, owner_kind, owner_session_id
            FROM triggers
            WHERE NOT {OWNER_KIND_SHAPE_PREDICATE}
            LIMIT 20
        """)
    ).fetchall()
    if bad:
        raise RuntimeError(f"triggers violating the owner-kind shape: {bad!r}")
    op.execute(
        "ALTER TABLE triggers ADD CONSTRAINT triggers_owner_kind_shape "
        f"CHECK ({OWNER_KIND_SHAPE_PREDICATE})"
    )
    op.execute(
        "CREATE UNIQUE INDEX triggers_operator_name ON triggers (account_id, name) "
        "WHERE owner_kind = 'operator'"
    )

    op.execute("ALTER TABLE trigger_runs ALTER COLUMN owner_session_id DROP NOT NULL")

    orphans = bind.execute(
        sa.text("""
            SELECT t.id, t.account_id
            FROM triggers AS t
            WHERE NOT EXISTS (SELECT 1 FROM accounts AS a WHERE a.id = t.account_id)
            LIMIT 20
        """)
    ).fetchall()
    if orphans:
        raise RuntimeError(f"triggers whose account no longer exists: {orphans!r}")
    op.execute(
        "ALTER TABLE triggers ADD CONSTRAINT triggers_account_id_fkey "
        "FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE NOT VALID"
    )
    op.execute("ALTER TABLE triggers VALIDATE CONSTRAINT triggers_account_id_fkey")


def downgrade() -> None:
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_account_id_fkey")
    op.execute("DELETE FROM trigger_runs WHERE owner_session_id IS NULL")
    op.execute("ALTER TABLE trigger_runs ALTER COLUMN owner_session_id SET NOT NULL")
    op.execute("DROP INDEX triggers_operator_name")
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_owner_kind_shape")
    op.execute("DELETE FROM triggers WHERE owner_kind = 'operator'")
    op.execute("ALTER TABLE triggers ALTER COLUMN owner_session_id SET NOT NULL")
    op.execute("ALTER TABLE triggers DROP COLUMN owner_kind")
