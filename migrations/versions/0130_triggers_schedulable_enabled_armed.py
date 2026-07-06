"""Triggers: make the #925 zombie state unrepresentable (#1678).

The missing half of the 0108 schema symmetry. Migration 0108 pinned the
*reactive* arm (``triggers_reactive_no_next_fire``:
``source NOT IN ('run_completion','external_event') OR next_fire IS NULL``);
this adds the *schedulable* arm:

    triggers_schedulable_enabled_armed:
        NOT enabled OR source NOT IN ('cron','one_shot') OR next_fire IS NOT NULL

i.e. "an enabled schedulable (cron / one_shot) trigger always has a non-NULL
``next_fire``". The scheduler's claim/MIN queries both filter
``next_fire IS NOT NULL`` (``fetch_and_claim_due_triggers`` /
``fetch_next_trigger_event``), so an enabled schedulable row with NULL
``next_fire`` is a permanently dead row that looks alive — the #925 incident
state, produced by a manual ``UPDATE … SET enabled=true``. This CHECK makes
that state unwritable for every writer alike (service code, model tool, claim
advance, auto-disable, session clone, AND manual psql), which lets the #957
runtime heal in ``services.triggers.update_trigger`` be deleted in the same PR
(under the CHECK its ``current.enabled ∧ next_fire IS NULL`` disjunct is
provably dead for cron rows).

``upgrade()``:

1. Behavioral no-op cleanup: DISABLE every existing violator row
   (``enabled AND source IN ('cron','one_shot') AND next_fire IS NULL``)
   BEFORE the validating ``ADD CONSTRAINT`` — the 0066 lesson: a plain
   ``ADD CONSTRAINT`` validation scan fails hard on any pre-existing violator,
   so they must be cleaned first or the prod migrate aborts. These rows never
   fired (the claim filter excluded them), so disabling is behaviorally a
   no-op; the disable is LOGGED (per the build-guidance review — a silent
   disable inside a fix that targets silent failures would be un-auditable),
   and each disabled row needs an explicit re-enable to schedule again.
   We DISABLE rather than arm: arming to ``now()`` would surprise-fire dead
   rows on deploy, and computing the proper cron slot needs croniter inside a
   migration (importing app code — against migration conventions).

2. ``ADD CONSTRAINT triggers_schedulable_enabled_armed`` — a brief ACCESS
   EXCLUSIVE lock + one read-only validation scan; negligible on the
   per-account-capped triggers table, zero row rewrites.

``downgrade()`` drops the CHECK. No data reconstruction is needed — the
constraint only ever *rejected* writes, it never rewrote a row.

Revision ID: 0130
Revises: 0129
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0130"
down_revision: str = "0129"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

# "enabled schedulable ⇒ armed" — the schedulable complement of 0108's
# reactive ``triggers_reactive_no_next_fire``. Kept as SQL literals to mirror
# the 0108 CHECK's own (un-drift-pinned) literals; the schedulable source set
# is ('cron', 'one_shot').
SCHEDULABLE_ENABLED_ARMED_PREDICATE = (
    "NOT enabled OR source NOT IN ('cron', 'one_shot') OR next_fire IS NOT NULL"
)


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Disable pre-existing violators BEFORE the validation scan (0066
    #    lesson). RETURNING so the auto-disable is auditable — a silent disable
    #    inside a fix aimed at silent failures would be ironic (#1678 review).
    disabled = bind.execute(
        sa.text(
            """
            UPDATE triggers
               SET enabled = false
             WHERE enabled
               AND source IN ('cron', 'one_shot')
               AND next_fire IS NULL
            RETURNING id, owner_session_id, account_id, name, source
            """
        )
    ).fetchall()
    if disabled:
        logger.warning(
            "0130: auto-disabled %d enabled-but-unarmed schedulable trigger row(s) "
            "(the #925 zombie state; behaviorally a no-op — these rows never fired). "
            "Each needs an explicit re-enable to schedule again: %r",
            len(disabled),
            [tuple(r) for r in disabled],
        )

    # 2. The schedulable arm — a brief ACCESS EXCLUSIVE lock + one read-only
    #    validation scan; the violator rows are already disabled, so it passes.
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_schedulable_enabled_armed "
        f"CHECK ({SCHEDULABLE_ENABLED_ARMED_PREDICATE})"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_schedulable_enabled_armed")
