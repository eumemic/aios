"""NOTIFY ``aios_scheduled_tasks_due`` on a bare ``next_fire`` change (#940).

The event-driven scheduler (:mod:`aios.harness.scheduler`) sleeps until the
next due ``next_fire`` instant, woken early by a NOTIFY on
``aios_scheduled_tasks_due`` or the cold-path heartbeat cap. Pre-0086 the
trigger function ``notify_scheduled_tasks_due()`` gated its UPDATE branch on
source / source_spec / enabled / running_since→NULL changes only — NOT
``next_fire``. Any path that rescheduled a row by writing ``next_fire`` alone
(without touching those columns) left the sleeping scheduler unaware for up
to ``_HEARTBEAT_SECONDS``.

Additive fix: ``CREATE OR REPLACE`` the function body, adding exactly one
clause ``OR OLD.next_fire IS DISTINCT FROM NEW.next_fire`` to the UPDATE gate.
The function name and channel string stay byte-identical, and the existing
``triggers_notify`` trigger keeps pointing at the replaced body — no trigger
recreate needed. Backward-compatible: pre-deploy code that already NOTIFYs on
the gated columns is unaffected; this only widens when a NOTIFY is emitted.

Revision ID: 0086
Revises: 0085
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0086"
down_revision: str = "0085"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# New body — the 0083 source/source_spec/enabled/running_since gate plus the
# ``next_fire`` clause (#940). Function name + channel byte-identical to 0083.
_NEW_NOTIFY_FN = r"""
CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
    ELSIF (TG_OP = 'UPDATE') THEN
        IF (
            OLD.source IS DISTINCT FROM NEW.source
            OR OLD.source_spec IS DISTINCT FROM NEW.source_spec
            OR OLD.enabled IS DISTINCT FROM NEW.enabled
            OR OLD.next_fire IS DISTINCT FROM NEW.next_fire
            OR (OLD.running_since IS NOT NULL AND NEW.running_since IS NULL)
        ) THEN
            PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
        END IF;
    ELSIF (TG_OP = 'DELETE') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', OLD.id);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
"""

# Previous body (for downgrade) — the 0083 ``_NEW_NOTIFY_FN`` verbatim, gating
# on source/source_spec/enabled/running_since with NO ``next_fire`` clause.
# Named ``_PREV_*`` to distinguish it from 0083's own ``_OLD_NOTIFY_FN`` (the
# pre-rename schedule/fire_at body), which is not what we restore here.
_PREV_NOTIFY_FN = r"""
CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
    ELSIF (TG_OP = 'UPDATE') THEN
        IF (
            OLD.source IS DISTINCT FROM NEW.source
            OR OLD.source_spec IS DISTINCT FROM NEW.source_spec
            OR OLD.enabled IS DISTINCT FROM NEW.enabled
            OR (OLD.running_since IS NOT NULL AND NEW.running_since IS NULL)
        ) THEN
            PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
        END IF;
    ELSIF (TG_OP = 'DELETE') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', OLD.id);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.execute(_NEW_NOTIFY_FN)


def downgrade() -> None:
    op.execute(_PREV_NOTIFY_FN)
