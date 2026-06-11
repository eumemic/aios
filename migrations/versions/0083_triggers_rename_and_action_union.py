"""Rename ``session_scheduled_tasks`` → ``triggers``; ``source`` + ``action`` union.

Generalizes the per-session scheduled-task table into one ``triggers`` entity
carrying a ``source`` (what fires it: ``cron`` / ``one_shot``) and an
``action`` (what runs: ``sandbox_command`` / ``wake_owner``). Replaces the
``schedule``-vs-``fire_at`` nullable-XOR with a ``source_spec``-shape-matches-
``source`` CHECK plus a matching ``action``-shape CHECK; backfills every
existing row to ``action = sandbox_command`` (verbatim today's behavior,
including ``schedule_wake``-origin rows — ``wake_owner`` is opt-in going
forward, never backfilled).

Landmines handled (see the design contract, #818):
- The CHECK predicates are ``COALESCE((CASE…), false)``-wrapped: ``->`` on an
  absent key returns SQL NULL, a NULL CHECK *passes*, so without the COALESCE
  a row missing a required key would insert and the validating SELECT would
  be blind to it. The predicate is defined ONCE as a module constant and
  shared between the constraint and the diagnostic SELECT so they can't drift.
- The NOTIFY trigger function ``notify_scheduled_tasks_due()`` is rewritten to
  read ``source`` / ``source_spec`` / ``enabled`` / the runner-clear edge
  (the old body gated on ``schedule`` / ``fire_at``, the columns this slice
  drops — plpgsql is late-bound, so leaving it would hard-error every UPDATE
  after the drop). Channel string and function name stay byte-identical.
- The agent ``tools`` jsonb (``agents`` + ``agent_versions``) is rewritten
  ``schedule_task_* → trigger_*`` so ``ToolSpec`` validation on agent read
  doesn't reject stored configs. The ``WHERE … LIKE`` guard is load-bearing:
  ``jsonb_agg`` over an empty set returns NULL, which would null the NOT NULL
  ``tools`` column of an empty-tools agent.

No ``@dataclass`` anywhere (alembic synthetic-module load crashes on it); the
validating SELECT returns plain ``Row`` tuples. The f-strings interpolate only
module-level developer-controlled constants.

Revision ID: 0083
Revises: 0082
Create Date: 2026-06-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0083"
down_revision: str = "0082"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Shared shape predicates — defined ONCE, interpolated into both the
# ADD CONSTRAINT and the validating SELECT. The outer COALESCE(…, false) is
# load-bearing (see module docstring): it collapses the absent-key NULL to
# false so a row missing a required key is rejected, while additive extra
# keys (e.g. a future ``timezone``) still pass.
SOURCE_SPEC_PREDICATE = """COALESCE((
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
        ELSE false
    END
), false)"""


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

# The original 0059 body (for downgrade) — gates on schedule/fire_at.
_OLD_NOTIFY_FN = r"""
CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
    ELSIF (TG_OP = 'UPDATE') THEN
        IF (
            OLD.schedule IS DISTINCT FROM NEW.schedule
            OR OLD.fire_at IS DISTINCT FROM NEW.fire_at
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

# Tool-name rewrite for the agent ``tools`` jsonb (both tables share shape).
_TOOL_RENAME_SQL = """
UPDATE {table} SET tools = (
    SELECT jsonb_agg(
        CASE elem ->> 'type'
            WHEN '{old_add}'    THEN jsonb_set(elem, '{{type}}', '"{new_create}"')
            WHEN '{old_remove}' THEN jsonb_set(elem, '{{type}}', '"{new_remove}"')
            WHEN '{old_update}' THEN jsonb_set(elem, '{{type}}', '"{new_update}"')
            WHEN '{old_list}'   THEN jsonb_set(elem, '{{type}}', '"{new_list}"')
            ELSE elem
        END ORDER BY ord
    )
    FROM jsonb_array_elements(tools) WITH ORDINALITY AS t(elem, ord)
)
WHERE tools::text LIKE '%{like}%'
"""


def _rewrite_tool_names(*, forward: bool) -> None:
    """Rewrite built-in tool names inside ``agents.tools`` + ``agent_versions.tools``.

    Forward: ``schedule_task_{add,remove,update,list}`` → ``trigger_{create,
    remove,update,list}``. Reverse for downgrade. The CASE keys on ``type``,
    which for built-ins IS the model-facing name and for custom/mcp tools is
    the discriminator — so a custom tool merely *named* ``schedule_task_add``
    has ``type="custom"`` and is correctly left untouched.
    """
    if forward:
        params = {
            "old_add": "schedule_task_add",
            "old_remove": "schedule_task_remove",
            "old_update": "schedule_task_update",
            "old_list": "schedule_task_list",
            "new_create": "trigger_create",
            "new_remove": "trigger_remove",
            "new_update": "trigger_update",
            "new_list": "trigger_list",
            "like": "schedule_task_",
        }
    else:
        params = {
            "old_add": "trigger_create",
            "old_remove": "trigger_remove",
            "old_update": "trigger_update",
            "old_list": "trigger_list",
            "new_create": "schedule_task_add",
            "new_remove": "schedule_task_remove",
            "new_update": "schedule_task_update",
            "new_list": "schedule_task_list",
            "like": "trigger_",
        }
    for table in ("agents", "agent_versions"):
        op.execute(_TOOL_RENAME_SQL.format(table=table, **params))


def upgrade() -> None:
    # 1-2. Rename the table and the owner column.
    op.execute("ALTER TABLE session_scheduled_tasks RENAME TO triggers")
    op.execute("ALTER TABLE triggers RENAME COLUMN session_id TO owner_session_id")

    # 3. Hygiene renames (catalog-local; constraint/index names do NOT follow
    #    a table rename automatically).
    op.execute("ALTER INDEX sched_tasks_by_session RENAME TO triggers_by_owner_session")
    op.execute("ALTER INDEX sched_tasks_due RENAME TO triggers_due")
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_pkey TO triggers_pkey"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "session_scheduled_tasks_session_id_fkey TO triggers_owner_session_id_fkey"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "session_scheduled_tasks_session_id_name_key TO triggers_owner_session_id_name_key"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "session_scheduled_tasks_name_check TO triggers_name_check"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "session_scheduled_tasks_last_fire_status_check TO triggers_last_fire_status_check"
    )

    # 4. New columns (nullable for now; backfilled then SET NOT NULL).
    op.execute(
        "ALTER TABLE triggers "
        "ADD COLUMN source text, ADD COLUMN source_spec jsonb, ADD COLUMN action jsonb"
    )

    # 5-6. Rewrite the NOTIFY function body (new shape) and re-point the
    #      trigger. Function name + channel stay byte-identical.
    op.execute(_NEW_NOTIFY_FN)
    op.execute("DROP TRIGGER session_scheduled_tasks_notify ON triggers")
    op.execute(
        "CREATE TRIGGER triggers_notify "
        "AFTER INSERT OR UPDATE OR DELETE ON triggers "
        "FOR EACH ROW EXECUTE FUNCTION notify_scheduled_tasks_due()"
    )

    # 7. Backfill — every row → action=sandbox_command, source from which
    #    of schedule/fire_at was set (the 0059 XOR guarantees exactly one).
    op.execute("""
        UPDATE triggers SET
            source = CASE WHEN schedule IS NOT NULL THEN 'cron' ELSE 'one_shot' END,
            source_spec = CASE
                WHEN schedule IS NOT NULL THEN jsonb_build_object('schedule', schedule)
                ELSE jsonb_build_object(
                    'fire_at',
                    to_char(fire_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
                )
            END,
            action = jsonb_build_object(
                'kind', 'sandbox_command',
                'command', command,
                'timeout_seconds', timeout_seconds,
                'max_output_bytes', max_output_bytes
            )
    """)

    # 8. Validating SELECT (shares the COALESCE-wrapped predicates) — names
    #    offending rows instead of aborting on the first like ADD CONSTRAINT.
    bad = (
        op.get_bind()
        .execute(
            sa.text(f"""
                SELECT id, source, source_spec, action
                FROM triggers
                WHERE NOT {SOURCE_SPEC_PREDICATE} OR NOT {ACTION_PREDICATE}
                LIMIT 20
            """)
        )
        .fetchall()
    )
    if bad:
        raise RuntimeError(f"trigger backfill produced rows violating the shape contract: {bad!r}")

    # 9. Lock the new columns down.
    op.execute(
        "ALTER TABLE triggers "
        "ALTER COLUMN source SET NOT NULL, "
        "ALTER COLUMN source_spec SET NOT NULL, "
        "ALTER COLUMN action SET NOT NULL"
    )

    # 10. Shape CHECKs (share the predicates with step 8).
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape "
        f"CHECK ({SOURCE_SPEC_PREDICATE})"
    )
    op.execute(
        f"ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape CHECK ({ACTION_PREDICATE})"
    )

    # 11. Drop the old XOR constraint + the definition columns it guarded.
    op.execute("ALTER TABLE triggers DROP CONSTRAINT sched_tasks_schedule_xor_fire_at")
    op.execute(
        "ALTER TABLE triggers "
        "DROP COLUMN schedule, DROP COLUMN fire_at, DROP COLUMN command, "
        "DROP COLUMN timeout_seconds, DROP COLUMN max_output_bytes"
    )

    # 12. Rewrite stored agent tool names so ToolSpec validation accepts them.
    _rewrite_tool_names(forward=True)


def downgrade() -> None:
    # Fail hard on any wake_owner row — it has no command; inventing one would
    # be a silent lie. Sandbox-only data is fully reconstructible.
    n = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM triggers WHERE action->>'kind' <> 'sandbox_command'"))
        .scalar()
    )
    if n:
        raise RuntimeError(f"cannot downgrade: {n} non-sandbox_command trigger rows")

    _rewrite_tool_names(forward=False)

    op.execute(
        "ALTER TABLE triggers "
        "ADD COLUMN schedule text, ADD COLUMN fire_at timestamptz, "
        "ADD COLUMN command text, ADD COLUMN timeout_seconds integer, "
        "ADD COLUMN max_output_bytes integer"
    )
    op.execute("""
        UPDATE triggers SET
            schedule = CASE WHEN source = 'cron' THEN source_spec ->> 'schedule' END,
            fire_at = CASE
                WHEN source = 'one_shot' THEN (source_spec ->> 'fire_at')::timestamptz
            END,
            command = action ->> 'command',
            timeout_seconds = (action ->> 'timeout_seconds')::int,
            max_output_bytes = (action ->> 'max_output_bytes')::int
    """)
    op.execute(
        "ALTER TABLE triggers "
        "ALTER COLUMN command SET NOT NULL, "
        "ALTER COLUMN timeout_seconds SET NOT NULL, "
        "ALTER COLUMN max_output_bytes SET NOT NULL"
    )
    op.execute("""
        ALTER TABLE triggers
        ADD CONSTRAINT sched_tasks_schedule_xor_fire_at
        CHECK (
            (schedule IS NOT NULL AND fire_at IS NULL)
            OR (schedule IS NULL AND fire_at IS NOT NULL)
        )
    """)

    # Restore the old NOTIFY body (gates on schedule/fire_at).
    op.execute(_OLD_NOTIFY_FN)

    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_source_spec_shape")
    op.execute("ALTER TABLE triggers DROP CONSTRAINT triggers_action_shape")
    op.execute(
        "ALTER TABLE triggers "
        "DROP COLUMN source, DROP COLUMN source_spec, DROP COLUMN action"
    )

    # Reverse the trigger + hygiene renames + table/column renames.
    op.execute("DROP TRIGGER triggers_notify ON triggers")
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "triggers_last_fire_status_check TO session_scheduled_tasks_last_fire_status_check"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "triggers_name_check TO session_scheduled_tasks_name_check"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "triggers_owner_session_id_name_key TO session_scheduled_tasks_session_id_name_key"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT "
        "triggers_owner_session_id_fkey TO session_scheduled_tasks_session_id_fkey"
    )
    op.execute(
        "ALTER TABLE triggers RENAME CONSTRAINT triggers_pkey TO session_scheduled_tasks_pkey"
    )
    op.execute("ALTER INDEX triggers_due RENAME TO sched_tasks_due")
    op.execute("ALTER INDEX triggers_by_owner_session RENAME TO sched_tasks_by_session")
    op.execute("ALTER TABLE triggers RENAME COLUMN owner_session_id TO session_id")
    op.execute("ALTER TABLE triggers RENAME TO session_scheduled_tasks")
    op.execute(
        "CREATE TRIGGER session_scheduled_tasks_notify "
        "AFTER INSERT OR UPDATE OR DELETE ON session_scheduled_tasks "
        "FOR EACH ROW EXECUTE FUNCTION notify_scheduled_tasks_due()"
    )
