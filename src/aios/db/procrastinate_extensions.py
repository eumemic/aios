"""Aios-side extensions to procrastinate's schema.

Procrastinate emits ``job_inserted`` NOTIFY only on INSERT, so a queued
same-session wake's lock-release transition is invisible to the worker
and pickup waits up to ``fetch_job_polling_interval`` (5s default).
This trigger fires the same NOTIFY on ``doing → !doing`` transitions
(issue #237). The 5s polling remains as a correctness backstop.

The DDL takes an AccessExclusive lock on ``procrastinate_jobs`` — the
hottest table in the system — so ``aios migrate`` only runs it when the
installed function/trigger differ from the definitions below (compared via
``pg_get_functiondef`` / ``pg_get_triggerdef``, which take no table lock),
and then only under a bounded ``lock_timeout`` (issue #2539).
"""

from __future__ import annotations

LOCK_RELEASE_FUNCTION_NAME = "aios_notify_job_lock_released_v1"
LOCK_RELEASE_TRIGGER_NAME = "aios_jobs_notify_lock_released_v1"

# The plpgsql body exactly as stored in ``pg_proc.prosrc`` (everything between
# the dollar quotes, including the leading/trailing newlines).
_LOCK_RELEASE_FUNCTION_BODY = """
DECLARE
    payload TEXT;
BEGIN
    SELECT json_build_object('type', 'job_inserted', 'job_id', NEW.id)::text INTO payload;
    PERFORM pg_notify('procrastinate_queue_v1#' || NEW.queue_name, payload);
    PERFORM pg_notify('procrastinate_any_queue_v1', payload);
    RETURN NEW;
END;
"""

LOCK_RELEASE_TRIGGER_DDL = f"""
CREATE OR REPLACE FUNCTION {LOCK_RELEASE_FUNCTION_NAME}()
    RETURNS trigger
    LANGUAGE plpgsql
AS $${_LOCK_RELEASE_FUNCTION_BODY}$$;

DROP TRIGGER IF EXISTS {LOCK_RELEASE_TRIGGER_NAME} ON procrastinate_jobs;

CREATE TRIGGER {LOCK_RELEASE_TRIGGER_NAME}
    AFTER UPDATE OF status ON procrastinate_jobs
    FOR EACH ROW
    WHEN (OLD.status = 'doing' AND NEW.status != 'doing')
    EXECUTE FUNCTION {LOCK_RELEASE_FUNCTION_NAME}();
"""

# Single catalog read describing what is installed.  Touches only
# pg_proc / pg_trigger / pg_namespace (no lock on procrastinate_jobs), so it
# returns immediately even while e.g. pg_dump holds AccessShare on the table.
LOCK_RELEASE_STATE_SQL = f"""
SELECT
    (SELECT pg_get_functiondef(p.oid)
       FROM pg_proc p
      WHERE p.oid = to_regprocedure('{LOCK_RELEASE_FUNCTION_NAME}()')) AS functiondef,
    (SELECT pg_get_triggerdef(t.oid)
       FROM pg_trigger t
      WHERE t.tgrelid = to_regclass('procrastinate_jobs')
        AND t.tgname = '{LOCK_RELEASE_TRIGGER_NAME}'
        AND t.tgenabled = 'O') AS triggerdef,
    (SELECT n.nspname
       FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
      WHERE p.oid = to_regprocedure('{LOCK_RELEASE_FUNCTION_NAME}()')) AS function_schema,
    (SELECT n.nspname
       FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE c.oid = to_regclass('procrastinate_jobs')) AS table_schema
"""


def _quote_ident(name: str) -> str:
    if name.isidentifier() and name == name.lower():
        return name
    return '"' + name.replace('"', '""') + '"'


def expected_functiondef(schema: str) -> str:
    """``pg_get_functiondef`` output for the function installed by the DDL."""
    return (
        f"CREATE OR REPLACE FUNCTION {_quote_ident(schema)}.{LOCK_RELEASE_FUNCTION_NAME}()\n"
        " RETURNS trigger\n"
        " LANGUAGE plpgsql\n"
        f"AS $function${_LOCK_RELEASE_FUNCTION_BODY}$function$\n"
    )


def expected_triggerdef(table_schema: str) -> str:
    """``pg_get_triggerdef`` output for the trigger installed by the DDL.

    The function is resolved through ``search_path`` at DDL time, so it is
    printed unqualified (it lives in the same schema as the table).
    """
    return (
        f"CREATE TRIGGER {LOCK_RELEASE_TRIGGER_NAME} AFTER UPDATE OF status ON "
        f"{_quote_ident(table_schema)}.procrastinate_jobs FOR EACH ROW WHEN "
        "(((old.status = 'doing'::procrastinate_job_status) AND "
        "(new.status <> 'doing'::procrastinate_job_status))) "
        f"EXECUTE FUNCTION {LOCK_RELEASE_FUNCTION_NAME}()"
    )


def lock_release_trigger_is_current(
    *,
    functiondef: str | None,
    triggerdef: str | None,
    function_schema: str | None,
    table_schema: str | None,
) -> bool:
    """True iff the installed function and trigger match the DDL exactly.

    Any missing piece or textual drift returns False, which makes the caller
    (re-)apply the DDL under a bounded ``lock_timeout`` — the safe direction.
    """
    if functiondef is None or triggerdef is None or function_schema is None:
        return False
    if table_schema is None or function_schema != table_schema:
        return False
    return functiondef == expected_functiondef(function_schema) and (
        triggerdef == expected_triggerdef(table_schema)
    )
