"""Aios-side extensions to procrastinate's schema.

Procrastinate emits ``job_inserted`` NOTIFY only on INSERT, so a queued
same-session wake's lock-release transition is invisible to the worker
and pickup waits up to ``fetch_job_polling_interval`` (5s default).
This trigger fires the same NOTIFY on ``doing → !doing`` transitions
(issue #237). The 5s polling remains as a correctness backstop.
"""

from __future__ import annotations

LOCK_RELEASE_TRIGGER_DDL = """
CREATE OR REPLACE FUNCTION aios_notify_job_lock_released_v1()
    RETURNS trigger
    LANGUAGE plpgsql
AS $$
DECLARE
    payload TEXT;
BEGIN
    SELECT json_build_object('type', 'job_inserted', 'job_id', NEW.id)::text INTO payload;
    PERFORM pg_notify('procrastinate_queue_v1#' || NEW.queue_name, payload);
    PERFORM pg_notify('procrastinate_any_queue_v1', payload);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS aios_jobs_notify_lock_released_v1 ON procrastinate_jobs;

CREATE TRIGGER aios_jobs_notify_lock_released_v1
    AFTER UPDATE OF status ON procrastinate_jobs
    FOR EACH ROW
    WHEN (OLD.status = 'doing' AND NEW.status != 'doing')
    EXECUTE FUNCTION aios_notify_job_lock_released_v1();
"""
