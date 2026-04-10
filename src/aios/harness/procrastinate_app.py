"""Procrastinate App singleton.

Phase 2 introduces a job queue substrate. Procrastinate uses psycopg3 (its own
``PsycopgConnector``), independent from our ``asyncpg`` pool. Both run in the
same process — they don't share connections, just the same Postgres database.

Both the api process (which calls ``defer_async`` to enqueue ``wake_session``
jobs) and the worker process (which actually runs the jobs) import this module
and use the same ``app`` singleton.

The factory imports ``aios.harness.tasks`` at the bottom so the
``@app.task(...)`` decorators register against this exact ``app`` instance.
This is the standard procrastinate idiom for resolving the circular import
between the app module and the tasks module.
"""

from __future__ import annotations

from procrastinate import App, PsycopgConnector

from aios.config import get_settings


def _sync_dsn(db_url: str) -> str:
    """Strip any +async driver suffix from a Postgres DSN.

    psycopg3 accepts the bare ``postgresql://`` form. ``postgresql+psycopg://``
    and ``postgresql+asyncpg://`` are SQLAlchemy/alembic conventions that need
    to be stripped before passing to psycopg.
    """
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


def _build_connector() -> PsycopgConnector:
    settings = get_settings()
    return PsycopgConnector(conninfo=_sync_dsn(settings.db_url))


# Module-level singleton. Constructed eagerly at import time so that any
# `from aios.harness.procrastinate_app import app` line in `tasks.py` can
# resolve immediately. The tasks module imports happen AFTER this assignment
# (see the import at the bottom of the file), which is the standard
# procrastinate idiom for breaking the circular import.
app: App = App(connector=_build_connector())

# Register tasks. This MUST come after `app` is assigned so that
# `from aios.harness.procrastinate_app import app` inside tasks.py resolves.
from aios.harness import tasks  # noqa: E402, F401  -- side-effecting registration
