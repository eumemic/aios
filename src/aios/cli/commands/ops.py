"""Operator subcommands: ``api``, ``worker``, ``migrate``.

These are lifted almost verbatim from the old ``__main__.py`` implementation.
They start long-running processes (uvicorn, procrastinate worker) or run
migrations — they do NOT talk to the HTTP API.
"""

from __future__ import annotations

import asyncio

import typer


def _run_api() -> int:
    import uvicorn

    from aios.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "aios.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # we configure structlog ourselves
    )
    return 0


def _run_worker() -> int:
    from aios.harness.worker import worker_main
    from aios.logging import get_logger

    try:
        asyncio.run(worker_main())
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except BaseException:
        get_logger("aios.worker").exception("worker.unexpected_exit")
        raise
    return 0


def _run_migrate() -> int:
    from aios.config import get_settings
    from aios.db.migrations import apply_procrastinate_schema, upgrade_to_head

    db_url = get_settings().db_url
    upgrade_to_head(db_url)
    asyncio.run(apply_procrastinate_schema(db_url, verbose=True))
    return 0


def register(app: typer.Typer) -> None:
    """Attach the operator commands to the root app."""

    @app.command("api", help="Run the aios HTTP API server (uvicorn).")
    def api() -> None:
        raise typer.Exit(_run_api())

    @app.command("worker", help="Run the aios worker (procrastinate).")
    def worker() -> None:
        raise typer.Exit(_run_worker())

    @app.command(
        "migrate",
        help="Apply alembic migrations and the procrastinate schema if missing.",
    )
    def migrate() -> None:
        raise typer.Exit(_run_migrate())
