"""Operator subcommands: ``api``, ``worker``, ``migrate``.

These are lifted almost verbatim from the old ``__main__.py`` implementation.
They start long-running processes (uvicorn, procrastinate worker) or run
migrations — they do NOT talk to the HTTP API.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

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


async def _apply_procrastinate_schema_if_missing() -> None:
    """Apply procrastinate's schema if missing, then idempotently install
    aios's lock-release trigger.

    ``apply_schema_async`` isn't idempotent, hence the ``to_regclass``
    guard; the trigger DDL is.
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool
    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL
    from aios.harness.procrastinate_app import app as procrastinate_app

    settings = get_settings()
    pool = await create_pool(settings.db_url, max_size=2)
    try:
        async with pool.acquire() as conn:
            present = await conn.fetchval("SELECT to_regclass('procrastinate_jobs')")
            if present is None:
                print("applying procrastinate schema...", file=sys.stderr)
                await procrastinate_app.open_async()
                try:
                    await procrastinate_app.schema_manager.apply_schema_async()
                finally:
                    await procrastinate_app.close_async()
                print("procrastinate schema applied", file=sys.stderr)
            else:
                print("procrastinate schema already present, skipping", file=sys.stderr)
            await conn.execute(LOCK_RELEASE_TRIGGER_DDL)
            print("aios lock-release trigger ensured", file=sys.stderr)
    finally:
        await pool.close()


def _run_migrate() -> int:
    rc = subprocess.call(["alembic", "upgrade", "head"])
    if rc != 0:
        return rc
    asyncio.run(_apply_procrastinate_schema_if_missing())
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
