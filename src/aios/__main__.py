"""CLI entrypoint: ``python -m aios <subcommand>``.

Subcommands:

* ``aios api``     — uvicorn boot of the FastAPI app
* ``aios worker``  — procrastinate worker process (runs the harness loop)
* ``aios migrate`` — alembic upgrade head + procrastinate schema apply
* ``aios tail``    — structured real-time session event viewer (SSE client)
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys


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

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(worker_main())
    return 0


async def _apply_procrastinate_schema_if_missing() -> None:
    """Apply procrastinate's schema if it isn't already present.

    procrastinate's ``apply_schema_async`` is NOT idempotent — running it
    twice raises ``DuplicateTable``. The standard guard is the Postgres
    ``to_regclass`` function, which returns NULL if the named relation
    doesn't exist.
    """
    import asyncpg

    from aios.config import get_settings
    from aios.db.pool import create_pool
    from aios.harness.procrastinate_app import app as procrastinate_app

    settings = get_settings()
    pool = await create_pool(settings.db_url, max_size=2)
    try:
        async with pool.acquire() as conn:
            present = await conn.fetchval("SELECT to_regclass('procrastinate_jobs')")
            if present is not None:
                print("procrastinate schema already present, skipping", file=sys.stderr)
                return
    finally:
        await pool.close()

    print("applying procrastinate schema...", file=sys.stderr)
    await procrastinate_app.open_async()
    try:
        await procrastinate_app.schema_manager.apply_schema_async()
    finally:
        await procrastinate_app.close_async()
    print("procrastinate schema applied", file=sys.stderr)
    # Avoid an unused-import warning while still validating asyncpg is on the
    # path — the function above relies on it transitively via create_pool.
    _ = asyncpg


def _run_migrate() -> int:
    rc = subprocess.call(["alembic", "upgrade", "head"])
    if rc != 0:
        return rc
    asyncio.run(_apply_procrastinate_schema_if_missing())
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: aios <api|worker|migrate|tail>", file=sys.stderr)
        return 2

    cmd = sys.argv[1]
    match cmd:
        case "api":
            return _run_api()
        case "worker":
            return _run_worker()
        case "migrate":
            return _run_migrate()
        case "tail":
            from aios.tail import run as _run_tail

            return _run_tail(sys.argv[2:])
        case _:
            print(f"aios: unknown subcommand {cmd!r}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
