"""Worker process entrypoint.

``aios serve worker`` runs :func:`worker_main` in an asyncio event loop. It:

1. Configures structlog
2. Opens the asyncpg pool
3. Constructs the libsodium CryptoBox
4. Creates the SandboxRegistry, TaskRegistry, and McpSessionPool
5. Stashes globals on :mod:`aios.harness.runtime`
6. Opens the procrastinate connector
7. Recovers orphaned sessions (re-enqueue stuck ones)
8. Reaps orphaned sandbox containers
9. Starts the container idle-TTL reaper
10. Starts ``app.run_worker_async`` which blocks until SIGTERM/SIGINT

Shutdown: procrastinate's signal handlers stop accepting new jobs and wait
for in-flight jobs. The ``finally`` block then cancels in-flight tool tasks,
releases all containers, closes MCP sessions, and closes connections.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import aios.tools  # noqa: F401  — side-effect: register built-in tools

if TYPE_CHECKING:
    import asyncpg
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.procrastinate_app import app as procrastinate_app
from aios.harness.sweep import wake_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.logging import configure_logging, get_logger
from aios.mcp.pool import McpSessionPool
from aios.sandbox.registry import SandboxRegistry


def _make_worker_id() -> str:
    from ulid import ULID

    return f"worker_{ULID()}"


async def worker_main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.worker")

    pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    sandbox_registry = SandboxRegistry()
    task_registry = TaskRegistry()
    mcp_session_pool = McpSessionPool()

    runtime.pool = pool
    runtime.crypto_box = crypto_box
    runtime.worker_id = _make_worker_id()
    runtime.sandbox_registry = sandbox_registry
    runtime.task_registry = task_registry
    runtime.mcp_session_pool = mcp_session_pool

    await procrastinate_app.open_async()

    log.info(
        "worker.startup",
        worker_id=runtime.worker_id,
        concurrency=settings.worker_concurrency,
    )

    sweep_task: asyncio.Task[None] | None = None

    try:
        # Startup sweep: repair ghosts and wake sessions needing inference.
        sweep = await wake_sessions_needing_inference(pool, task_registry)
        if sweep.woken_sessions or sweep.repaired_ghosts:
            log.info(
                "worker.startup_sweep",
                woken=sweep.woken_sessions,
                repaired_ghosts=sweep.repaired_ghosts,
            )

        # Reap orphaned sandbox containers.
        async with pool.acquire() as conn:
            active_session_ids = await queries.list_running_session_ids(conn)
        reaped = await sandbox_registry.reap_orphans(active_session_ids)
        if reaped:
            log.info("worker.reaped_orphan_containers", count=reaped)

        # Start container idle-TTL reaper.
        sandbox_registry.start_reaper(idle_timeout=settings.container_idle_timeout_seconds)

        # Start periodic sweep (every 30s).
        sweep_task = asyncio.create_task(
            _periodic_sweep(pool, task_registry, interval=30),
            name="periodic_sweep",
        )

        await procrastinate_app.run_worker_async(
            queues=["sessions"],
            concurrency=settings.worker_concurrency,
            wait=True,
            install_signal_handlers=True,
        )
    finally:
        log.info("worker.shutdown")
        if sweep_task is not None:
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task
        sandbox_registry.stop_reaper()
        await task_registry.shutdown()
        await sandbox_registry.release_all()
        await mcp_session_pool.close_all()
        await procrastinate_app.close_async()
        await pool.close()


async def _periodic_sweep(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    *,
    interval: int = 30,
) -> None:
    """Background task: run the sweep periodically."""
    log = get_logger("aios.worker.sweep")
    while True:
        await asyncio.sleep(interval)
        try:
            sweep = await wake_sessions_needing_inference(pool, task_registry)
            if sweep.woken_sessions or sweep.repaired_ghosts:
                log.info(
                    "periodic_sweep.woken",
                    count=sweep.woken_sessions,
                    repaired_ghosts=sweep.repaired_ghosts,
                )
        except Exception:
            log.exception("periodic_sweep.failed")
