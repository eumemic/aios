"""Worker process entrypoint.

``aios serve worker`` runs :func:`worker_main` in an asyncio event loop. It:

1. Configures structlog
2. Opens the asyncpg pool
3. Acquires a Postgres advisory lock to refuse a duplicate worker
4. Constructs the libsodium CryptoBox
5. Creates the SandboxRegistry, TaskRegistry, and McpSessionPool
6. Resolves and starts the connector subprocess supervisor
7. Stashes globals on :mod:`aios.harness.runtime`
8. Opens the procrastinate connector
9. Recovers orphaned sessions (re-enqueue stuck ones)
10. Reaps orphaned sandbox containers
11. Starts the container idle-TTL reaper
12. Starts ``app.run_worker_async`` which blocks until SIGTERM/SIGINT

Shutdown: procrastinate's signal handlers stop accepting new jobs and wait
for in-flight jobs. The ``finally`` block then cancels in-flight tool tasks,
releases all containers, closes MCP sessions, stops the connector
supervisor, and closes connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Any

import asyncpg

import aios.tools  # noqa: F401  — side-effect: register built-in tools

if TYPE_CHECKING:
    pass
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool, normalize_dsn
from aios.harness import runtime
from aios.harness.connector_supervisor import (
    ConnectorSubprocessRegistry,
    resolve_connector_specs,
)
from aios.harness.procrastinate_app import app as procrastinate_app
from aios.harness.sweep import (
    reap_stalled_jobs,
    wake_sessions_needing_inference,
)
from aios.harness.task_registry import TaskRegistry
from aios.logging import configure_logging, get_logger
from aios.mcp.pool import McpSessionPool
from aios.sandbox.registry import SandboxRegistry

# 64-bit hash of the lock identifier; stable across processes / restarts.
# Generated once via Postgres ``hashtextextended('aios_worker_connector_supervisor', 0)``
# and inlined so we don't burn a query just to compute it.  The text key
# stays in code as documentation of *what* this number means.
_WORKER_LOCK_KEY_TEXT = "aios_worker_connector_supervisor"


def _make_worker_id() -> str:
    from ulid import ULID

    return f"worker_{ULID()}"


async def worker_main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.worker")

    # Single-instance guard.  Two `aios worker` processes against the same
    # database would race for connector subprocess ownership (signal-cli's
    # local socket, telegram's bot session, etc.) so we refuse to boot a
    # second worker by holding a session-scoped advisory lock on a
    # dedicated connection.  Pool-borrowed connections release the lock
    # on return, so the lock conn is intentionally NOT in the pool.
    lock_conn = await _acquire_worker_lock(settings.db_url, log)
    if lock_conn is None:
        sys.exit(1)

    # Everything below holds resources that need ordered teardown; the
    # try/finally wraps the entire construction so a partial-startup
    # failure (e.g. ``resolve_connector_specs`` rejecting a bad entry
    # in ``connectors_enabled``, ``create_pool`` racing a temporarily
    # unreachable DB) still releases the advisory lock and any
    # already-built resource.
    pool: asyncpg.Pool[Any] | None = None
    sandbox_registry: SandboxRegistry | None = None
    task_registry: TaskRegistry | None = None
    mcp_session_pool: McpSessionPool | None = None
    connector_registry: ConnectorSubprocessRegistry | None = None
    procrastinate_opened = False
    sweep_task: asyncio.Task[None] | None = None

    try:
        pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
        crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
        sandbox_registry = SandboxRegistry()
        task_registry = TaskRegistry()
        mcp_session_pool = McpSessionPool()
        connector_specs = resolve_connector_specs(settings)
        connector_registry = ConnectorSubprocessRegistry(connector_specs, settings=settings)

        runtime.pool = pool
        runtime.crypto_box = crypto_box
        runtime.worker_id = _make_worker_id()
        runtime.sandbox_registry = sandbox_registry
        runtime.task_registry = task_registry
        runtime.mcp_session_pool = mcp_session_pool
        runtime.connector_subprocess_registry = connector_registry

        await procrastinate_app.open_async()
        procrastinate_opened = True
        await connector_registry.start()

        log.info(
            "worker.startup",
            worker_id=runtime.worker_id,
            concurrency=settings.worker_concurrency,
            connectors=connector_registry.names,
        )

        # Startup sweep:
        #   1. Reap stalled procrastinate jobs (workers that died without
        #      releasing their session lock — laptop sleep, OOM, crash).
        #      Must run BEFORE the wake sweep so freshly-unblocked sessions
        #      get re-enqueued in the same pass.
        #   2. Repair tool-call ghosts and wake sessions needing inference.
        await reap_stalled_jobs(procrastinate_app.job_manager)
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
            _periodic_sweep(pool, task_registry, procrastinate_app.job_manager, interval=30),
            name="periodic_sweep",
        )

        await procrastinate_app.run_worker_async(
            queues=["sessions", "connectors"],
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
        if sandbox_registry is not None:
            sandbox_registry.stop_reaper()
        if task_registry is not None:
            await task_registry.shutdown()
        if sandbox_registry is not None:
            await sandbox_registry.release_all()
        if mcp_session_pool is not None:
            await mcp_session_pool.close_all()
        if connector_registry is not None:
            await connector_registry.shutdown()
        if procrastinate_opened:
            await procrastinate_app.close_async()
        if pool is not None:
            await pool.close()
        # Lock conn drops last so single-instance enforcement holds for
        # the entire shutdown sequence (a parallel `aios worker` mid-startup
        # would still get refused while we tear down).
        with contextlib.suppress(asyncpg.PostgresError, OSError):
            await lock_conn.close()
        runtime.connector_subprocess_registry = None


async def _acquire_worker_lock(db_url: str, log: Any) -> asyncpg.Connection[Any] | None:
    """Try to grab the single-worker advisory lock on a dedicated connection.

    Returns the held connection on success (caller must keep it alive),
    or ``None`` when another worker already owns the lock.  Postgres
    releases session-scoped advisory locks on connection close, so the
    caller's only obligation is to close the connection on shutdown.

    The connection is dedicated — never returned to the pool — because
    pool reset would issue ``DISCARD ALL``, which releases advisory
    locks and silently drops the guarantee.
    """
    dsn = normalize_dsn(db_url)
    conn = await asyncpg.connect(dsn)
    try:
        held: bool = await conn.fetchval(
            "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
            _WORKER_LOCK_KEY_TEXT,
        )
    except Exception:
        await conn.close()
        raise
    if not held:
        log.error(
            "worker.duplicate_instance_refused",
            lock_key=_WORKER_LOCK_KEY_TEXT,
        )
        await conn.close()
        return None
    return conn


async def _periodic_sweep(
    pool: asyncpg.Pool[Any],
    task_registry: TaskRegistry,
    job_manager: Any,
    *,
    interval: int = 30,
) -> None:
    """Background task: run the sweep periodically."""
    log = get_logger("aios.worker.sweep")
    while True:
        await asyncio.sleep(interval)
        try:
            # Reap stalled jobs first so any unblocked sessions get re-enqueued
            # in the same tick (mirrors worker_main's startup sequence).
            await reap_stalled_jobs(job_manager)
            sweep = await wake_sessions_needing_inference(pool, task_registry)
            if sweep.woken_sessions or sweep.repaired_ghosts:
                log.info(
                    "periodic_sweep.woken",
                    count=sweep.woken_sessions,
                    repaired_ghosts=sweep.repaired_ghosts,
                )
        except Exception:
            log.exception("periodic_sweep.failed")
