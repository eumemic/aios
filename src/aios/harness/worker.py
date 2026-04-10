"""Worker process entrypoint.

``aios serve worker`` runs :func:`worker_main` in an asyncio event loop. It:

1. Configures structlog (TTY-friendly in dev, JSON in production)
2. Opens the asyncpg pool
3. Constructs the libsodium vault from ``AIOS_VAULT_KEY``
4. Opens the procrastinate App's psycopg connector
5. Stashes pool/vault/worker_id on the :mod:`aios.harness.runtime` module so
   procrastinate task bodies can read them (procrastinate registers tasks at
   import time, so closures over local state aren't an option)
6. Runs :func:`recover_orphans` to find and re-enqueue any sessions left
   in a half-state by a previous worker death
7. Starts ``app.run_worker_async`` which blocks until SIGTERM or SIGINT,
   pulling jobs from the ``sessions`` queue at the configured concurrency

Shutdown: procrastinate's signal handlers stop accepting new jobs, wait for
in-flight jobs to finish (up to a generous timeout), and return cleanly.
The ``finally`` then closes the procrastinate connector and the asyncpg pool.
"""

from __future__ import annotations

from aios.config import get_settings
from aios.crypto.vault import Vault
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.harness.procrastinate_app import app as procrastinate_app
from aios.harness.resume import recover_orphans
from aios.ids import make_id
from aios.logging import configure_logging, get_logger


def _make_worker_id() -> str:
    """Generate a fresh worker id for this process.

    The id is a ULID with a ``worker`` prefix. Each worker process gets a
    unique id at startup; restarts produce a new id. The DB lease columns
    track exactly which process holds a session.
    """
    # `make_id` enforces canonical resource prefixes; "worker" isn't a
    # resource type, so build the id manually.
    from ulid import ULID

    return f"worker_{ULID()}"


async def worker_main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.worker")

    pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
    vault = Vault.from_base64(settings.vault_key.get_secret_value())

    runtime.pool = pool
    runtime.vault = vault
    runtime.worker_id = _make_worker_id()

    await procrastinate_app.open_async()

    log.info(
        "worker.startup",
        worker_id=runtime.worker_id,
        concurrency=settings.worker_concurrency,
    )

    try:
        recovered = await recover_orphans(pool, procrastinate_app)
        if recovered:
            log.info("worker.recovered_orphans", count=recovered)

        await procrastinate_app.run_worker_async(
            queues=["sessions"],
            concurrency=settings.worker_concurrency,
            wait=True,
            install_signal_handlers=True,
        )
    finally:
        log.info("worker.shutdown")
        await procrastinate_app.close_async()
        await pool.close()


# Reference imports for static analysis — `make_id` is the canonical id maker
# but we use a worker-specific shape above. Keep the symbol exported for tests.
_ = make_id
