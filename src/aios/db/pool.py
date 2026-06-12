"""asyncpg connection pool helpers.

The API and worker processes each construct their own pool via
:func:`create_pool` at startup and stash it on their own state
(``app.state.pool`` / ``runtime.pool``).  Tests do the same.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.config import get_settings
from aios.logging import get_logger

_KNOWN_POOL_COUNT = 2  # API pool + worker pool (production call sites)
log = get_logger("aios.db.pool")

LISTENER_TCP_KEEPALIVE_SETTINGS = {
    "tcp_keepalives_idle": "60",
    "tcp_keepalives_interval": "10",
    "tcp_keepalives_count": "5",
}

_POOL_TCP_KEEPALIVE_SETTINGS = LISTENER_TCP_KEEPALIVE_SETTINGS


def normalize_dsn(db_url: str) -> str:
    """Strip SQLAlchemy/alembic driver prefixes; asyncpg wants bare ``postgresql://``."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


def listener_application_name(instance_id: str | None = None) -> str:
    """Postgres ``application_name`` tag for dedicated SSE/notify listener conns.

    Single source of truth shared by the listener connect path
    (:func:`aios.db.listen._connect_listener`) and the e2e leak test, which
    filters ``pg_stat_activity`` by this exact label so its backend count is
    scoped to THIS aios instance's listeners — robust to concurrent backends
    from other xdist workers / the app pool. ``instance_id`` defaults to
    ``get_settings().instance_id``; passed explicitly only by tests.
    Truncated to 63 characters; ``instance_id`` is ASCII (Settings pattern
    ``^[a-z_][a-z0-9_]*$``), so the result stays within Postgres's 63-byte
    ``application_name`` limit (``NAMEDATALEN - 1``).
    """
    iid = instance_id if instance_id is not None else get_settings().instance_id
    return f"aios-listener:{iid}"[:63]


async def create_pool(db_url: str, *, min_size: int = 1, max_size: int = 8) -> asyncpg.Pool[Any]:
    """Create a new asyncpg pool against ``db_url``."""
    # asyncpg exposes no client-side keepalive kwarg, so the statement/idle
    # timeouts and TCP keepalive are applied as Postgres USERSET GUCs on every
    # pooled connection — bounding runaway scans, idle-in-txn leaks, and dead
    # connections behind a silently-dropped TCP link.
    pool = await asyncpg.create_pool(
        dsn=normalize_dsn(db_url),
        min_size=min_size,
        max_size=max_size,
        server_settings={
            "statement_timeout": "30000",
            "idle_in_transaction_session_timeout": "60000",
            **_POOL_TCP_KEEPALIVE_SETTINGS,
        },
    )
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for {db_url}")
    async with pool.acquire() as conn:
        pg_max_connections = int(await conn.fetchval("SHOW max_connections"))
    total_pool_capacity = max_size * _KNOWN_POOL_COUNT
    if total_pool_capacity >= pg_max_connections:
        log.warning(
            "db.pool.unsafe_max_size",
            max_size=max_size,
            known_pool_count=_KNOWN_POOL_COUNT,
            total_pool_capacity=total_pool_capacity,
            pg_max_connections=pg_max_connections,
        )
    return pool
