"""asyncpg connection pool helpers.

The API and worker processes each construct their own pool via
:func:`create_pool` at startup and stash it on their own state
(``app.state.pool`` / ``runtime.pool``).  Tests do the same.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.logging import get_logger

_KNOWN_POOL_COUNT = 2  # API pool + worker pool (production call sites)
log = get_logger("aios.db.pool")


def normalize_dsn(db_url: str) -> str:
    """Strip SQLAlchemy/alembic driver prefixes; asyncpg wants bare ``postgresql://``."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


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
            "tcp_keepalives_idle": "60",
            "tcp_keepalives_interval": "10",
            "tcp_keepalives_count": "5",
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
