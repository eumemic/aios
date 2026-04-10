"""asyncpg connection pool singleton.

The pool is created on first call to :func:`get_pool` and lives for the
process lifetime. Tests construct their own pools via :func:`create_pool`
and bypass the singleton.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.config import get_settings

_pool: asyncpg.Pool[Any] | None = None


async def create_pool(db_url: str, *, min_size: int = 1, max_size: int = 16) -> asyncpg.Pool[Any]:
    """Create a new asyncpg pool against ``db_url``.

    Used by tests and by :func:`get_pool` for the singleton case. Strips a
    ``+psycopg`` driver prefix if present so the same URL works for both
    alembic (sync psycopg) and the runtime (asyncpg).
    """
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn=db_url, min_size=min_size, max_size=max_size)
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for {db_url}")
    return pool


async def get_pool() -> asyncpg.Pool[Any]:
    """Return the process-wide pool, creating it on first call."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
    return _pool


async def close_pool() -> None:
    """Close the process-wide pool. Idempotent."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
