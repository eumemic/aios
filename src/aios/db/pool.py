"""asyncpg connection pool helpers.

The API and worker processes each construct their own pool via
:func:`create_pool` at startup and stash it on their own state
(``app.state.pool`` / ``runtime.pool``).  Tests do the same.
"""

from __future__ import annotations

from typing import Any

import asyncpg


def normalize_dsn(db_url: str) -> str:
    """Strip SQLAlchemy/alembic driver prefixes; asyncpg wants bare ``postgresql://``."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


async def create_pool(db_url: str, *, min_size: int = 1, max_size: int = 16) -> asyncpg.Pool[Any]:
    """Create a new asyncpg pool against ``db_url``."""
    pool = await asyncpg.create_pool(
        dsn=normalize_dsn(db_url), min_size=min_size, max_size=max_size
    )
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for {db_url}")
    return pool
