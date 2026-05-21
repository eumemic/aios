"""Postgres introspection helpers for tests."""

from __future__ import annotations

import asyncpg


async def count_active_backends(db_url: str) -> int:
    """Count this database's client backends, excluding the measurement conn itself."""
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.fetchval(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
        return int(result)
    finally:
        await conn.close()
