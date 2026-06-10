"""Postgres introspection helpers for tests."""

from __future__ import annotations

import asyncpg


async def count_active_backends(db_url: str, application_name: str | None = None) -> int:
    """Count this database's client backends, excluding the measurement conn itself.

    When ``application_name`` is given, count only backends carrying that exact
    ``application_name`` — used by the SSE leak e2e test to scope its count to a
    single aios instance's dedicated listener connections, so concurrent
    backends from other xdist workers or the app pool don't move the count.
    Default ``None`` preserves the original global count (the integration
    cancellation test relies on it).
    """
    conn = await asyncpg.connect(db_url)
    try:
        if application_name is None:
            result = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            )
        else:
            result = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                "AND application_name = $1",
                application_name,
            )
        return int(result)
    finally:
        await conn.close()
