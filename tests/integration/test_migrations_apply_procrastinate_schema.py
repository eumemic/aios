"""End-to-end verify ``apply_procrastinate_schema`` accepts SQLAlchemy-suffixed URLs.

Regression test for the ``aios migrate`` crash introduced by 3704f9c8: an
operator-suffixed ``AIOS_DB_URL`` (``postgresql+asyncpg://`` / ``postgresql+psycopg://``)
is accepted by every *other* ``aios migrate`` step — ``_migration_admission`` via
``_sync_db_url``, ``upgrade_to_head`` via ``migrations/env.py``'s scheme rewrite —
and by the worker pool (``aios.db.pool.create_pool`` via ``normalize_dsn``), but
``apply_procrastinate_schema`` passed the raw URL straight to ``asyncpg.connect`` /
``procrastinate.PsycopgConnector`` — both of which reject the driver suffix at DSN
parse time. ``aios migrate`` therefore committed every Alembic migration and then
crashed before creating ``procrastinate_jobs`` and the aios lock-release trigger,
blocking worker startup. The fix routes both consumers through ``normalize_dsn``.

The companion unit suite (``tests/unit/test_migration_apply_procrastinate_schema.py``)
pins the normalization at the ``asyncpg.connect`` / ``PsycopgConnector`` call
boundary with mocked I/O. These tests replay the *production* ``aios migrate``
ordering against a real Postgres — ``upgrade_to_head`` then
``apply_procrastinate_schema`` — feeding a suffixed URL into both, and assert the
procrastinate schema and the aios lock-release trigger actually land.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest

from aios.db.migrations import apply_procrastinate_schema, upgrade_to_head
from tests.conftest import needs_docker


def _suffix(url: str, scheme: str) -> str:
    assert url.startswith("postgresql://"), url
    return url.replace("postgresql://", scheme, 1)


_SUFFICES = ["postgresql+asyncpg://", "postgresql+psycopg://"]


@needs_docker
@pytest.mark.integration
@pytest.mark.parametrize("scheme", _SUFFICES)
def test_apply_procrastinate_schema_applies_with_suffixed_url(
    migration_db_url: str, scheme: str
) -> None:
    """Replays ``aios migrate`` against a fresh DB with a suffixed AIOS_DB_URL.

    The reported crash: ``upgrade_to_head`` commits, then
    ``asyncpg.connect(postgresql+asyncpg://…)`` raises before creating
    ``procrastinate_jobs``. Post-fix the whole path succeeds.
    """
    bare = migration_db_url
    suffixed = _suffix(bare, scheme)

    # Step 1 — Alembic migrations. ``env.py`` rewrites the suffix for the sync
    # SQLAlchemy driver, so the suffixed URL succeeds (mirrors production).
    upgrade_to_head(suffixed)

    # Step 2 — the crash site. Pre-fix this raised ClientConfigurationError.
    asyncio.run(apply_procrastinate_schema(suffixed, verbose=True))

    async def verify() -> None:
        conn: asyncpg.Connection[Any] = await asyncpg.connect(bare)
        try:
            # Procrastinate's core schema landed via the PsycopgConnector path.
            # ``procrastinate_jobs`` is the table the ``to_regclass`` guard checks
            # (it's what decides whether ``apply_schema_async`` runs at all), and
            # a non-trivial procrastinate_% table count confirms the FULL schema
            # applied (not just the jobs table) — robust to the exact set of
            # tables this procrastinate version ships with.
            assert await conn.fetchval("SELECT to_regclass('procrastinate_jobs')") is not None
            proca_tables = await conn.fetchval(
                "SELECT count(*) FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename LIKE 'procrastinate_%'"
            )
            assert proca_tables >= 3, (
                f"expected procrastinate's full schema, found only {proca_tables} procrastinate_* tables"
            )

            # The aios lock-release trigger landed (LOCK_RELEASE_TRIGGER_DDL).
            trigger = await conn.fetchval(
                "SELECT tgname FROM pg_trigger WHERE tgname = 'aios_jobs_notify_lock_released_v1'"
            )
            assert trigger == "aios_jobs_notify_lock_released_v1", (
                f"lock-release trigger missing: {trigger!r}"
            )
        finally:
            await conn.close()

    asyncio.run(verify())


@needs_docker
@pytest.mark.integration
@pytest.mark.parametrize("scheme", _SUFFICES)
def test_apply_procrastinate_schema_idempotent_with_suffixed_url(
    migrated_db_url: str, scheme: str
) -> None:
    """Re-running ``aios migrate`` with a suffixed URL is a no-op on the schema.

    The bug report's recovery is "correct ``AIOS_DB_URL`` and re-run"; this pins
    that a re-run — even routing *toward* a suffixed URL — re-enters the
    ``to_regclass('procrastinate_jobs')`` guard, skips re-applying procrastinate's
    non-idempotent ``apply_schema_async``, and only re-ensures the trigger,
    without raising on the suffix. The session ``migrated_db_url`` fixture has
    already applied the schema with the bare URL.
    """
    bare = migrated_db_url
    suffixed = _suffix(bare, scheme)

    # Must not raise; must not attempt to re-apply the (non-idempotent) schema.
    asyncio.run(apply_procrastinate_schema(suffixed, verbose=True))

    async def verify() -> None:
        conn: asyncpg.Connection[Any] = await asyncpg.connect(bare)
        try:
            assert await conn.fetchval("SELECT to_regclass('procrastinate_jobs')") is not None
            trigger = await conn.fetchval(
                "SELECT tgname FROM pg_trigger WHERE tgname = 'aios_jobs_notify_lock_released_v1'"
            )
            assert trigger == "aios_jobs_notify_lock_released_v1"
        finally:
            await conn.close()

    asyncio.run(verify())
