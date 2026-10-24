"""Migration 0195 against a live Postgres (#2393): the concurrent build of
``events_tool_marker_idx`` has a bounded lock wait, and a retry repairs it.

``CREATE INDEX CONCURRENTLY`` waits for every transaction that holds a lock
conflicting with SHARE on ``events``.  A session left idle in a transaction
after writing an event is such a holder.  Without ``lock_timeout`` the build
waits for it indefinitely and stalls the deploy.  With the bound, the
migration fails loudly.  The failed build leaves an INVALID index behind,
and the ``DROP INDEX CONCURRENTLY IF EXISTS`` at the top of ``upgrade`` makes
the retry build a valid one.
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
import psycopg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_INDEX = "events_tool_marker_idx"


async def _fetch(db_url: str, sql: str) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql))
    finally:
        await conn.close()


def _index_validity(db_url: str) -> list[bool]:
    rows = asyncio.run(
        _fetch(
            db_url,
            "SELECT i.indisvalid FROM pg_index i "
            f"JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = '{_INDEX}'",
        )
    )
    return [r["indisvalid"] for r in rows]


def _version(db_url: str) -> str:
    (row,) = asyncio.run(_fetch(db_url, "SELECT version_num FROM alembic_version"))
    return str(row["version_num"])


@needs_docker
@pytest.mark.integration
def test_upgrade_builds_a_valid_index(migration_db_url: str) -> None:
    db_url = migration_db_url
    up = run_alembic(["upgrade", "0195"], db_url)
    assert up.returncode == 0, up.stderr
    assert _index_validity(db_url) == [True]
    down = run_alembic(["downgrade", "0194"], db_url)
    assert down.returncode == 0, down.stderr
    assert _index_validity(db_url) == []


@needs_docker
@pytest.mark.integration
def test_blocked_build_fails_fast_and_retry_replaces_invalid_remnant(
    migration_db_url: str,
) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0194"], db_url).returncode == 0

    # An open transaction holding ROW EXCLUSIVE on events, like a session left
    # idle in a transaction after writing an event.
    holder = psycopg.connect(db_url)
    try:
        holder.execute("LOCK TABLE events IN ROW EXCLUSIVE MODE")
        started = time.monotonic()
        up = run_alembic(["upgrade", "0195"], db_url)
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()

    # Fails loudly on the bound instead of waiting for the holder.
    assert up.returncode != 0
    assert "lock timeout" in up.stderr, up.stderr
    assert elapsed < 60, elapsed
    assert _version(db_url) == "0194"
    # The interrupted concurrent build left an INVALID index behind.
    assert _index_validity(db_url) == [False]

    # Retry with the holder gone: the remnant is dropped and rebuilt valid.
    retry = run_alembic(["upgrade", "0195"], db_url)
    assert retry.returncode == 0, retry.stderr
    assert _version(db_url) == "0195"
    assert _index_validity(db_url) == [True]
