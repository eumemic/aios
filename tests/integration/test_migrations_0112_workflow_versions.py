"""Migration 0112 adds ``workflow_versions`` (immutable definition history).

Covers the two halves the migration must get right:

* the **backfill** — every workflow that existed before the migration gets a
  single version row at its current version (older history is unrecoverable and
  intentionally not invented); and
* the **insert-only trigger** — version rows reject ``UPDATE`` at the database.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# Two pre-existing workflows at different versions (0075-era in-place updates: the
# head survives on the row, older snapshots are gone). The backfill snapshots the
# surviving head, and ONLY the head.
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script)
VALUES ('wf_a', 'acc_root', 'alpha', 1, 'SCRIPT_A'),
       ('wf_b', 'acc_root', 'beta', 5, 'SCRIPT_B');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetch(db_url: str, sql: str, *args: object) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str, *args: object) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql, *args)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_backfill_snapshots_each_workflow_head(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0111"], db_url)
    assert up.returncode == 0, f"upgrade to 0111 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0112"], db_url)
    assert up.returncode == 0, f"upgrade to 0112 failed:\n{up.stderr}\n{up.stdout}"

    rows = asyncio.run(
        _fetch(
            db_url,
            "SELECT workflow_id, account_id, version, name, script "
            "FROM workflow_versions ORDER BY workflow_id",
        )
    )
    # Exactly one snapshot per workflow, at its surviving head — no invented history.
    assert [(r["workflow_id"], r["version"], r["name"], r["script"]) for r in rows] == [
        ("wf_a", 1, "alpha", "SCRIPT_A"),
        ("wf_b", 5, "beta", "SCRIPT_B"),
    ]
    assert all(r["account_id"] == "acc_root" for r in rows)


@needs_docker
@pytest.mark.integration
def test_version_rows_are_insert_only(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0112"], db_url)
    assert up.returncode == 0, f"upgrade to 0112 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))
    asyncio.run(
        _execute(
            db_url,
            "INSERT INTO workflow_versions "
            "(workflow_id, account_id, version, name, script) "
            "VALUES ('wf_a', 'acc_root', 1, 'alpha', 'SCRIPT_A')",
        )
    )

    with pytest.raises(asyncpg.PostgresError):
        asyncio.run(
            _execute(
                db_url,
                "UPDATE workflow_versions SET script = 'X' WHERE workflow_id = 'wf_a'",
            )
        )


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_table_and_constraint(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    assert _run_alembic(["upgrade", "0112"], db_url).returncode == 0
    down = _run_alembic(["downgrade", "0111"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"

    exists = asyncio.run(_fetch(db_url, "SELECT to_regclass('public.workflow_versions') AS t"))
    assert exists[0]["t"] is None
