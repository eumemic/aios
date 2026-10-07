"""Migration 0194 against a live Postgres (#2513): ``wf_runs.visibility`` is
constrained to its known values.

The readers treat every value other than ``account`` as launcher-private, so an
unknown value would silently mean "private". The CHECK makes it unrepresentable.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from tests.conftest import needs_docker
from tests.helpers.alembic import run_alembic

_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_c', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_c', 'env', '{}'::jsonb, 'acc_c');
INSERT INTO wf_runs (id, account_id, environment_id, script, script_sha,
                     host_semantics_epoch, status)
VALUES ('run_existing', 'acc_c', 'env_c', 'x', 'sha', 0, 'completed');
"""

_FORCE_UNKNOWN_SQL = """
ALTER TABLE wf_runs DISABLE TRIGGER wf_runs_stamp_visibility_trg;
INSERT INTO wf_runs (id, account_id, environment_id, script, script_sha,
                     host_semantics_epoch, status, visibility)
VALUES ('run_unknown', 'acc_c', 'env_c', 'x', 'sha', 0, 'completed', 'everyone');
"""


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _fetch(db_url: str, sql: str) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql))
    finally:
        await conn.close()


def _constraint(db_url: str) -> list[asyncpg.Record]:
    return asyncio.run(
        _fetch(
            db_url,
            "SELECT convalidated FROM pg_constraint WHERE conname = 'wf_runs_visibility_known'",
        )
    )


@needs_docker
@pytest.mark.integration
def test_upgrade_constrains_visibility_to_its_known_values(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0191"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL))
    up = run_alembic(["upgrade", "0194"], db_url)
    assert up.returncode == 0, up.stderr

    (con,) = _constraint(db_url)
    assert con["convalidated"] is True
    with pytest.raises(asyncpg.CheckViolationError):
        asyncio.run(_execute(db_url, _FORCE_UNKNOWN_SQL))
    # The insert trigger's own stamps still pass.
    rows = asyncio.run(_fetch(db_url, "SELECT id, visibility FROM wf_runs"))
    assert {r["id"]: r["visibility"] for r in rows} == {"run_existing": "account"}


@needs_docker
@pytest.mark.integration
def test_upgrade_refuses_a_row_with_an_unknown_visibility(migration_db_url: str) -> None:
    """A pre-existing unknown value fails the deploy loudly instead of being guessed;
    once the row is repaired, the upgrade re-runs cleanly."""
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0191"], db_url).returncode == 0
    asyncio.run(_execute(db_url, _SEED_SQL + _FORCE_UNKNOWN_SQL))
    up = run_alembic(["upgrade", "0194"], db_url)
    assert up.returncode != 0
    assert [c["convalidated"] for c in _constraint(db_url)] in ([], [False])
    (version,) = asyncio.run(_fetch(db_url, "SELECT version_num FROM alembic_version"))
    assert version["version_num"] == "0191"

    asyncio.run(_execute(db_url, "DELETE FROM wf_runs WHERE id = 'run_unknown'"))
    up = run_alembic(["upgrade", "0194"], db_url)
    assert up.returncode == 0, up.stderr
    assert [c["convalidated"] for c in _constraint(db_url)] == [True]


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_the_constraint(migration_db_url: str) -> None:
    db_url = migration_db_url
    assert run_alembic(["upgrade", "0194"], db_url).returncode == 0
    down = run_alembic(["downgrade", "0191"], db_url)
    assert down.returncode == 0, down.stderr
    assert _constraint(db_url) == []
    assert run_alembic(["upgrade", "0194"], db_url).returncode == 0
    assert len(_constraint(db_url)) == 1
