"""Migration 0118 adds ``wf_runs.source_version`` + the strict composite FK to
``workflow_versions`` (Phase 2 of 3).

Covers the load-bearing halves:

* the **column + composite FK** — a new run's non-NULL ``source_version`` must
  resolve to a ``workflow_versions`` row (MATCH SIMPLE exempts NULL legacy rows);
* the **best-effort historical backfill** — a run whose ``script_sha`` matches
  exactly one kept version gets ``source_version`` set; a run whose sha matches
  >1 version (a rename-only bump keeps the script byte-identical) stays NULL.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_SCRIPT_UNIQ = "async def main(input):\n    return 1\n"
_SCRIPT_DUP = "async def main(input):\n    return 2\n"  # byte-identical across v1+v2 of wf_b

_SHA_UNIQ = hashlib.sha256(_SCRIPT_UNIQ.encode("utf-8")).hexdigest()
_SHA_DUP = hashlib.sha256(_SCRIPT_DUP.encode("utf-8")).hexdigest()

# wf_a: a single version (v1) — its sha is unambiguous, so a run on it backfills.
# wf_b: two versions (v1, v2) with a BYTE-IDENTICAL script (a rename-only bump) —
#       the sha matches both, so a run on it must stay NULL (NULL-on-ambiguity).
_SEED_SQL = f"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, config, account_id)
VALUES ('env_root', 'env', '{{}}'::jsonb, 'acc_root');
INSERT INTO workflows (id, account_id, name, version, script)
VALUES ('wf_a', 'acc_root', 'alpha', 1, '{_SCRIPT_UNIQ}'),
       ('wf_b', 'acc_root', 'beta', 2, '{_SCRIPT_DUP}');
INSERT INTO workflow_versions (workflow_id, account_id, version, name, script)
VALUES ('wf_a', 'acc_root', 1, 'alpha', '{_SCRIPT_UNIQ}'),
       ('wf_b', 'acc_root', 1, 'beta-old', '{_SCRIPT_DUP}'),
       ('wf_b', 'acc_root', 2, 'beta', '{_SCRIPT_DUP}');
INSERT INTO wf_runs
    (id, workflow_id, account_id, environment_id, script, script_sha,
     host_semantics_epoch, status)
VALUES
    ('run_unambiguous', 'wf_a', 'acc_root', 'env_root', '{_SCRIPT_UNIQ}',
     '{_SHA_UNIQ}', 0, 'completed'),
    ('run_ambiguous', 'wf_b', 'acc_root', 'env_root', '{_SCRIPT_DUP}',
     '{_SHA_DUP}', 0, 'completed');
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
        return list(await conn.fetch(sql, *args))
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
def test_backfill_sets_unambiguous_and_leaves_ambiguous_null(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0116"], db_url)
    assert up.returncode == 0, f"upgrade to 0116 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0118"], db_url)
    assert up.returncode == 0, f"upgrade to 0118 failed:\n{up.stderr}\n{up.stdout}"

    rows = asyncio.run(_fetch(db_url, "SELECT id, source_version FROM wf_runs ORDER BY id"))
    by_id = {r["id"]: r["source_version"] for r in rows}
    # Unambiguous sha → backfilled to the single matching version (v1 of wf_a).
    assert by_id["run_unambiguous"] == 1
    # sha matches BOTH v1 and v2 of wf_b → NULL-on-ambiguity.
    assert by_id["run_ambiguous"] is None


@needs_docker
@pytest.mark.integration
def test_composite_fk_rejects_dangling_pointer(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0118"], db_url)
    assert up.returncode == 0, f"upgrade to 0118 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    # A run pointing at a version that does not exist (v99) must be rejected by
    # the now-VALIDATEd composite FK.
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        asyncio.run(
            _execute(
                db_url,
                "INSERT INTO wf_runs "
                "(id, workflow_id, account_id, environment_id, script, script_sha, "
                " source_version, host_semantics_epoch, status) "
                "VALUES ('run_bad', 'wf_a', 'acc_root', 'env_root', 'x', 'sha', "
                " 99, 0, 'pending')",
            )
        )

    # A NULL source_version (legacy/unbackfillable) is exempt (MATCH SIMPLE).
    asyncio.run(
        _execute(
            db_url,
            "INSERT INTO wf_runs "
            "(id, workflow_id, account_id, environment_id, script, script_sha, "
            " source_version, host_semantics_epoch, status) "
            "VALUES ('run_null', 'wf_a', 'acc_root', 'env_root', 'x', 'sha', "
            " NULL, 0, 'pending')",
        )
    )
    rows = asyncio.run(_fetch(db_url, "SELECT source_version FROM wf_runs WHERE id = 'run_null'"))
    assert rows[0]["source_version"] is None


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_column_and_constraint(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    assert _run_alembic(["upgrade", "0118"], db_url).returncode == 0
    down = _run_alembic(["downgrade", "0116"], db_url)
    assert down.returncode == 0, f"downgrade failed:\n{down.stderr}\n{down.stdout}"

    cols = asyncio.run(
        _fetch(
            db_url,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'wf_runs' AND column_name = 'source_version'",
        )
    )
    assert cols == []
