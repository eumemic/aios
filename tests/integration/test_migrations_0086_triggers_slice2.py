"""Integration tests for migration 0086 (triggers slice 2).

The e2e suite runs against a DB already at head; these tests own the migration
mechanics themselves: a clean up/down round-trip with zero slice-2 rows, the
fail-hard downgrade refusal once a ``run_completion``/``workflow`` row exists,
and the ``triggers_run_completion_no_next_fire`` guard (the DB half of the §3
"reactive rows are unschedulable by the tick" invariant — a service-layer slip
here would be a tick-speed hot re-claim runaway, hence the constraint).

Each test mutates ``alembic_version``, so the container is function-scoped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# FK chain for seeding a trigger row at head (NOT-NULL-without-default columns
# only; mirrors test_migrations_0083_triggers.py).
_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_mig', NULL, TRUE, 'mig-test') ON CONFLICT DO NOTHING;
INSERT INTO environments (id, name, account_id)
VALUES ('env_mig', 'mig-env', 'acc_mig') ON CONFLICT DO NOTHING;
INSERT INTO agents (id, name, model, account_id)
VALUES ('agn_mig', 'mig-agent', 'fake/test', 'acc_mig') ON CONFLICT DO NOTHING;
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
VALUES ('ses_mig', 'agn_mig', 'env_mig', '/tmp/ws-mig', 'acc_mig') ON CONFLICT DO NOTHING;
"""

# A run_completion trigger row in its first-shipped shape: workflow_id +
# statuses in source_spec, NULL next_fire, a wake_owner action (environment_id
# stays NULL — the iff constraint requires that for non-workflow kinds).
_RUN_COMPLETION_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action, enabled, next_fire)
VALUES
    ('trig_rc', 'ses_mig', 'acc_mig', 'watch-row', 'run_completion',
     '{"workflow_id": "wf_watched", "statuses": ["completed", "errored", "cancelled"]}'::jsonb,
     '{"kind": "wake_owner", "content": "a run completed"}'::jsonb,
     TRUE, NULL);
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _table_exists(db_url: str, name: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval("SELECT to_regclass($1)", f"public.{name}") is not None
    finally:
        await conn.close()


async def _column_exists(db_url: str, table: str, column: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name = $1 AND column_name = $2)",
                table,
                column,
            )
        )
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_and_clean_downgrade_round_trip(postgres: object) -> None:
    """With zero slice-2 rows, 0086 upgrades and downgrades cleanly: the audit
    table and the env column appear at head and vanish on downgrade."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0086"], db_url)
    assert up.returncode == 0, f"upgrade to 0086 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_table_exists(db_url, "trigger_runs"))
    assert asyncio.run(_column_exists(db_url, "triggers", "environment_id"))

    down = _run_alembic(["downgrade", "0085"], db_url)
    assert down.returncode == 0, f"downgrade to 0085 failed:\n{down.stderr}\n{down.stdout}"
    assert not asyncio.run(_table_exists(db_url, "trigger_runs"))
    assert not asyncio.run(_column_exists(db_url, "triggers", "environment_id"))


@needs_docker
@pytest.mark.integration
def test_downgrade_refuses_slice2_rows(postgres: object) -> None:
    """A run_completion row is unrepresentable under the 0083 predicates, so
    the downgrade fails hard and rolls back (the 0083 wake_owner stance)."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0086"], db_url)
    assert up.returncode == 0, f"upgrade to 0086 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _RUN_COMPLETION_ROW_SQL))

    down = _run_alembic(["downgrade", "0085"], db_url)
    assert down.returncode != 0, f"downgrade should have failed loud:\n{down.stdout}"
    assert "cannot downgrade" in down.stderr

    # Rolled back: still at head, audit table + column intact.
    assert asyncio.run(_table_exists(db_url, "trigger_runs"))
    assert asyncio.run(_column_exists(db_url, "triggers", "environment_id"))


@needs_docker
@pytest.mark.integration
def test_run_completion_rows_reject_next_fire(postgres: object) -> None:
    """The resolved sign-off #2 guard: a run_completion row can never carry a
    next_fire, so it is unschedulable by the scheduler tick BY CONSTRAINT, not
    merely by service-layer discipline."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0086"], db_url)
    assert up.returncode == 0, f"upgrade to 0086 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _RUN_COMPLETION_ROW_SQL))

    async def _set_next_fire() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            with pytest.raises(asyncpg.CheckViolationError) as excinfo:
                await conn.execute("UPDATE triggers SET next_fire = now() WHERE id = 'trig_rc'")
            assert "triggers_run_completion_no_next_fire" in str(excinfo.value)
        finally:
            await conn.close()

    asyncio.run(_set_next_fire())
