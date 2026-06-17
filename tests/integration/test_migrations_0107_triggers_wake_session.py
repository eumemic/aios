"""Integration tests for migration 0107 (triggers wake_session action, #1280).

Owns the migration mechanics: a clean up/down round-trip, the fail-hard
downgrade refusal once a ``wake_session`` row exists, that a ``wake_session``
row INSERTs under the new CHECK and is REJECTED under the old, and that every
pre-existing action kind still inserts under the new CHECK.

Each test mutates ``alembic_version``, so the container is function-scoped.
Modeled on tests/integration/test_migrations_0086_triggers_slice2.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# FK chain for seeding a trigger row at head (NOT-NULL-without-default columns).
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

# A wake_session trigger row in its first-shipped shape: target_session_id +
# content in the action, NULL next_fire (cron source carries one; we use a cron
# source so it is schedulable, but a wake_session action — no environment_id, so
# the iff constraint is satisfied with left-side false = right-side false).
_WAKE_SESSION_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action, enabled, next_fire)
VALUES
    ('trig_ws', 'ses_mig', 'acc_mig', 'wake-row', 'cron',
     '{"schedule": "*/5 * * * *"}'::jsonb,
     '{"kind": "wake_session", "target_session_id": "sess_other", "content": "go look"}'::jsonb,
     TRUE, now());
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


async def _insert_raises(db_url: str, sql: str) -> bool:
    """Return True iff the INSERT raises a CheckViolationError."""
    conn = await asyncpg.connect(db_url)
    try:
        try:
            await conn.execute(sql)
        except asyncpg.CheckViolationError:
            return True
        return False
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_and_clean_downgrade_round_trip(postgres: object) -> None:
    """With zero wake_session rows, 0107 upgrades and downgrades cleanly."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0107"], db_url)
    assert up.returncode == 0, f"upgrade to 0107 failed:\n{up.stderr}\n{up.stdout}"

    down = _run_alembic(["downgrade", "0106"], db_url)
    assert down.returncode == 0, f"downgrade to 0106 failed:\n{down.stderr}\n{down.stdout}"


@needs_docker
@pytest.mark.integration
def test_wake_session_row_accepted_under_new_check_rejected_under_old(postgres: object) -> None:
    """A wake_session row INSERTs at head (0107) and is rejected at the prior
    revision (0106) — the CHECK swap is what makes the kind representable."""
    db_url = _alembic_url(postgres)

    # Under 0106 (prior head) the row is unrepresentable: ELSE false.
    up106 = _run_alembic(["upgrade", "0106"], db_url)
    assert up106.returncode == 0, f"upgrade to 0106 failed:\n{up106.stderr}\n{up106.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    assert asyncio.run(_insert_raises(db_url, _WAKE_SESSION_ROW_SQL)), (
        "wake_session row should violate the 0106 action CHECK"
    )

    # Under 0107 the same row inserts.
    up107 = _run_alembic(["upgrade", "0107"], db_url)
    assert up107.returncode == 0, f"upgrade to 0107 failed:\n{up107.stderr}\n{up107.stdout}"
    asyncio.run(_execute(db_url, _WAKE_SESSION_ROW_SQL))


@needs_docker
@pytest.mark.integration
def test_every_prior_kind_still_inserts_under_new_check(postgres: object) -> None:
    """The four pre-existing branches stay byte-identical, so every prior
    action kind still inserts under the 0107 CHECK."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0107"], db_url)
    assert up.returncode == 0, f"upgrade to 0107 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    rows = {
        "sandbox": (
            "'cron'",
            '\'{"schedule": "*/5 * * * *"}\'::jsonb',
            '\'{"kind": "sandbox_command", "command": "echo hi", '
            '"timeout_seconds": 60, "max_output_bytes": 2048}\'::jsonb',
            "NULL",
        ),
        "wake_owner": (
            "'cron'",
            '\'{"schedule": "*/5 * * * *"}\'::jsonb',
            '\'{"kind": "wake_owner", "content": "hi"}\'::jsonb',
            "NULL",
        ),
        "workflow": (
            "'cron'",
            '\'{"schedule": "*/5 * * * *"}\'::jsonb',
            '\'{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": null, '
            '"input_template": null, "vault_ids": []}\'::jsonb',
            "'env_mig'",
        ),
    }

    async def _insert_all() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            for name, (src, spec, action, env) in rows.items():
                # workflow rows carry environment_id (iff constraint).
                await conn.execute(
                    f"INSERT INTO triggers (id, owner_session_id, account_id, name, source, "
                    f"source_spec, action, enabled, next_fire, environment_id) VALUES "
                    f"('trig_{name}', 'ses_mig', 'acc_mig', '{name}', {src}, {spec}, {action}, "
                    f"TRUE, now(), {env})"
                )
        finally:
            await conn.close()

    asyncio.run(_insert_all())


@needs_docker
@pytest.mark.integration
def test_downgrade_refuses_wake_session_rows(postgres: object) -> None:
    """A wake_session row is unrepresentable under the prior predicate, so the
    downgrade fails hard and rolls back (the 0086 stance)."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0107"], db_url)
    assert up.returncode == 0, f"upgrade to 0107 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _WAKE_SESSION_ROW_SQL))

    down = _run_alembic(["downgrade", "0106"], db_url)
    assert down.returncode != 0, f"downgrade should have failed loud:\n{down.stdout}"
    assert "cannot downgrade" in down.stderr
