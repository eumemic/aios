"""Integration tests for migration 0130 (triggers schedulable-enabled-armed, #1678).

Owns the migration mechanics: a clean up/down round-trip; that the new
``triggers_schedulable_enabled_armed`` CHECK rejects the #925 zombie state
(enabled cron / one_shot with NULL ``next_fire``) at head and admits it at the
prior revision (0129); that the upgrade AUTO-DISABLES pre-existing violator
rows BEFORE the validating ADD CONSTRAINT (the 0066 lesson) so a prod migrate
with a real zombie in the table does not abort; and that the arm is scoped to
schedulable sources only — an enabled run_completion row keeps NULL
``next_fire`` under both the 0108 reactive arm and this schedulable arm.

Each test mutates ``alembic_version``, so the container is function-scoped.
Modeled on tests/integration/test_migrations_0108_triggers_external_event.py.
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

# The #925 zombie: an ENABLED cron row with NULL next_fire — the state a manual
# `UPDATE … SET enabled=true` leaves behind. Unrepresentable under 0130.
_ZOMBIE_CRON_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire)
VALUES
    ('trig_zombie', 'ses_mig', 'acc_mig', 'zombie', 'cron',
     '{"schedule": "*/5 * * * *"}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb,
     TRUE, NULL);
"""

# A zombied one-shot (settled fork 1: the schedulable arm covers one_shot too).
_ZOMBIE_ONE_SHOT_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire)
VALUES
    ('trig_zombie_os', 'ses_mig', 'acc_mig', 'zombie-os', 'one_shot',
     '{"fire_at": "2026-06-11T09:00:00Z"}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb,
     TRUE, NULL);
"""

# A well-formed enabled cron row (armed) — passes the CHECK.
_ARMED_CRON_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire)
VALUES
    ('trig_armed', 'ses_mig', 'acc_mig', 'armed', 'cron',
     '{"schedule": "*/5 * * * *"}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb,
     TRUE, now());
"""

# A DISABLED cron row with NULL next_fire — legitimate, passes the CHECK
# (NOT enabled ⇒ first disjunct true).
_DISABLED_CRON_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire)
VALUES
    ('trig_disabled', 'ses_mig', 'acc_mig', 'disabled', 'cron',
     '{"schedule": "*/5 * * * *"}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb,
     FALSE, NULL);
"""

# An enabled run_completion row with NULL next_fire — reactive, NULL by design
# (0108 reactive arm REQUIRES it); the schedulable arm must NOT reject it.
_REACTIVE_ENABLED_NULL_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire)
VALUES
    ('trig_reactive', 'ses_mig', 'acc_mig', 'reactive', 'run_completion',
     '{"workflow_id": "wf_x", "statuses": ["completed"]}'::jsonb,
     '{"kind": "wake_owner", "content": "go"}'::jsonb,
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


async def _insert_raises(db_url: str, sql: str) -> bool:
    """Return True iff the statement raises a CHECK violation."""
    conn = await asyncpg.connect(db_url)
    try:
        try:
            await conn.execute(sql)
        except asyncpg.CheckViolationError:
            return True
        return False
    finally:
        await conn.close()


async def _fetchval(db_url: str, sql: str) -> object:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_and_clean_downgrade_round_trip(postgres: object) -> None:
    """With zero violator rows, 0130 upgrades and downgrades cleanly."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0130"], db_url)
    assert up.returncode == 0, f"upgrade to 0130 failed:\n{up.stderr}\n{up.stdout}"

    down = _run_alembic(["downgrade", "0129"], db_url)
    assert down.returncode == 0, f"downgrade to 0129 failed:\n{down.stderr}\n{down.stdout}"


@needs_docker
@pytest.mark.integration
def test_zombie_cron_rejected_under_new_check_accepted_under_old(postgres: object) -> None:
    """An enabled cron row with NULL next_fire (the #925 zombie) is rejected at
    head (0130) and admitted at the prior revision (0129) — the CHECK is what
    makes the state unrepresentable."""
    db_url = _alembic_url(postgres)

    # Under 0129 (prior head) the zombie is representable (the schema admits it).
    up129 = _run_alembic(["upgrade", "0129"], db_url)
    assert up129.returncode == 0, f"upgrade to 0129 failed:\n{up129.stderr}\n{up129.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _ZOMBIE_CRON_ROW_SQL))

    # Clean it up before upgrading (the migration would auto-disable it; here we
    # test the constraint directly on a fresh row, so remove the seeded zombie).
    asyncio.run(_execute(db_url, "DELETE FROM triggers WHERE id = 'trig_zombie'"))

    # Under 0130 the same row is rejected.
    up130 = _run_alembic(["upgrade", "0130"], db_url)
    assert up130.returncode == 0, f"upgrade to 0130 failed:\n{up130.stderr}\n{up130.stdout}"
    assert asyncio.run(_insert_raises(db_url, _ZOMBIE_CRON_ROW_SQL)), (
        "enabled cron with NULL next_fire must be rejected under 0130"
    )


@needs_docker
@pytest.mark.integration
def test_zombie_one_shot_rejected_under_new_check(postgres: object) -> None:
    """Settled fork 1: the schedulable arm covers one_shot too — an enabled
    one-shot with NULL next_fire is rejected under 0130."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0130"], db_url)
    assert up.returncode == 0, f"upgrade to 0130 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    assert asyncio.run(_insert_raises(db_url, _ZOMBIE_ONE_SHOT_ROW_SQL)), (
        "enabled one_shot with NULL next_fire must be rejected under 0130"
    )


@needs_docker
@pytest.mark.integration
def test_legitimate_rows_still_insert(postgres: object) -> None:
    """The CHECK admits every legitimate shape: an armed enabled cron, a
    disabled cron with NULL next_fire, and an enabled reactive (run_completion)
    row with NULL next_fire (excluded from the schedulable arm)."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0130"], db_url)
    assert up.returncode == 0, f"upgrade to 0130 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    # None of these raise.
    asyncio.run(_execute(db_url, _ARMED_CRON_ROW_SQL))
    asyncio.run(_execute(db_url, _DISABLED_CRON_ROW_SQL))
    asyncio.run(_execute(db_url, _REACTIVE_ENABLED_NULL_ROW_SQL))


@needs_docker
@pytest.mark.integration
def test_upgrade_auto_disables_preexisting_violators(postgres: object) -> None:
    """The 0066 lesson: pre-existing violator rows are DISABLED before the
    validating ADD CONSTRAINT, so the migrate succeeds on a table that already
    holds a #925 zombie — and the auto-disable is behaviorally a no-op (the row
    never fired), leaving the row present but ``enabled = false``."""
    db_url = _alembic_url(postgres)

    # Seed zombies (cron + one_shot) at the prior revision, then migrate.
    up129 = _run_alembic(["upgrade", "0129"], db_url)
    assert up129.returncode == 0, f"upgrade to 0129 failed:\n{up129.stderr}\n{up129.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _ZOMBIE_CRON_ROW_SQL))
    asyncio.run(_execute(db_url, _ZOMBIE_ONE_SHOT_ROW_SQL))
    # A legitimately armed row must survive untouched.
    asyncio.run(_execute(db_url, _ARMED_CRON_ROW_SQL))

    up130 = _run_alembic(["upgrade", "0130"], db_url)
    assert up130.returncode == 0, (
        f"upgrade to 0130 must not abort on pre-existing zombies:\n{up130.stderr}\n{up130.stdout}"
    )

    # The zombies are disabled (behavioral no-op), not deleted.
    for zid in ("trig_zombie", "trig_zombie_os"):
        enabled = asyncio.run(_fetchval(db_url, f"SELECT enabled FROM triggers WHERE id = '{zid}'"))
        assert enabled is False, f"{zid} should have been auto-disabled"
    # The armed row is untouched.
    armed = asyncio.run(_fetchval(db_url, "SELECT enabled FROM triggers WHERE id = 'trig_armed'"))
    assert armed is True
