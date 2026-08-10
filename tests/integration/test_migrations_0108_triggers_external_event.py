"""Integration tests for migration 0108 (triggers external_event source, #1281).

Owns the migration mechanics: a clean up/down round-trip, the fail-hard
downgrade refusal once an ``external_event`` row exists, that an
``external_event`` row INSERTs under the new shape CHECK and is REJECTED under
the old, the ingest-token iff constraint and the unique-hash index, the
broadened reactive-no-next-fire guard, the extended ``trigger_runs``
trigger_context CHECK, and that every pre-existing source kind still inserts.

Each test mutates ``alembic_version``, so the container is function-scoped.
Modeled on tests/integration/test_migrations_0107_triggers_wake_session.py.
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

# An external_event trigger in its first-shipped shape: {}-source_spec, NULL
# next_fire (reactive — unschedulable), a wake_owner action (no environment_id,
# so the iff constraint holds: left false = right false), and a stored
# ingest_token_hash (required by the ingest-token iff constraint).
_EXTERNAL_EVENT_ROW_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action,
     enabled, next_fire, ingest_token_hash)
VALUES
    ('trig_ee', 'ses_mig', 'acc_mig', 'ee-row', 'external_event',
     '{}'::jsonb,
     '{"kind": "wake_owner", "content": "go look"}'::jsonb,
     TRUE, NULL, 'deadbeef');
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
    """Return True iff the statement raises an integrity error (CHECK/unique)."""
    conn = await asyncpg.connect(db_url)
    try:
        try:
            await conn.execute(sql)
        except (asyncpg.CheckViolationError, asyncpg.UniqueViolationError):
            return True
        return False
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_and_clean_downgrade_round_trip(postgres: object) -> None:
    """With zero external_event rows, 0108 upgrades and downgrades cleanly."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"

    down = _run_alembic(["downgrade", "0107"], db_url)
    assert down.returncode == 0, f"downgrade to 0107 failed:\n{down.stderr}\n{down.stdout}"


@needs_docker
@pytest.mark.integration
def test_cron_row_with_timezone_key_accepted_under_check(postgres: object) -> None:
    """Zero-migration proof for CronSource.timezone (#1378): a cron row carrying
    a ``timezone`` key in ``source_spec`` satisfies ``triggers_source_spec_shape``
    with NO DDL — the cron arm only asserts ``schedule`` is a string and
    ``fire_at`` is absent, and the COALESCE wrapper tolerates additive keys."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    tz_cron = (
        "INSERT INTO triggers "
        "(id, owner_session_id, account_id, name, source, source_spec, action, "
        "enabled, next_fire) VALUES "
        "('trig_tzcron', 'ses_mig', 'acc_mig', 'tz-cron', 'cron', "
        '\'{"schedule": "0 9 * * *", "timezone": "America/New_York"}\'::jsonb, '
        '\'{"kind": "wake_owner", "content": "hi"}\'::jsonb, '
        "TRUE, now())"
    )
    # Succeeds (no exception) — the additive timezone key passes the CHECK.
    asyncio.run(_execute(db_url, tz_cron))


@needs_docker
@pytest.mark.integration
def test_external_event_row_accepted_under_new_check_rejected_under_old(postgres: object) -> None:
    """An external_event row INSERTs at head (0108) and is rejected at the prior
    revision (0107) — the shape CHECK swap is what makes the kind representable."""
    db_url = _alembic_url(postgres)

    # Under 0107 (prior head) the row is unrepresentable: source_spec ELSE false
    # AND there is no ingest_token_hash column.
    up107 = _run_alembic(["upgrade", "0107"], db_url)
    assert up107.returncode == 0, f"upgrade to 0107 failed:\n{up107.stderr}\n{up107.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    # Column does not exist yet → UndefinedColumnError (not a CHECK violation).
    assert asyncio.run(_column_absent_or_rejected(db_url, _EXTERNAL_EVENT_ROW_SQL)), (
        "external_event row must not be insertable under 0107"
    )

    # Under 0108 the same row inserts.
    up108 = _run_alembic(["upgrade", "0108"], db_url)
    assert up108.returncode == 0, f"upgrade to 0108 failed:\n{up108.stderr}\n{up108.stdout}"
    asyncio.run(_execute(db_url, _EXTERNAL_EVENT_ROW_SQL))


async def _column_absent_or_rejected(db_url: str, sql: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        try:
            await conn.execute(sql)
        except (
            asyncpg.CheckViolationError,
            asyncpg.UndefinedColumnError,
        ):
            return True
        return False
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_ingest_token_iff_external_event_constraint(postgres: object) -> None:
    """The iff constraint binds both directions: an external_event row WITHOUT a
    hash is rejected, and a non-external_event row WITH a hash is rejected."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    # external_event with NULL hash → violation.
    ee_no_hash = """
    INSERT INTO triggers
        (id, owner_session_id, account_id, name, source, source_spec, action,
         enabled, next_fire, ingest_token_hash)
    VALUES ('trig_eebad', 'ses_mig', 'acc_mig', 'ee-bad', 'external_event',
            '{}'::jsonb, '{"kind": "wake_owner", "content": "x"}'::jsonb,
            TRUE, NULL, NULL);
    """
    assert asyncio.run(_insert_raises(db_url, ee_no_hash))

    # cron WITH a hash → violation.
    cron_with_hash = """
    INSERT INTO triggers
        (id, owner_session_id, account_id, name, source, source_spec, action,
         enabled, next_fire, ingest_token_hash)
    VALUES ('trig_cronbad', 'ses_mig', 'acc_mig', 'cron-bad', 'cron',
            '{"schedule": "*/5 * * * *"}'::jsonb,
            '{"kind": "wake_owner", "content": "x"}'::jsonb,
            TRUE, now(), 'deadbeef');
    """
    assert asyncio.run(_insert_raises(db_url, cron_with_hash))


@needs_docker
@pytest.mark.integration
def test_ingest_token_hash_is_unique(postgres: object) -> None:
    """The partial unique index is the double-mint collision guard."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _EXTERNAL_EVENT_ROW_SQL))  # hash 'deadbeef'

    dup = """
    INSERT INTO triggers
        (id, owner_session_id, account_id, name, source, source_spec, action,
         enabled, next_fire, ingest_token_hash)
    VALUES ('trig_dup', 'ses_mig', 'acc_mig', 'ee-dup', 'external_event',
            '{}'::jsonb, '{"kind": "wake_owner", "content": "x"}'::jsonb,
            TRUE, NULL, 'deadbeef');
    """
    assert asyncio.run(_insert_raises(db_url, dup))


@needs_docker
@pytest.mark.integration
def test_external_event_with_next_fire_rejected(postgres: object) -> None:
    """The broadened reactive-no-next-fire guard forbids a non-NULL next_fire on
    an external_event row (same carve-out as run_completion)."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    with_next_fire = """
    INSERT INTO triggers
        (id, owner_session_id, account_id, name, source, source_spec, action,
         enabled, next_fire, ingest_token_hash)
    VALUES ('trig_eenf', 'ses_mig', 'acc_mig', 'ee-nf', 'external_event',
            '{}'::jsonb, '{"kind": "wake_owner", "content": "x"}'::jsonb,
            TRUE, now(), 'cafe1234');
    """
    assert asyncio.run(_insert_raises(db_url, with_next_fire))


@needs_docker
@pytest.mark.integration
def test_trigger_runs_external_event_context_accepted(postgres: object) -> None:
    """The extended trigger_runs.trigger_context CHECK admits 'external_event'
    (rejected before 0108)."""
    db_url = _alembic_url(postgres)

    # Prior revision rejects the new context value.
    up107 = _run_alembic(["upgrade", "0107"], db_url)
    assert up107.returncode == 0, f"upgrade to 0107 failed:\n{up107.stderr}\n{up107.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    run_row = """
    INSERT INTO trigger_runs
        (id, trigger_id, account_id, owner_session_id, trigger_name,
         trigger_context, event, status)
    VALUES ('trun_ee', 'trig_x', 'acc_mig', 'ses_mig', 'ee-row',
            'external_event', '{"a": 1}'::jsonb, 'pending');
    """
    assert asyncio.run(_insert_raises(db_url, run_row))

    # Head admits it.
    up108 = _run_alembic(["upgrade", "0108"], db_url)
    assert up108.returncode == 0, f"upgrade to 0108 failed:\n{up108.stderr}\n{up108.stdout}"
    asyncio.run(_execute(db_url, run_row))


@needs_docker
@pytest.mark.integration
def test_every_prior_source_kind_still_inserts_under_new_check(postgres: object) -> None:
    """The three pre-existing source branches stay byte-identical, so every
    prior source kind still inserts under the 0108 shape CHECK."""
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))

    rows = {
        "cron": (
            "'cron'",
            '\'{"schedule": "*/5 * * * *"}\'::jsonb',
            "now()",
        ),
        "one_shot": (
            "'one_shot'",
            '\'{"fire_at": "2999-01-01T00:00:00+00:00"}\'::jsonb',
            "now()",
        ),
        "run_completion": (
            "'run_completion'",
            '\'{"workflow_id": "wf_x", "statuses": ["completed"]}\'::jsonb',
            "NULL",
        ),
    }

    async def _insert_all() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            for name, (src, spec, next_fire) in rows.items():
                await conn.execute(
                    "INSERT INTO triggers (id, owner_session_id, account_id, name, source, "
                    "source_spec, action, enabled, next_fire) VALUES "
                    f"('trig_{name}', 'ses_mig', 'acc_mig', '{name}', {src}, {spec}, "
                    f'\'{{"kind": "wake_owner", "content": "hi"}}\'::jsonb, '
                    f"TRUE, {next_fire})"
                )
        finally:
            await conn.close()

    asyncio.run(_insert_all())


@needs_docker
@pytest.mark.integration
def test_downgrade_refuses_external_event_rows(postgres: object) -> None:
    """An external_event row is unrepresentable under the prior predicate, so the
    downgrade fails hard and rolls back (the 0086 stance)."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _EXTERNAL_EVENT_ROW_SQL))

    down = _run_alembic(["downgrade", "0107"], db_url)
    assert down.returncode != 0, f"downgrade should have failed loud:\n{down.stdout}"
    assert "cannot downgrade" in down.stderr
