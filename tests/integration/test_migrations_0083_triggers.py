"""Integration tests for migration 0083 (session_scheduled_tasks → triggers).

The e2e suite (``tests/e2e/test_triggers.py``) runs against a DB already at
head, where 0083 created ``triggers`` empty — so the step-7 backfill of
PRE-EXISTING rows is never exercised there (the ``session_scheduled_tasks``
table is created empty and immediately renamed). These tests close that gap:
migrate to 0081, seed real old-shape rows, upgrade to 0083, and assert the
schedule→cron / fire_at→one_shot mapping, the ``to_char`` microsecond fire_at
serialization, the verbatim ``sandbox_command`` assembly (a schedule_wake-
origin ``tool wake_self`` row STAYS ``sandbox_command`` — ``wake_owner`` is
opt-in going forward, never backfilled), and the fail-hard validating SELECT.

Each test mutates ``alembic_version``, so the container is function-scoped.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# The FK chain required to land a session_scheduled_tasks row at revision 0081
# (columns introspected from a 0081 DB): accounts → environments → agents →
# sessions → session_scheduled_tasks. Only the NOT-NULL-without-default columns
# are supplied; everything else takes its column default.
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

# (a) cron, (b) one_shot, (c) schedule_wake-origin one_shot whose command is the
# `tool wake_self` idiom. Non-default timeout/output on (a)/(b) prove the
# action materialization copies the actual stored values, not the defaults.
_OLD_ROWS_SQL = r"""
INSERT INTO session_scheduled_tasks
    (id, session_id, account_id, name, schedule, command, timeout_seconds, max_output_bytes)
VALUES ('sched_cron', 'ses_mig', 'acc_mig', 'cron-row', '0 2 * * *', 'echo backup', 120, 2048);

INSERT INTO session_scheduled_tasks
    (id, session_id, account_id, name, fire_at, command, timeout_seconds, max_output_bytes)
VALUES ('sched_oneshot', 'ses_mig', 'acc_mig', 'oneshot-row',
        '2026-06-12 08:30:00.123456+00'::timestamptz, 'echo import', 600, 4096);

INSERT INTO session_scheduled_tasks
    (id, session_id, account_id, name, fire_at, command)
VALUES ('sched_wake', 'ses_mig', 'acc_mig', 'wake-row',
        '2026-06-12 09:00:00+00'::timestamptz,
        'tool wake_self ''{"content":"poll done"}''');
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


async def _fetch_triggers(db_url: str) -> dict[str, dict[str, Any]]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT id, owner_session_id, source, source_spec, action FROM triggers"
        )
        return {
            r["id"]: {
                "owner_session_id": r["owner_session_id"],
                "source": r["source"],
                "source_spec": json.loads(r["source_spec"]),
                "action": json.loads(r["action"]),
            }
            for r in rows
        }
    finally:
        await conn.close()


async def _table_exists(db_url: str, name: str) -> bool:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval("SELECT to_regclass($1)", f"public.{name}") is not None
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_backfill_maps_old_rows_verbatim(postgres: object) -> None:
    """Seed cron / one_shot / schedule_wake-origin rows at 0081, upgrade to 0083,
    assert the backfill mapping + verbatim sandbox_command assembly."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0081"], db_url)
    assert up.returncode == 0, f"upgrade to 0081 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _OLD_ROWS_SQL))

    up = _run_alembic(["upgrade", "0083"], db_url)
    assert up.returncode == 0, f"upgrade to 0083 failed:\n{up.stderr}\n{up.stdout}"

    # Table + owner column renamed.
    assert not asyncio.run(_table_exists(db_url, "session_scheduled_tasks"))
    assert asyncio.run(_table_exists(db_url, "triggers"))

    rows = asyncio.run(_fetch_triggers(db_url))
    assert set(rows) == {"sched_cron", "sched_oneshot", "sched_wake"}

    # (a) cron → source='cron', schedule carried; action=sandbox_command with the
    #     actual (non-default) timeout/output values materialized.
    cron = rows["sched_cron"]
    assert cron["owner_session_id"] == "ses_mig"
    assert cron["source"] == "cron"
    assert cron["source_spec"] == {"schedule": "0 2 * * *"}
    assert cron["action"] == {
        "kind": "sandbox_command",
        "command": "echo backup",
        "timeout_seconds": 120,
        "max_output_bytes": 2048,
    }

    # (b) one_shot → source='one_shot', fire_at serialized with to_char microsecond
    #     precision + trailing Z.
    one_shot = rows["sched_oneshot"]
    assert one_shot["source"] == "one_shot"
    assert one_shot["source_spec"] == {"fire_at": "2026-06-12T08:30:00.123456Z"}
    assert one_shot["action"]["command"] == "echo import"
    assert one_shot["action"]["timeout_seconds"] == 600
    assert one_shot["action"]["max_output_bytes"] == 4096

    # (c) schedule_wake-origin row (command is `tool wake_self …`) STAYS
    #     sandbox_command — wake_owner is opt-in going forward, never backfilled.
    wake = rows["sched_wake"]
    assert wake["source"] == "one_shot"
    assert wake["action"]["kind"] == "sandbox_command"
    assert wake["action"]["command"] == 'tool wake_self \'{"content":"poll done"}\''


@needs_docker
@pytest.mark.integration
def test_validating_select_fails_hard_on_malformed_row(postgres: object) -> None:
    """A pre-0083 row that backfills to an invalid shape (both schedule and
    fire_at NULL — only reachable by dropping the 0059 XOR) is named by the
    in-migration validating SELECT, which aborts the upgrade BEFORE the new
    columns are made NOT NULL / the CHECKs are added."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0081"], db_url)
    assert up.returncode == 0, f"upgrade to 0081 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    # Drop the 0059 XOR so a both-NULL row can land; the backfill turns it into a
    # one_shot with a json-null fire_at — exactly what the validating SELECT
    # exists to catch.
    asyncio.run(
        _execute(
            db_url,
            "ALTER TABLE session_scheduled_tasks "
            "DROP CONSTRAINT sched_tasks_schedule_xor_fire_at;"
            "INSERT INTO session_scheduled_tasks (id, session_id, account_id, name, command) "
            "VALUES ('sched_bad', 'ses_mig', 'acc_mig', 'malformed', 'echo x');",
        )
    )

    up = _run_alembic(["upgrade", "0083"], db_url)
    assert up.returncode != 0, f"upgrade should have failed loud:\n{up.stdout}"
    assert "violating the shape contract" in up.stderr
    assert "sched_bad" in up.stderr

    # The abort happened before the rename committed — the transaction rolled
    # back, so the old table is intact and `triggers` was never created.
    assert asyncio.run(_table_exists(db_url, "session_scheduled_tasks"))
    assert not asyncio.run(_table_exists(db_url, "triggers"))
