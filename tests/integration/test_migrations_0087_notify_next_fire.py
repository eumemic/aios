"""Integration tests for migration 0087 (NOTIFY on ``next_fire`` change, #940).

The pre-0087 ``notify_scheduled_tasks_due`` UPDATE gate fired only on
source / source_spec / enabled / running_since→NULL changes — NOT on a bare
``next_fire`` edit. Any path that reschedules a row without touching those
columns left the sleeping event-driven scheduler unaware for up to
``_HEARTBEAT_SECONDS``. 0087 adds ``OR OLD.next_fire IS DISTINCT FROM
NEW.next_fire`` to that gate.

These tests migrate a real Postgres to 0087, seed one valid ``triggers`` row,
and assert a PURE ``next_fire`` UPDATE produces a NOTIFY on
``aios_scheduled_tasks_due`` — and that downgrading to 0086 restores the
silent behavior. The e2e suite runs at head where ``triggers`` is created
empty, so the NOTIFY-on-next_fire edge is only exercised here.

Each test mutates ``alembic_version``, so the container is function-scoped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# FK chain required to land a ``triggers`` row at revision 0087:
# accounts → environments → agents → sessions → triggers. Only NOT-NULL-
# without-default columns are supplied; everything else takes its default.
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

# One valid cron trigger. ``source_spec`` satisfies the 0083 SOURCE_SPEC CHECK
# (cron → schedule string, no fire_at); ``action`` satisfies the ACTION CHECK
# (sandbox_command → command/timeout_seconds/max_output_bytes, no content).
# ``enabled`` defaults TRUE.
_TRIGGER_SQL = """
INSERT INTO triggers
    (id, owner_session_id, account_id, name, source, source_spec, action, next_fire)
VALUES (
    'trg_nf', 'ses_mig', 'acc_mig', 'nf-row', 'cron',
    '{"schedule":"*/5 * * * *"}'::jsonb,
    '{"kind":"sandbox_command","command":"echo hi","timeout_seconds":60,"max_output_bytes":65536}'::jsonb,
    now() + interval '1 hour'
);
"""

# Pure ``next_fire`` edit — touches NOTHING else.
_NEXT_FIRE_ONLY_UPDATE = (
    "UPDATE triggers SET next_fire = next_fire + interval '7 minutes' WHERE id = 'trg_nf'"
)


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


async def _next_fire_update_notifies(db_url: str, *, timeout_s: float) -> bool:
    """LISTEN on ``aios_scheduled_tasks_due``, run a pure ``next_fire`` UPDATE
    on a SEPARATE connection, and wait up to ``timeout_s`` for a notification.

    Returns True if a NOTIFY arrived, False on timeout.
    """
    got = asyncio.Event()

    def _on_notify(*_args: object) -> None:
        got.set()

    listen_conn = await asyncpg.connect(db_url)
    try:
        await listen_conn.add_listener("aios_scheduled_tasks_due", _on_notify)
        # Separate connection for the write — a notify is delivered to the
        # listening session regardless, but keeping them distinct mirrors the
        # real scheduler topology (dedicated LISTEN conn vs. the app pool).
        await _execute(db_url, _NEXT_FIRE_ONLY_UPDATE)
        try:
            await asyncio.wait_for(got.wait(), timeout=timeout_s)
            return True
        except TimeoutError:
            return False
    finally:
        await listen_conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_0087_next_fire_only_update_notifies(postgres: object) -> None:
    """At 0087, a pure ``next_fire`` UPDATE emits a NOTIFY within ~2s."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0087"], db_url)
    assert up.returncode == 0, f"upgrade to 0087 failed:\n{up.stderr}\n{up.stdout}"

    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _TRIGGER_SQL))

    assert asyncio.run(_next_fire_update_notifies(db_url, timeout_s=2.0)), (
        "next_fire-only UPDATE did not produce a NOTIFY at revision 0087"
    )


@needs_docker
@pytest.mark.integration
def test_downgrade_0086_next_fire_only_update_does_not_notify(postgres: object) -> None:
    """Downgrading to 0086 restores the pre-fix body: a pure ``next_fire``
    UPDATE is silent (mirrors the action-only negative test)."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0087"], db_url)
    assert up.returncode == 0, f"upgrade to 0087 failed:\n{up.stderr}\n{up.stdout}"

    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _TRIGGER_SQL))

    down = _run_alembic(["downgrade", "0086"], db_url)
    assert down.returncode == 0, f"downgrade to 0086 failed:\n{down.stderr}\n{down.stdout}"

    assert not asyncio.run(_next_fire_update_notifies(db_url, timeout_s=0.5)), (
        "next_fire-only UPDATE produced a NOTIFY after downgrade to 0086"
    )
