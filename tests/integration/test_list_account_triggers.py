"""Account-wide trigger read (#1673) — ``queries.list_account_triggers`` against
a real Postgres.

This is the filed blocking precondition for the ops-agent O7 trigger-liveness
auditor: a workflow run in an account must be able to enumerate EVERY enabled
trigger across that account (each sentinel on its own session) and read each
one's ``next_fire`` — a defense-in-depth liveness read. (#1678 made the #925
zombie class ``enabled=true, next_fire=NULL`` cron rows unrepresentable at write
time via the ``triggers_schedulable_enabled_armed`` CHECK, so the read is now a
liveness backstop, not the sole corrective.)

Owns the account-scope obligations:
- account-wide: triggers from *different* sessions in the account are all seen.
- the #925 zombie is now unrepresentable: the raw ``UPDATE`` that forced an
  enabled cron row to ``next_fire=NULL`` is rejected by the schedulable arm
  (#1678) — there is no zombie for the read to surface.
- archived-session rows are filtered (they can never fire, same as the
  scheduler's claim/MIN queries).
- ``enabled_only`` (default True) excludes disabled rows; False includes them.
- cross-account isolation: another account's triggers are never returned.
- the projected shape carries ``owner_session_id`` + the ``source_kind``
  discriminator the auditor branches on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import runtime
from aios.models.triggers import TriggerCreate
from aios.services import triggers as trig_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

ACC = "acc_lat_root"
OTHER = "acc_lat_child"

_CRON_BODY = {
    "name": "nightly",
    "source": {"kind": "cron", "schedule": "0 3 * * *"},
    "action": {"kind": "wake_owner", "content": "audit"},
    "enabled": True,
}


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                f"VALUES ('{ACC}', NULL, TRUE, 'lat-root'), "
                f"('{OTHER}', '{ACC}', FALSE, 'lat-child')"
            )
        yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _add(
    pool: asyncpg.Pool[Any], session_id: str, *, account_id: str, **overrides: Any
) -> str:
    body = {**_CRON_BODY, **overrides}
    echo = await trig_service.add_trigger(
        pool, session_id, TriggerCreate.model_validate(body), account_id=account_id
    )
    return echo.id


async def test_account_wide_across_sessions(pool: asyncpg.Pool[Any]) -> None:
    """The core acceptance: triggers owned by DIFFERENT sessions in the account
    are all enumerated by one account-scoped read — the session-scoped
    ``list_triggers`` cannot do this."""
    _, _, s1 = await seed_agent_env_session(pool, account_id=ACC, prefix="a1")
    _, _, s2 = await seed_agent_env_session(pool, account_id=ACC, prefix="a2")
    await _add(pool, s1.id, account_id=ACC, name="t1")
    await _add(pool, s2.id, account_id=ACC, name="t2")

    async with pool.acquire() as conn:
        echoes = await queries.list_account_triggers(conn, account_id=ACC)

    owners = {e.owner_session_id for e in echoes}
    names = {e.name for e in echoes}
    assert owners == {s1.id, s2.id}  # both sessions' triggers, one read
    assert names == {"t1", "t2"}


async def test_zombie_next_fire_null_is_unrepresentable(pool: asyncpg.Pool[Any]) -> None:
    """The #925 zombie the feature originally surfaced is now unrepresentable
    (#1678): the raw ``UPDATE triggers SET next_fire = NULL`` that forced an
    enabled cron row into the incident state is rejected at write time by the
    ``triggers_schedulable_enabled_armed`` CHECK — so there is no zombie for the
    auditor read to surface. The read remains a defense-in-depth liveness
    backstop; the enumeration itself is exercised by the tests above."""
    _, _, s = await seed_agent_env_session(pool, account_id=ACC, prefix="z")
    tid = await _add(pool, s.id, account_id=ACC, name="zombie")
    # The #925 incident-recovery anti-pattern (enabled cron + next_fire NULL) now
    # fails loudly at the offending write instead of parking a dead-but-alive row.
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError, match="schedulable_enabled_armed"):
            await conn.execute("UPDATE triggers SET next_fire = NULL WHERE id = $1", tid)

    # The row is untouched — still armed, still enabled, still enumerable.
    async with pool.acquire() as conn:
        echoes = await queries.list_account_triggers(conn, account_id=ACC)
    row = next(e for e in echoes if e.name == "zombie")
    assert row.enabled is True
    assert row.next_fire is not None  # the arm held: no zombie was written
    assert row.source_kind == "cron"


async def test_archived_session_rows_are_filtered(pool: asyncpg.Pool[Any]) -> None:
    """A trigger on an archived session can never fire (the scheduler filters
    ``s.archived_at IS NULL``), so it isn't part of the live-liveness population."""
    _, _, live = await seed_agent_env_session(pool, account_id=ACC, prefix="live")
    _, _, dead = await seed_agent_env_session(pool, account_id=ACC, prefix="dead")
    await _add(pool, live.id, account_id=ACC, name="live-t")
    await _add(pool, dead.id, account_id=ACC, name="dead-t")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archived_at = now() WHERE id = $1", dead.id)
        echoes = await queries.list_account_triggers(conn, account_id=ACC)

    assert {e.name for e in echoes} == {"live-t"}


async def test_enabled_only_flag(pool: asyncpg.Pool[Any]) -> None:
    """Default excludes disabled rows (the population the non-null invariant is
    about); ``enabled_only=False`` includes them."""
    _, _, s = await seed_agent_env_session(pool, account_id=ACC, prefix="en")
    await _add(pool, s.id, account_id=ACC, name="on", enabled=True)
    await _add(pool, s.id, account_id=ACC, name="off", enabled=False)

    async with pool.acquire() as conn:
        default = await queries.list_account_triggers(conn, account_id=ACC)
        allrows = await queries.list_account_triggers(conn, account_id=ACC, enabled_only=False)

    assert {e.name for e in default} == {"on"}
    assert {e.name for e in allrows} == {"on", "off"}


async def test_cross_account_isolation(pool: asyncpg.Pool[Any]) -> None:
    """A read scoped to ACC never returns another account's triggers — the
    isolation boundary the run path (``account_id=run.account_id``) rides on."""
    _, _, mine = await seed_agent_env_session(pool, account_id=ACC, prefix="mine")
    _, _, theirs = await seed_agent_env_session(pool, account_id=OTHER, prefix="theirs")
    await _add(pool, mine.id, account_id=ACC, name="mine-t")
    await _add(pool, theirs.id, account_id=OTHER, name="theirs-t")

    async with pool.acquire() as conn:
        echoes = await queries.list_account_triggers(conn, account_id=ACC)

    assert {e.name for e in echoes} == {"mine-t"}
    assert all(e.owner_session_id == mine.id for e in echoes)


async def test_run_completion_source_kind_projected(pool: asyncpg.Pool[Any]) -> None:
    """``source_kind`` carries the discriminator the auditor branches on to EXEMPT
    reactive sources (``run_completion`` legitimately has ``next_fire=NULL``)."""
    from aios.db.queries import workflows as wf_queries

    _, _, s = await seed_agent_env_session(pool, account_id=ACC, prefix="rc")
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id=ACC, name="w-rc", script="async def main(input):\n    return {}\n"
        )
    await _add(
        pool,
        s.id,
        account_id=ACC,
        name="on-complete",
        source={"kind": "run_completion", "workflow_id": wf.id, "statuses": ["completed"]},
        action={"kind": "wake_owner", "content": "audit"},
    )

    async with pool.acquire() as conn:
        echoes = await queries.list_account_triggers(conn, account_id=ACC)

    rc = next(e for e in echoes if e.name == "on-complete")
    assert rc.source_kind == "run_completion"
    assert rc.next_fire is None  # reactive → legitimately null → the auditor exempts it


# ─── operator triggers (#2473) are part of the account population (#2525) ────


async def _operator_trigger(
    pool: asyncpg.Pool[Any], env_id: str, *, name: str, enabled: bool = True
) -> str:
    from aios.db.queries import workflows as wf_queries
    from aios.models.triggers import OperatorTriggerCreate

    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn,
            account_id=ACC,
            name=f"w-{name}",
            script="async def main(input):\n    return {}\n",
        )
    echo = await trig_service.add_operator_trigger(
        pool,
        OperatorTriggerCreate.model_validate(
            {
                "name": name,
                "source": {"kind": "cron", "schedule": "0 4 * * *"},
                "action": {"kind": "workflow", "workflow_id": wf.id, "budget_usd": 1.5},
                "environment_id": env_id,
                "enabled": enabled,
            }
        ),
        account_id=ACC,
    )
    return echo.id


async def test_operator_triggers_are_listed_with_owner_kind(pool: asyncpg.Pool[Any]) -> None:
    """An operator trigger has no session (``owner_session_id`` NULL). An INNER JOIN on
    ``sessions`` dropped it, so every account-wide liveness monitor was blind to it
    (#2525). It is returned under both ``enabled_only`` settings, tagged ``operator``,
    alongside the session triggers, which are tagged ``session``."""
    _, env, s = await seed_agent_env_session(pool, account_id=ACC, prefix="opl")
    await _add(pool, s.id, account_id=ACC, name="sess-t")
    op_on = await _operator_trigger(pool, env.id, name="op-on")
    op_off = await _operator_trigger(pool, env.id, name="op-off", enabled=False)

    async with pool.acquire() as conn:
        allrows = await queries.list_account_triggers(conn, account_id=ACC, enabled_only=False)
        default = await queries.list_account_triggers(conn, account_id=ACC)

    by_name = {e.name: e for e in allrows}
    assert set(by_name) == {"sess-t", "op-on", "op-off"}
    assert (by_name["op-on"].id, by_name["op-off"].id) == (op_on, op_off)
    for name in ("op-on", "op-off"):
        assert by_name[name].owner_kind == "operator"
        assert by_name[name].owner_session_id is None
        assert by_name[name].source_kind == "cron"
    assert by_name["op-on"].next_fire is not None
    assert by_name["sess-t"].owner_kind == "session"
    assert by_name["sess-t"].owner_session_id == s.id

    assert {e.name for e in default} == {"sess-t", "op-on"}
    # The JSON the tool and run surfaces return carries an explicit null.
    dumped = by_name["op-on"].model_dump(mode="json")
    assert dumped["owner_kind"] == "operator" and dumped["owner_session_id"] is None


async def test_operator_trigger_survives_archiving_a_session(pool: asyncpg.Pool[Any]) -> None:
    """The archived-session filter still drops session triggers and only them: an
    operator trigger is not owned by the archived session, so it stays listed."""
    _, env, dead = await seed_agent_env_session(pool, account_id=ACC, prefix="opdead")
    await _add(pool, dead.id, account_id=ACC, name="dead-t")
    await _operator_trigger(pool, env.id, name="op-live")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archived_at = now() WHERE id = $1", dead.id)
        echoes = await queries.list_account_triggers(conn, account_id=ACC, enabled_only=False)

    assert [(e.name, e.owner_kind) for e in echoes] == [("op-live", "operator")]


async def test_another_accounts_operator_triggers_are_not_listed(
    pool: asyncpg.Pool[Any],
) -> None:
    _, env, _ = await seed_agent_env_session(pool, account_id=ACC, prefix="opiso")
    await _operator_trigger(pool, env.id, name="op-mine")
    async with pool.acquire() as conn:
        echoes = await queries.list_account_triggers(conn, account_id=OTHER, enabled_only=False)

    assert echoes == []
