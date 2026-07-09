"""Integration test: ``clone_session`` enforces the per-account enabled-trigger
cap (Plan 004).

Every OTHER write path that creates or re-enables a trigger enforces
``Settings.triggers_per_account_max`` under ``acquire_account_triggers_lock``
— ``add_trigger``, ``update_trigger``'s re-enable, and the create-session
attach path (see ``tests/e2e/test_triggers.py::TestPerAccountCap`` and
``TestAdvisoryLockSerializesCapCheck``). ``clone_session`` copies every parent
trigger row with ``enabled`` and ``next_fire`` preserved (``TRIGGERS_POLICY``
marks both ``Arm.COPY``), so without a cap check a tenant sitting at the cap
could multiply enabled, armed triggers without bound by repeatedly cloning a
trigger-bearing session.

This pins the fix: EVERY clone acquires the SAME advisory lock the sibling
paths use — unconditionally, before the parent ``FOR UPDATE`` — to match
their advisory-first / session-row-second lock order (a session-row-first
clone would AB/BA-deadlock a concurrent ``add_trigger``). With the lock held,
the parent's enabled count is exact through the copy, so a trigger-bearing
clone raises ``RateLimitedError`` if the account's existing enabled count
plus the clone's would exceed the cap — atomically with the copy (same
transaction), so a blocked clone leaves no partial state. A trigger-free
parent still takes the lock (for ordering) but skips the cap CHECK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool, register_jsonb_codec
from aios.db.queries.sessions import clone_session
from aios.errors import RateLimitedError
from aios.ids import SESSION, TRIGGER, make_id

pytestmark = pytest.mark.integration

ACCOUNT = "acc_clone_trigcap"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'clone-trigcap')",
                ACCOUNT,
            )
        yield pool
    finally:
        await pool.close()


async def _seed_agent_env(conn: asyncpg.Connection[Any]) -> tuple[str, str]:
    agent_id = make_id("agent")
    env_id = make_id("env")
    await conn.execute(
        "INSERT INTO agents (id, account_id, name, model, system, version) "
        "VALUES ($1, $2, 'a', 'openrouter/test', '', 1)",
        agent_id,
        ACCOUNT,
    )
    await conn.execute(
        "INSERT INTO environments (id, account_id, name) VALUES ($1, $2, 'e')",
        env_id,
        ACCOUNT,
    )
    return agent_id, env_id


async def _insert_session(
    conn: asyncpg.Connection[Any], agent_id: str, env_id: str, *, workspace: str
) -> str:
    session_id = make_id(SESSION)
    await conn.execute(
        """
        INSERT INTO sessions (id, agent_id, environment_id, agent_version, title,
            metadata, workspace_volume_path, env, account_id, last_event_seq)
        VALUES ($1, $2, $3, 1, 't', '{}'::jsonb, $4, '{}'::jsonb, $5, 0)
        """,
        session_id,
        agent_id,
        env_id,
        workspace,
        ACCOUNT,
    )
    return session_id


async def _insert_trigger(
    conn: asyncpg.Connection[Any], owner_session_id: str, *, name: str, enabled: bool
) -> str:
    """A ``cron`` trigger — armed (``next_fire`` set) when enabled, per the
    0130 ``triggers_schedulable_enabled_armed`` CHECK (enabled schedulable
    rows must carry a non-NULL ``next_fire``)."""
    trigger_id = make_id(TRIGGER)
    await conn.execute(
        f"""
        INSERT INTO triggers (id, owner_session_id, account_id, name, source,
            source_spec, action, enabled, next_fire)
        VALUES ($1, $2, $3, $4, 'cron', '{{"schedule":"* * * * *"}}'::jsonb,
            '{{"kind":"wake_owner","content":"hi"}}'::jsonb, $5, {"now()" if enabled else "NULL"})
        """,
        trigger_id,
        owner_session_id,
        ACCOUNT,
        name,
        enabled,
    )
    return trigger_id


async def _count_account_enabled_triggers(conn: asyncpg.Connection[Any]) -> int:
    result: int = await conn.fetchval(
        "SELECT COUNT(*) FROM triggers WHERE account_id = $1 AND enabled",
        ACCOUNT,
    )
    return result


def _set_cap(monkeypatch: Any, cap: int) -> None:
    """Patch the per-account trigger cap to ``cap``. ``clone_session`` reads it
    via ``aios.config.get_settings`` (a function-local import), so that is the
    symbol to patch."""
    from aios.config import Settings

    original = Settings()
    monkeypatch.setattr(
        "aios.config.get_settings",
        lambda: original.model_copy(update={"triggers_per_account_max": cap}),
    )


async def test_clone_blocked_at_cap(pool: asyncpg.Pool[Any], monkeypatch: Any) -> None:
    """Account already at the cap (via the parent's own enabled trigger) —
    cloning would add another enabled trigger and breach it. The whole clone
    must roll back: no new session, no new trigger rows."""
    _set_cap(monkeypatch, 1)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        parent = await _insert_session(conn, agent_id, env_id, workspace="/w/p")
        await _insert_trigger(conn, parent, name="cron1", enabled=True)

        session_count_before: int = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        trigger_count_before = await _count_account_enabled_triggers(conn)
        assert trigger_count_before == 1

        with pytest.raises(RateLimitedError, match="active-trigger cap"):
            await clone_session(conn, parent, account_id=ACCOUNT, workspace_path="/w/clone")

        session_count_after: int = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        assert session_count_after == session_count_before
        assert await _count_account_enabled_triggers(conn) == trigger_count_before


async def test_clone_allowed_under_cap(pool: asyncpg.Pool[Any], monkeypatch: Any) -> None:
    """Account well under the cap — clone succeeds and the copied triggers
    stay enabled + armed (existing copy semantics preserved)."""
    _set_cap(monkeypatch, 5)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        parent = await _insert_session(conn, agent_id, env_id, workspace="/w/p")
        await _insert_trigger(conn, parent, name="cron1", enabled=True)
        await _insert_trigger(conn, parent, name="cron2", enabled=True)

        clone = await clone_session(conn, parent, account_id=ACCOUNT, workspace_path="/w/clone")

        trows = await conn.fetch(
            "SELECT * FROM triggers WHERE owner_session_id = $1 ORDER BY name", clone.id
        )
        assert len(trows) == 2
        for row in trows:
            assert row["enabled"] is True
            assert row["next_fire"] is not None
        assert await _count_account_enabled_triggers(conn) == 4  # 2 parent + 2 clone


async def test_clone_trigger_free_unaffected_by_cap(
    pool: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """A parent with zero trigger rows clones fine even when the account is
    already AT the cap from an unrelated session — a trigger-free parent still
    takes the advisory lock (for ordering) but skips the cap CHECK, since its
    empty row set means the copy INSERT definitively copies nothing."""
    _set_cap(monkeypatch, 1)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)

        # A sibling session pins the account at the cap.
        capped = await _insert_session(conn, agent_id, env_id, workspace="/w/capped")
        await _insert_trigger(conn, capped, name="cron-capped", enabled=True)

        parent = await _insert_session(conn, agent_id, env_id, workspace="/w/p")
        # No triggers on the parent at all.

        clone = await clone_session(conn, parent, account_id=ACCOUNT, workspace_path="/w/clone")

        trows = await conn.fetch("SELECT * FROM triggers WHERE owner_session_id = $1", clone.id)
        assert trows == []


async def test_clone_disabled_only_unaffected_by_cap(
    pool: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """A parent whose triggers are all disabled clones fine regardless of the
    account cap — only enabled triggers count toward it, so the cap CHECK
    passes (enabled_to_clone=0) even though the parent owns trigger rows."""
    _set_cap(monkeypatch, 1)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)

        # A sibling session pins the account at the cap.
        capped = await _insert_session(conn, agent_id, env_id, workspace="/w/capped")
        await _insert_trigger(conn, capped, name="cron-capped", enabled=True)

        parent = await _insert_session(conn, agent_id, env_id, workspace="/w/p")
        await _insert_trigger(conn, parent, name="paused1", enabled=False)
        await _insert_trigger(conn, parent, name="paused2", enabled=False)

        clone = await clone_session(conn, parent, account_id=ACCOUNT, workspace_path="/w/clone")

        trows = await conn.fetch(
            "SELECT * FROM triggers WHERE owner_session_id = $1 ORDER BY name", clone.id
        )
        assert len(trows) == 2
        assert all(row["enabled"] is False for row in trows)


async def test_clone_blocked_below_cap_by_added_copies(
    pool: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """The account is STRICTLY BELOW the cap, but the clone's copies would push
    it over: cap=3, the parent is the account's only trigger owner with 2
    enabled → existing_account=2 (below 3), the clone would add 2 more →
    2 + 2 = 4 > 3, so it must raise and roll back. This isolates the
    load-bearing ``+ enabled_to_clone`` addend: a regression to a bare
    ``existing_account >= cap`` check would wrongly ALLOW this clone (2 < 3),
    so no other test catches it.

    (The parent must be the sole trigger owner because ``count_account_triggers``
    counts the parent's OWN enabled rows in ``existing_account`` — a sibling
    holding the other slots would push ``existing_account`` to/over the cap and
    make even the regressed ``>=`` check fire, defeating the discrimination.)"""
    _set_cap(monkeypatch, 3)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)

        parent = await _insert_session(conn, agent_id, env_id, workspace="/w/p")
        await _insert_trigger(conn, parent, name="cron1", enabled=True)
        await _insert_trigger(conn, parent, name="cron2", enabled=True)

        session_count_before: int = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        enabled_before = await _count_account_enabled_triggers(conn)
        assert enabled_before == 2  # parent's 2, the account's only enabled rows — below cap 3

        with pytest.raises(RateLimitedError, match="active-trigger cap"):
            await clone_session(conn, parent, account_id=ACCOUNT, workspace_path="/w/clone")

        # Whole clone rolled back: no new session, no copied triggers.
        session_count_after: int = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        assert session_count_after == session_count_before
        assert await _count_account_enabled_triggers(conn) == enabled_before


async def test_concurrent_clones_serialize_on_account_lock(
    pool: asyncpg.Pool[Any], monkeypatch: Any
) -> None:
    """Two concurrent clones of two DIFFERENT trigger-bearing parents (same
    account, one enabled slot left) must serialize on the account advisory
    lock: exactly one fits, the other raises ``RateLimitedError``. Distinct
    parents matter — clones of the same parent already serialize on its
    ``FOR UPDATE`` row lock, which would mask a missing advisory lock.
    Mirrors ``tests/e2e/test_triggers.py::TestAdvisoryLockSerializesCapCheck``
    for the clone path."""
    import asyncio

    _set_cap(monkeypatch, 3)

    async with pool.acquire() as conn:
        await register_jsonb_codec(conn)
        agent_id, env_id = await _seed_agent_env(conn)
        parents: list[str] = []
        for n in range(2):
            parent = await _insert_session(conn, agent_id, env_id, workspace=f"/w/p{n}")
            await _insert_trigger(conn, parent, name=f"cron{n}", enabled=True)
            parents.append(parent)
    # Account at 2/3 enabled; each clone would add 1 — only one fits.

    async def _clone(n: int, parent: str) -> bool:
        async with pool.acquire() as conn:
            await register_jsonb_codec(conn)
            try:
                await clone_session(conn, parent, account_id=ACCOUNT, workspace_path=f"/w/clone{n}")
                return True
            except RateLimitedError:
                return False

    results = await asyncio.gather(*(_clone(n, p) for n, p in enumerate(parents)))
    successes = sum(results)
    assert successes == 1, f"expected exactly 1 clone to fit the last slot, got {results}"

    async with pool.acquire() as conn:
        assert await _count_account_enabled_triggers(conn) == 3
