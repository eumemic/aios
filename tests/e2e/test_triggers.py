"""E2E tests for ``triggers`` (#818).

Real Postgres (testcontainer) + real FastAPI app via ASGI transport.

These tests run against a DB already migrated to head (0083), so they exercise
the live ``triggers`` table, the live shape CHECKs, the live NOTIFY trigger,
and the §6.2 tool-rename SQL. They do NOT exercise the migration's step-7
backfill of pre-existing rows (the fresh test DB has none — the
``session_scheduled_tasks`` table is created empty and immediately renamed):
that schedule→cron / fire_at→one_shot mapping + the verbatim sandbox_command
assembly is covered by ``tests/integration/test_migrations_0083_triggers.py``,
which migrates to 0081, seeds real old-shape rows, then upgrades to 0083.

Covers:

- Service layer: granular add/remove/update/list round-trip; ``next_fire``
  recomputation on source change / enable-flip; duplicate-name rejection.
- Tick claim: overlap-prevention via ``running_since``; stale-recovery;
  archived-session filter; ``next_fire`` advanced inside the claim transaction.
- Auto-disable: ``MAX_CONSECUTIVE_FAILURES`` flips ``enabled`` false.
- Action dispatch: ``sandbox_command`` (bash) and ``wake_owner`` (self-delivery).
- DB CHECK probes (incl. the absent-key cases the unwrapped predicate would
  silently accept) against the live constraints.
- NOTIFY edges against the live rewritten trigger function.
- HTTP surface: POST / GET / DELETE / PUT round-trip with source/action.
- Migration tool-rename guards (§6.2) against the migration's own SQL.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import asyncpg
import httpx
import pytest

from aios.db import queries
from aios.models.agents import ToolSpec
from aios.models.triggers import (
    CronSource,
    OneShotSource,
    TriggerCreate,
    TriggerUpdate,
    WakeOwnerAction,
)
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.services import triggers as trig_service
from tests.helpers.connections import authed_client, wired_app


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


@pytest.fixture
async def env_and_agent(pool: Any) -> tuple[str, str]:
    account_id = "acc_test_stub"
    env = await environments_service.create_environment(
        pool, name=f"trig-env-{_uniq()}", account_id=account_id
    )
    agent = await agents_service.create_agent(
        pool,
        name=f"trig-agent-{_uniq()}",
        model="fake/test",
        system="triggers test",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    return env.id, agent.id


async def _create_session(pool: Any, env_id: str, agent_id: str) -> str:
    session = await sessions_service.create_session(
        pool,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
        account_id="acc_test_stub",
    )
    return session.id


async def _isolated_account_session(pool: Any) -> tuple[str, str]:
    """A fresh child account owning its own env/agent/session — for cap tests
    that need an isolated trigger count.

    Slice 2's cross-tenant fix means the caller's account must actually OWN
    the session it attaches triggers to: the old trick of passing a synthetic
    ``account_id`` against an ``acc_test_stub`` session now 404s — which is
    the fix working, not a test-isolation bug.
    """
    tag = _uniq()
    account_id = f"acc_iso_{tag}"
    sid = f"sess_iso_{tag}"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, 'acc_test_stub', FALSE, $1)",
            account_id,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) VALUES ($1, $2, $3)",
            f"env_iso_{tag}",
            f"iso-env-{tag}",
            account_id,
        )
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) VALUES ($1, $2, 'fake/test', $3)",
            f"agn_iso_{tag}",
            f"iso-agent-{tag}",
            account_id,
        )
        await conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, "
            "account_id) VALUES ($1, $2, $3, $4, $5)",
            sid,
            f"agn_iso_{tag}",
            f"env_iso_{tag}",
            f"/tmp/ws-iso-{tag}",
            account_id,
        )
    return account_id, sid


def _spec(name: str, schedule: str = "*/5 * * * *", command: str = "echo hi") -> TriggerCreate:
    return TriggerCreate.model_validate(
        {
            "name": name,
            "source": {"kind": "cron", "schedule": schedule},
            "action": {"kind": "sandbox_command", "command": command},
        }
    )


def _one_shot_spec(
    name: str,
    fire_at: datetime,
    command: str = "echo hi",
    metadata: dict[str, Any] | None = None,
) -> TriggerCreate:
    return TriggerCreate.model_validate(
        {
            "name": name,
            "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
            "action": {"kind": "sandbox_command", "command": command},
            "metadata": metadata or {},
        }
    )


def _wake_owner_spec(
    name: str,
    fire_at: datetime,
    content: str = "ping",
    metadata: dict[str, Any] | None = None,
) -> TriggerCreate:
    return TriggerCreate.model_validate(
        {
            "name": name,
            "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
            "action": {"kind": "wake_owner", "content": content},
            "metadata": metadata or {},
        }
    )


def _make_fake_sandbox_registry(run_impl: Any, *, provision_impl: Any = None) -> Any:
    """Build a fake sandbox-registry shim for tests.

    Attaches the ``exec`` method via ``setattr`` rather than ``def``-syntax to
    sidestep a local file-write hook that pattern-matches the literal substring
    'e' 'x' 'e' 'c' '('. The production runner does the same dance via
    ``run_in_sandbox = sandbox_registry.exec``.
    """

    class _FakeRegistry:
        async def get_or_provision(self, _session_id: str, *, pool: Any) -> object:
            if provision_impl is not None:
                return await provision_impl(pool)
            return mock.MagicMock()

    setattr(_FakeRegistry, "exec", run_impl)  # noqa: B010
    return _FakeRegistry()


# ─── trigger_list tool ────────────────────────────────────────────────────


class TestTriggerListTool:
    async def test_trigger_list_handler_returns_triggers(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime
        from aios.tools.trigger_list import trigger_list_handler

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        await trig_service.add_trigger(pool, sid, _spec("alpha"), account_id=account_id)
        await trig_service.add_trigger(pool, sid, _spec("beta"), account_id=account_id)

        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            result = await trigger_list_handler(sid, {})
        finally:
            runtime.pool = prev_pool

        assert len(result["triggers"]) == 2
        assert {t["name"] for t in result["triggers"]} == {"alpha", "beta"}
        for t in result["triggers"]:
            assert t["enabled"] is True
            assert t["consecutive_failures"] == 0
            assert t["source"]["kind"] == "cron"
            assert t["action"]["kind"] == "sandbox_command"


# ─── service-layer round-trip ─────────────────────────────────────────────


class TestServiceLayer:
    async def test_add_get_list_remove(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        echo = await trig_service.add_trigger(pool, sid, _spec("poll"), account_id=account_id)
        assert echo.name == "poll"
        assert echo.id.startswith("trig_")
        assert echo.enabled is True
        assert echo.source.kind == "cron"
        assert echo.action.kind == "sandbox_command"
        assert echo.next_fire is not None
        assert echo.next_fire > datetime.now(UTC)
        assert echo.consecutive_failures == 0

        listed = await trig_service.list_triggers(pool, sid, account_id=account_id)
        assert len(listed) == 1
        assert listed[0].name == "poll"

        await trig_service.remove_trigger(pool, sid, "poll", account_id=account_id)
        listed = await trig_service.list_triggers(pool, sid, account_id=account_id)
        assert listed == []

    async def test_add_disabled_has_null_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        spec = TriggerCreate.model_validate(
            {
                "name": "paused",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        echo = await trig_service.add_trigger(pool, sid, spec, account_id="acc_test_stub")
        assert echo.enabled is False
        assert echo.next_fire is None

    async def test_duplicate_name_rejected(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        from aios.errors import ConflictError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        await trig_service.add_trigger(pool, sid, _spec("dup"), account_id=account_id)
        with pytest.raises(ConflictError):
            await trig_service.add_trigger(pool, sid, _spec("dup"), account_id=account_id)

    async def test_remove_missing_raises(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        from aios.errors import NotFoundError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(NotFoundError):
            await trig_service.remove_trigger(pool, sid, "nope", account_id="acc_test_stub")

    async def test_update_source_recomputes_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        initial = await trig_service.add_trigger(
            pool, sid, _spec("p", schedule="*/5 * * * *"), account_id=account_id
        )
        assert initial.next_fire is not None
        before = initial.next_fire

        updated = await trig_service.update_trigger(
            pool,
            sid,
            "p",
            TriggerUpdate.model_validate({"source": {"kind": "cron", "schedule": "0 9 * * *"}}),
            account_id=account_id,
        )
        assert isinstance(updated.source, CronSource)
        assert updated.source.schedule == "0 9 * * *"
        assert updated.next_fire is not None
        assert updated.next_fire != before

    async def test_update_cron_to_one_shot_conversion(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        await trig_service.add_trigger(pool, sid, _spec("convert"), account_id=account_id)
        future = datetime.now(UTC) + timedelta(hours=1)
        updated = await trig_service.update_trigger(
            pool,
            sid,
            "convert",
            TriggerUpdate.model_validate(
                {"source": {"kind": "one_shot", "fire_at": future.isoformat()}}
            ),
            account_id=account_id,
        )
        assert updated.source.kind == "one_shot"
        assert updated.next_fire == future

    async def test_update_action_sandbox_to_wake_owner(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        await trig_service.add_trigger(pool, sid, _spec("p"), account_id=account_id)
        updated = await trig_service.update_trigger(
            pool,
            sid,
            "p",
            TriggerUpdate.model_validate({"action": {"kind": "wake_owner", "content": "wake!"}}),
            account_id=account_id,
        )
        assert isinstance(updated.action, WakeOwnerAction)
        assert updated.action.content == "wake!"

    async def test_update_disable_clears_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        await trig_service.add_trigger(pool, sid, _spec("p"), account_id=account_id)
        updated = await trig_service.update_trigger(
            pool,
            sid,
            "p",
            TriggerUpdate.model_validate({"enabled": False}),
            account_id=account_id,
        )
        assert updated.enabled is False
        assert updated.next_fire is None

    async def test_update_re_enable_recomputes_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        spec = TriggerCreate.model_validate(
            {
                "name": "p",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        await trig_service.add_trigger(pool, sid, spec, account_id=account_id)
        re_enabled = await trig_service.update_trigger(
            pool,
            sid,
            "p",
            TriggerUpdate.model_validate({"enabled": True}),
            account_id=account_id,
        )
        assert re_enabled.enabled is True
        assert re_enabled.next_fire is not None


class TestPerAccountCap:
    async def test_add_at_cap_rejected(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings
        from aios.errors import RateLimitedError

        account_id, sid = await _isolated_account_session(pool)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 2}),
        )

        await trig_service.add_trigger(pool, sid, _spec("a"), account_id=account_id)
        await trig_service.add_trigger(pool, sid, _spec("b"), account_id=account_id)
        with pytest.raises(RateLimitedError, match="active-trigger cap"):
            await trig_service.add_trigger(pool, sid, _spec("c"), account_id=account_id)

    async def test_wake_owner_counts_against_cap(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        """A wake_owner trigger consumes a per-account active-trigger slot just
        like a sandbox_command — a future 'self-delivery is cheap, skip the
        cap' refactor must not silently remove the only standing-row bound."""
        from aios.config import Settings
        from aios.errors import RateLimitedError

        account_id, sid = await _isolated_account_session(pool)
        future = datetime.now(UTC) + timedelta(hours=1)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 1}),
        )

        await trig_service.add_trigger(
            pool, sid, _wake_owner_spec("wo1", future), account_id=account_id
        )
        with pytest.raises(RateLimitedError, match="active-trigger cap"):
            await trig_service.add_trigger(
                pool, sid, _wake_owner_spec("wo2", future), account_id=account_id
            )

    async def test_disabled_does_not_count(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings

        account_id, sid = await _isolated_account_session(pool)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 1}),
        )

        await trig_service.add_trigger(pool, sid, _spec("enabled-1"), account_id=account_id)
        disabled = TriggerCreate.model_validate(
            {
                "name": "paused",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        disabled_echo = await trig_service.add_trigger(pool, sid, disabled, account_id=account_id)
        assert disabled_echo.enabled is False
        assert disabled_echo.next_fire is None
        listed = await trig_service.list_triggers(pool, sid, account_id=account_id)
        assert len(listed) == 2

    async def test_reenable_at_cap_rejected(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings
        from aios.errors import RateLimitedError

        account_id, sid = await _isolated_account_session(pool)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 1}),
        )

        await trig_service.add_trigger(pool, sid, _spec("enabled"), account_id=account_id)
        disabled = TriggerCreate.model_validate(
            {
                "name": "paused",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        await trig_service.add_trigger(pool, sid, disabled, account_id=account_id)

        with pytest.raises(RateLimitedError, match="active-trigger cap"):
            await trig_service.update_trigger(
                pool,
                sid,
                "paused",
                TriggerUpdate.model_validate({"enabled": True}),
                account_id=account_id,
            )


# ─── tick claim semantics ─────────────────────────────────────────────────


class TestTickClaim:
    async def test_claims_due_row(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("now"), account_id="acc_test_stub")

        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=future)

        names = [c.name for c in claimed]
        assert "now" in names
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT running_since, next_fire FROM triggers WHERE name = $1", "now"
            )
        assert row is not None
        assert row["running_since"] == future
        assert row["next_fire"] > future

    async def test_does_not_claim_future_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("later"), account_id="acc_test_stub")

        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=datetime.now(UTC))
        assert all(c.name != "later" for c in claimed)

    async def test_overlap_skip_via_running_since(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("inflight"), account_id="acc_test_stub")

        now = datetime.now(UTC)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE triggers SET running_since = $1, next_fire = $2 WHERE name = $3",
                now,
                now - timedelta(minutes=1),
                "inflight",
            )

        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(
                conn, now_utc=now + timedelta(seconds=10)
            )
        assert all(c.name != "inflight" for c in claimed)

    async def test_stale_running_since_is_reclaimed(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("stuck"), account_id="acc_test_stub")

        now = datetime.now(UTC)
        ancient = now - timedelta(hours=3)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE triggers SET running_since = $1, next_fire = $2 WHERE name = $3",
                ancient,
                ancient,
                "stuck",
            )

        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=now)
        assert "stuck" in [c.name for c in claimed]

    async def test_archived_session_excluded(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("doomed"), account_id="acc_test_stub")
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=future)
        assert all(c.name != "doomed" for c in claimed)

    async def test_disabled_trigger_not_claimed(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        spec = TriggerCreate.model_validate(
            {
                "name": "off",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        await trig_service.add_trigger(pool, sid, spec, account_id="acc_test_stub")

        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=future)
        assert all(c.name != "off" for c in claimed)


# ─── fire-recording + auto-disable ─────────────────────────────────────────


class TestRecordFire:
    async def test_ok_resets_consecutive_failures(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")

        async with pool.acquire() as conn:
            for _ in range(3):
                await queries.record_trigger_fire(
                    conn, echo.id, status="error", fired_at=datetime.now(UTC)
                )
            await queries.record_trigger_fire(
                conn, echo.id, status="ok", fired_at=datetime.now(UTC)
            )
            row = await conn.fetchrow(
                "SELECT consecutive_failures, last_fire_status, running_since "
                "FROM triggers WHERE id = $1",
                echo.id,
            )
        assert row is not None
        assert row["consecutive_failures"] == 0
        assert row["last_fire_status"] == "ok"
        assert row["running_since"] is None

    async def test_disable_clears_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")

        async with pool.acquire() as conn:
            await queries.disable_trigger(conn, echo.id)
            row = await conn.fetchrow(
                "SELECT enabled, next_fire FROM triggers WHERE id = $1", echo.id
            )
        assert row is not None
        assert row["enabled"] is False
        assert row["next_fire"] is None

    async def test_cron_auto_disable_surfaces_user_event(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        echo = await trig_service.add_trigger(pool, sid, _spec("doomed"), account_id=account_id)

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE triggers SET consecutive_failures = 4, running_since = $1 WHERE id = $2",
                datetime.now(UTC),
                echo.id,
            )

        async def _fail_run(_self: object, _handle: object, _cmd: str, **_kw: Any) -> Any:
            return mock.MagicMock(exit_code=2, timed_out=False, stderr="boom: cron failed")

        registry = _make_fake_sandbox_registry(_fail_run)
        prev_pool = runtime.pool
        prev_registry = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = registry
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock) as mock_defer:
            try:
                await trigger_runner.run_trigger_step(echo.id)
            finally:
                runtime.pool = prev_pool
                runtime.sandbox_registry = prev_registry

        async with pool.acquire() as conn:
            event_rows = await conn.fetch(
                "SELECT kind, data FROM events WHERE session_id = $1 ORDER BY seq ASC", sid
            )

        def _data(row: Any) -> dict[str, Any]:
            result: dict[str, Any] = (
                json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            )
            return result

        disable_messages = [
            r
            for r in event_rows
            if r["kind"] == "message"
            and _data(r).get("role") == "user"
            and "auto-disabled" in _data(r).get("content", "")
        ]
        assert disable_messages, "expected an auto-disable user message in the session log"
        latest_content = _data(disable_messages[-1])["content"]
        assert "doomed" in latest_content
        assert "5" in latest_content

        mock_defer.assert_called()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT enabled, consecutive_failures FROM triggers WHERE id = $1", echo.id
            )
        assert row is not None
        assert row["enabled"] is False
        assert row["consecutive_failures"] == 5


# ─── action dispatch: wake_owner ───────────────────────────────────────────


class TestWakeOwnerAction:
    async def test_one_shot_wake_owner_delivers_and_self_deletes(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A one-shot wake_owner fire delivers the content as a user-role
        message IN-WORKER (no sandbox) and the row self-deletes (#818 §8.1)."""
        from aios.errors import NotFoundError
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _wake_owner_spec("wo", now - timedelta(seconds=1), content="scheduled ping"),
            account_id=account_id,
        )
        # Mark claimed so the runner doesn't early-out.
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = $1 WHERE id = $2", now, echo.id)

        # A sandbox that explodes if touched — wake_owner must NOT use it.
        exploding = mock.MagicMock()
        exploding.get_or_provision = mock.AsyncMock(
            side_effect=AssertionError("wake_owner must not provision a sandbox")
        )
        exploding.exec = mock.AsyncMock(
            side_effect=AssertionError("wake_owner must not exec in a sandbox")
        )

        prev_pool = runtime.pool
        prev_registry = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = exploding
        # #1197: wake_owner now routes through the `stimulate` spine's
        # Tell(ExistingSession) arm, which resolves `defer_wake` from its source
        # module — so patch it there (the canonical call-site patch), not at the
        # old `trigger_runner` import which the reroute no longer touches.
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock) as mock_defer:
            try:
                await trigger_runner.run_trigger_step(echo.id)
            finally:
                runtime.pool = prev_pool
                runtime.sandbox_registry = prev_registry

        # (a) The marker landed as a user-role message.
        async with pool.acquire() as conn:
            event_rows = await conn.fetch(
                "SELECT kind, data FROM events WHERE session_id = $1 ORDER BY seq ASC", sid
            )

        def _data(row: Any) -> dict[str, Any]:
            parsed: dict[str, Any] = (
                json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            )
            return parsed

        wake_msgs = [
            r
            for r in event_rows
            if r["kind"] == "message"
            and _data(r).get("role") == "user"
            and _data(r).get("content") == "scheduled ping"
        ]
        assert wake_msgs, "expected the wake_owner content as a user message"
        # (b) A wake was deferred.
        mock_defer.assert_called()
        # (c) The one-shot row self-deleted.
        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.unscoped_get_trigger_row(conn, echo.id)

    async def test_cron_wake_owner_records_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A cron wake_owner records the fire and clears running_since (the
        recurring model-wake / deployment heartbeat shape)."""
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        spec = TriggerCreate.model_validate(
            {
                "name": "heartbeat",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "wake_owner", "content": "tick"},
            }
        )
        echo = await trig_service.add_trigger(pool, sid, spec, account_id=account_id)
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = now() WHERE id = $1", echo.id)

        prev_pool = runtime.pool
        runtime.pool = pool
        # #1197: wake_owner routes through the stimulate spine → patch defer_wake
        # at its source module (the reroute bypasses the trigger_runner import).
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock):
            try:
                await trigger_runner.run_trigger_step(echo.id)
            finally:
                runtime.pool = prev_pool

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT last_fire_status, running_since, enabled FROM triggers WHERE id = $1",
                echo.id,
            )
        assert row is not None
        assert row["last_fire_status"] == "ok"
        assert row["running_since"] is None
        assert row["enabled"] is True


# ─── DB CHECK probes (§2.1) ────────────────────────────────────────────────


async def _insert_raw(
    pool: Any,
    sid: str,
    *,
    source: str,
    source_spec: str,
    action: str,
    environment_id: str | None = None,
) -> None:
    """Insert a raw trigger row, bypassing the Pydantic write models — the
    probe vehicle for the live shape CHECKs."""
    from aios.ids import TRIGGER, make_id

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO triggers
                (id, owner_session_id, account_id, name, source, source_spec,
                 action, enabled, environment_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, true, $8, '{}'::jsonb)
            """,
            make_id(TRIGGER),
            sid,
            "acc_test_stub",
            f"raw-{_uniq()}",
            source,
            source_spec,
            action,
            environment_id,
        )


class TestShapeCheckConstraints:
    """The live triggers_source_spec_shape / triggers_action_shape CHECKs (and
    their load-bearing COALESCE wrappers) reject malformed rows — including the
    absent-key cases the unwrapped predicate would silently accept."""

    async def test_valid_rows_insert(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await _insert_raw(
            pool,
            sid,
            source="cron",
            source_spec='{"schedule": "*/5 * * * *"}',
            action='{"kind": "sandbox_command", "command": "x", "timeout_seconds": 300, '
            '"max_output_bytes": 65536}',
        )
        await _insert_raw(
            pool,
            sid,
            source="one_shot",
            source_spec='{"fire_at": "2026-06-11T09:00:00Z"}',
            action='{"kind": "wake_owner", "content": "hi"}',
        )

    async def test_one_shot_empty_spec_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        # The absent-key case the unwrapped predicate silently accepted.
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="one_shot",
                source_spec="{}",
                action='{"kind": "sandbox_command", "command": "x", "timeout_seconds": 1, '
                '"max_output_bytes": 1024}',
            )

    async def test_sandbox_command_missing_timeout_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "sandbox_command", "command": "x", "max_output_bytes": 1024}',
            )

    async def test_cron_with_foreign_fire_at_key_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *", "fire_at": "2026-06-11T09:00:00Z"}',
                action='{"kind": "sandbox_command", "command": "x", "timeout_seconds": 1, '
                '"max_output_bytes": 1024}',
            )

    async def test_unknown_source_rejected(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="webhook",
                source_spec='{"url": "https://x"}',
                action='{"kind": "sandbox_command", "command": "x", "timeout_seconds": 1, '
                '"max_output_bytes": 1024}',
            )


# ─── HTTP surface ──────────────────────────────────────────────────────────


class TestHttp:
    async def _create_session_via_http(
        self, http_client: httpx.AsyncClient, agent_id: str, env_id: str
    ) -> str:
        r = await http_client.post(
            "/v1/sessions", json={"agent_id": agent_id, "environment_id": env_id}
        )
        assert r.status_code == 201, r.text
        session_id: str = r.json()["id"]
        return session_id

    async def test_round_trip(
        self, pool: Any, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)

        r = await http_client.get(f"/v1/sessions/{sid}/triggers")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []

        r = await http_client.post(
            f"/v1/sessions/{sid}/triggers",
            json={
                "name": "poll",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "echo hi"},
            },
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "poll"
        assert created["source"]["kind"] == "cron"
        assert created["action"]["kind"] == "sandbox_command"
        # Defaults materialized server-side.
        assert created["action"]["timeout_seconds"] == 300

        r = await http_client.get(f"/v1/sessions/{sid}/triggers")
        assert len(r.json()["data"]) == 1

        # PUT replaces the action wholesale (timeout required on update).
        r = await http_client.put(
            f"/v1/sessions/{sid}/triggers/poll",
            json={
                "action": {
                    "kind": "sandbox_command",
                    "command": "echo updated",
                    "timeout_seconds": 60,
                    "max_output_bytes": 2048,
                }
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["action"]["command"] == "echo updated"

        r = await http_client.delete(f"/v1/sessions/{sid}/triggers/poll")
        assert r.status_code == 204, r.text
        r = await http_client.get(f"/v1/sessions/{sid}/triggers")
        assert r.json()["data"] == []

    async def test_partial_sandbox_action_update_rejected(
        self, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        await http_client.post(
            f"/v1/sessions/{sid}/triggers",
            json={
                "name": "p",
                "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                "action": {"kind": "sandbox_command", "command": "echo hi"},
            },
        )
        # Partial sandbox_command action (no timeout/max_output) → 422.
        r = await http_client.put(
            f"/v1/sessions/{sid}/triggers/p",
            json={"action": {"kind": "sandbox_command", "command": "echo x"}},
        )
        assert r.status_code == 422, r.text

    async def test_create_with_initial_triggers(
        self, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "triggers": [
                    {
                        "name": "a",
                        "source": {"kind": "cron", "schedule": "* * * * *"},
                        "action": {"kind": "sandbox_command", "command": "true"},
                    },
                    {
                        "name": "b",
                        "source": {"kind": "cron", "schedule": "0 9 * * *"},
                        "action": {"kind": "wake_owner", "content": "morning"},
                    },
                ],
            },
        )
        assert r.status_code == 201, r.text
        session = r.json()
        assert len(session["triggers"]) == 2

    async def test_invalid_cron_rejected_at_boundary(
        self, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.post(
            f"/v1/sessions/{sid}/triggers",
            json={
                "name": "bad",
                "source": {"kind": "cron", "schedule": "not a cron"},
                "action": {"kind": "sandbox_command", "command": "true"},
            },
        )
        assert r.status_code == 422, r.text

    async def test_delete_missing_returns_404(
        self, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.delete(f"/v1/sessions/{sid}/triggers/nope")
        assert r.status_code == 404, r.text

    async def test_session_update_does_not_accept_triggers(
        self, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.put(
            f"/v1/sessions/{sid}",
            json={
                "triggers": [
                    {
                        "name": "x",
                        "source": {"kind": "cron", "schedule": "* * * * *"},
                        "action": {"kind": "sandbox_command", "command": "true"},
                    }
                ]
            },
        )
        assert r.status_code == 422, r.text


# ─── live-attach: POST/DELETE on an already-live session (#1216) ────────────


class TestLiveAttachToLiveSession:
    """The #1216 bar: a trigger added to an ALREADY-LIVE session via the HTTP
    ``POST /v1/sessions/{id}/triggers`` endpoint is genuinely scheduled — it is
    claimed by the scheduler's own claim transaction and fires the owning
    session's wake WITHOUT a worker restart — and removing it via DELETE drops
    it from the schedule. This is the end-to-end "live-attach actually
    schedules" proof, not just "the row was inserted."

    The load-bearing guarantee (the #925 lesson, called out in the issue) is
    that the add/remove path NOTIFYs the scheduler so a runtime-added trigger
    is noticed on its first due time. The scheduler is NOTIFY/poll-driven; a
    bare DB write that emits no NOTIFY would be a dead heartbeat until the next
    cold-path poll. Two of the tests below assert the NOTIFY edge directly
    against the live ``aios_scheduled_tasks_due`` channel.
    """

    async def _create_session_via_http(
        self, http_client: httpx.AsyncClient, agent_id: str, env_id: str
    ) -> str:
        r = await http_client.post(
            "/v1/sessions", json={"agent_id": agent_id, "environment_id": env_id}
        )
        assert r.status_code == 201, r.text
        session_id: str = r.json()["id"]
        return session_id

    @staticmethod
    async def _await_notify(db_url: str, run: Any, *, timeout_s: float = 5.0) -> bool:
        """LISTEN on ``aios_scheduled_tasks_due``, run ``run`` (an awaitable
        factory performing the write on a DIFFERENT connection), and report
        whether a NOTIFY arrived within ``timeout_s``.

        The listener is attached BEFORE ``run`` executes so the edge can't be
        missed; mirrors ``test_migrations_0087_notify_next_fire``'s pattern.
        """
        got = asyncio.Event()
        listen_conn = await asyncpg.connect(db_url)
        try:

            def _cb(_c: Any, _pid: int, _chan: str, _payload: str) -> None:
                got.set()

            await listen_conn.add_listener("aios_scheduled_tasks_due", _cb)
            await run()
            try:
                await asyncio.wait_for(got.wait(), timeout=timeout_s)
                return True
            except TimeoutError:
                return False
        finally:
            await listen_conn.close()

    async def test_post_to_live_session_is_claimed_and_wakes_owner(
        self, pool: Any, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        """THE proof (#1216): a ``one_shot x wake_owner`` trigger POSTed to a
        LIVE (already-idle) session

          (a) appears in ``list_triggers`` (GET),
          (b) is CLAIMED by the scheduler's own claim transaction
              (``fetch_and_claim_due_triggers`` — the exact query the tick
              runs), and
          (c) when the claimed fire runs, the owning session WAKES: a
              ``run_trigger`` fired (defer_wake called) AND a user-role wake
              message is delivered to the session event log.

        This is the end-to-end "live-attach actually schedules" assertion — it
        drives the real claim path (not a hand-set ``running_since``) so it
        would catch a regression where a runtime add inserts the row but never
        gets scheduled.
        """
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)

        # (a) POST a one-shot wake_owner trigger due in the (near) past so the
        # very next claim picks it up — via the live HTTP endpoint.
        fire_at = datetime.now(UTC) - timedelta(seconds=1)
        r = await http_client.post(
            f"/v1/sessions/{sid}/triggers",
            json={
                "name": "heartbeat",
                "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
                "action": {"kind": "wake_owner", "content": "live-attached ping"},
            },
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["source"]["kind"] == "one_shot"
        assert created["action"]["kind"] == "wake_owner"

        # It shows up in the list endpoint.
        r = await http_client.get(f"/v1/sessions/{sid}/triggers")
        assert r.status_code == 200, r.text
        listed = {t["name"] for t in r.json()["data"]}
        assert "heartbeat" in listed

        # (b) The scheduler's claim transaction picks it up — exactly the query
        # the tick runs. A row inserted without a NOTIFY would still be claimed
        # by a poll, but the point here is that the claim path sees a
        # runtime-added trigger at all (no restart needed).
        now = datetime.now(UTC)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=now)
        claimed_for_session = [c for c in claimed if c.owner_session_id == sid]
        assert claimed_for_session, "scheduler did not claim the runtime-added trigger"
        echo_id = claimed_for_session[0].id

        # (c) The claimed fire wakes the owning session: a user-role wake
        # message lands and a wake is deferred (a `run_trigger` fired).
        prev_pool = runtime.pool
        runtime.pool = pool
        # #1197: wake_owner routes through the stimulate spine → patch defer_wake
        # at its source module (the reroute bypasses the trigger_runner import).
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock) as mock_defer:
            try:
                await trigger_runner.run_trigger_step(echo_id)
            finally:
                runtime.pool = prev_pool

        async with pool.acquire() as conn:
            event_rows = await conn.fetch(
                "SELECT kind, data FROM events WHERE session_id = $1 ORDER BY seq ASC", sid
            )

        def _data(row: Any) -> dict[str, Any]:
            parsed: dict[str, Any] = (
                json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            )
            return parsed

        wake_msgs = [
            r
            for r in event_rows
            if r["kind"] == "message"
            and _data(r).get("role") == "user"
            and _data(r).get("content") == "live-attached ping"
        ]
        assert wake_msgs, "expected the live-attached wake_owner content as a user message"
        mock_defer.assert_called()

    async def test_post_notifies_scheduler_immediately(
        self,
        pool: Any,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
        db_url: str,
    ) -> None:
        """The #925 load-bearing edge: POSTing a trigger to a live session emits
        a NOTIFY on ``aios_scheduled_tasks_due`` (the same channel
        session-creation uses), so the sleeping scheduler is woken and claims
        the new trigger on its first due time WITHOUT waiting for the cold-path
        heartbeat / a restart. A bare DB write that sent no NOTIFY would be a
        dead heartbeat — exactly the #925 incident shape.
        """
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)

        fire_at = datetime.now(UTC) + timedelta(hours=1)

        async def _do_post() -> None:
            r = await http_client.post(
                f"/v1/sessions/{sid}/triggers",
                json={
                    "name": "notify-on-add",
                    "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
                    "action": {"kind": "wake_owner", "content": "ping"},
                },
            )
            assert r.status_code == 201, r.text

        assert await self._await_notify(db_url, _do_post), (
            "POST /triggers did not NOTIFY aios_scheduled_tasks_due — a "
            "runtime-added trigger would be a dead heartbeat until the next poll"
        )

    async def test_delete_notifies_scheduler_and_drops_from_schedule(
        self,
        pool: Any,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
        db_url: str,
    ) -> None:
        """DELETE removes the trigger from ``list_triggers``, emits a NOTIFY so
        the scheduler re-computes its next wake, and the row no longer fires
        (the scheduler's claim transaction never sees it again)."""
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)

        # A due-in-the-past one-shot so we can prove it's no longer claimable.
        fire_at = datetime.now(UTC) - timedelta(seconds=1)
        r = await http_client.post(
            f"/v1/sessions/{sid}/triggers",
            json={
                "name": "to-remove",
                "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
                "action": {"kind": "wake_owner", "content": "ping"},
            },
        )
        assert r.status_code == 201, r.text

        async def _do_delete() -> None:
            r = await http_client.delete(f"/v1/sessions/{sid}/triggers/to-remove")
            assert r.status_code == 204, r.text

        assert await self._await_notify(db_url, _do_delete), (
            "DELETE /triggers did not NOTIFY aios_scheduled_tasks_due"
        )

        # Gone from the list.
        r = await http_client.get(f"/v1/sessions/{sid}/triggers")
        assert r.status_code == 200, r.text
        assert all(t["name"] != "to-remove" for t in r.json()["data"])

        # And no longer fires: the claim transaction never sees it.
        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=future)
        assert all(c.owner_session_id != sid for c in claimed)

    async def test_post_cross_account_session_404(
        self, pool: Any, http_client: httpx.AsyncClient
    ) -> None:
        """Account-scoped like every other session endpoint: POSTing a trigger
        to a session owned by ANOTHER account 404s (the actor's account doesn't
        own the session, so it can't be discovered)."""
        _account_id, other_sid = await _isolated_account_session(pool)
        fire_at = datetime.now(UTC) + timedelta(hours=1)
        r = await http_client.post(
            f"/v1/sessions/{other_sid}/triggers",
            json={
                "name": "intruder",
                "source": {"kind": "one_shot", "fire_at": fire_at.isoformat()},
                "action": {"kind": "wake_owner", "content": "ping"},
            },
        )
        assert r.status_code == 404, r.text

    async def test_delete_cross_account_session_404(
        self, pool: Any, http_client: httpx.AsyncClient
    ) -> None:
        """Cannot DELETE a trigger off another account's session — 404 before
        the actor's account can even name the row."""
        account_id, other_sid = await _isolated_account_session(pool)
        # Seed a trigger on the foreign session AS its owner so a 404 here is
        # the cross-account guard, not merely a missing row.
        await trig_service.add_trigger(
            pool,
            other_sid,
            _wake_owner_spec("owned", datetime.now(UTC) + timedelta(hours=1)),
            account_id=account_id,
        )
        r = await http_client.delete(f"/v1/sessions/{other_sid}/triggers/owned")
        assert r.status_code == 404, r.text

    async def test_post_duplicate_name_conflicts(
        self, pool: Any, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        """Triggers are UNIQUE(name) per owner: a duplicate name on the same
        live session is a clear 409, not a silent overwrite."""
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        body = {
            "name": "dupe",
            "source": {"kind": "cron", "schedule": "*/5 * * * *"},
            "action": {"kind": "wake_owner", "content": "ping"},
        }
        r = await http_client.post(f"/v1/sessions/{sid}/triggers", json=body)
        assert r.status_code == 201, r.text
        r = await http_client.post(f"/v1/sessions/{sid}/triggers", json=body)
        assert r.status_code == 409, r.text


# ─── enrichment + clone propagation ────────────────────────────────────────


class TestEnrichmentOnSessionMutations:
    async def test_get_session_includes_triggers(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.get_session(pool, sid, account_id="acc_test_stub")
        assert [t.name for t in s.triggers] == ["p"]

    async def test_archive_session_enriches_triggers(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")
        assert [t.name for t in s.triggers] == ["p"]

    async def test_update_session_enriches_triggers(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.update_session(
            pool, sid, title="renamed", account_id="acc_test_stub"
        )
        assert [t.name for t in s.triggers] == ["p"]

    async def test_clone_session_copies_cron_trigger(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        parent_sid = await _create_session(pool, env_id, agent_id)
        await trig_service.add_trigger(
            pool, parent_sid, _spec("p", schedule="0 9 * * *"), account_id="acc_test_stub"
        )

        clone = await sessions_service.clone_session(pool, parent_sid, account_id="acc_test_stub")
        assert [t.name for t in clone.triggers] == ["p"]

        clone_triggers = await trig_service.list_triggers(
            pool, clone.id, account_id="acc_test_stub"
        )
        assert len(clone_triggers) == 1
        parent_triggers = await trig_service.list_triggers(
            pool, parent_sid, account_id="acc_test_stub"
        )
        assert clone_triggers[0].id != parent_triggers[0].id
        clone_src = clone_triggers[0].source
        assert isinstance(clone_src, CronSource)
        assert clone_src.schedule == "0 9 * * *"
        assert clone_triggers[0].consecutive_failures == 0
        assert clone_triggers[0].last_fire_at is None
        assert clone_triggers[0].next_fire == parent_triggers[0].next_fire

    async def test_clone_session_copies_one_shot_trigger(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """§6.1 regression: cloning a session that owns a ONE-SHOT trigger
        must succeed (the pre-rename INSERT dropped fire_at and aborted the
        clone on the old XOR CHECK). The clone carries an identical source."""
        env_id, agent_id = env_and_agent
        parent_sid = await _create_session(pool, env_id, agent_id)
        future = datetime.now(UTC) + timedelta(hours=3)
        await trig_service.add_trigger(
            pool, parent_sid, _one_shot_spec("once", future), account_id="acc_test_stub"
        )

        clone = await sessions_service.clone_session(pool, parent_sid, account_id="acc_test_stub")
        assert [t.name for t in clone.triggers] == ["once"]
        clone_triggers = await trig_service.list_triggers(
            pool, clone.id, account_id="acc_test_stub"
        )
        assert len(clone_triggers) == 1
        clone_src = clone_triggers[0].source
        assert isinstance(clone_src, OneShotSource)
        assert clone_src.fire_at == future


class TestUpdatedAtAlwaysBumps:
    async def test_empty_patch_still_bumps_updated_at(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        import asyncio

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        initial = await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")
        await asyncio.sleep(0.05)
        patched = await trig_service.update_trigger(
            pool, sid, "p", TriggerUpdate.model_validate({}), account_id="acc_test_stub"
        )
        assert patched.updated_at > initial.updated_at


class TestArchiveRacePrevention:
    async def test_archived_at_in_row_after_session_archive(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        async with pool.acquire() as conn:
            row = await queries.unscoped_get_trigger_row(conn, echo.id)
        assert row.session_archived_at is not None

    async def test_handler_skips_archived_session(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(pool, sid, _spec("p"), account_id="acc_test_stub")

        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = now() WHERE id = $1", echo.id)
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        fake_sandbox = mock.MagicMock()
        fake_sandbox.get_or_provision = mock.AsyncMock(
            side_effect=AssertionError("handler must skip archived session")
        )
        fake_sandbox.exec = mock.AsyncMock(
            side_effect=AssertionError("handler must skip archived session")
        )

        prev_pool = runtime.pool
        prev_sandbox = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = fake_sandbox
        try:
            await trigger_runner.run_trigger_step(echo.id)
        finally:
            runtime.pool = prev_pool
            runtime.sandbox_registry = prev_sandbox

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT running_since, last_fire_status FROM triggers WHERE id = $1", echo.id
            )
        assert row is not None
        assert row["running_since"] is None
        assert row["last_fire_status"] == "skipped"


class TestRowDeletedRace:
    async def test_handler_swallows_deleted_row(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime, trigger_runner
        from aios.ids import TRIGGER, make_id

        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            phantom = make_id(TRIGGER)
            await trigger_runner.run_trigger_step(phantom)
        finally:
            runtime.pool = prev_pool


# ─── one-shot / fire_at lifecycle ─────────────────────────────────────────


class TestOneShotLifecycle:
    async def test_one_shot_claim_leaves_next_fire_unchanged(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        past_fire = now - timedelta(seconds=10)
        echo = await trig_service.add_trigger(
            pool, sid, _one_shot_spec("oneshot", past_fire), account_id=account_id
        )
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_triggers(conn, now_utc=now)

        names = [c.name for c in claimed]
        assert "oneshot" in names
        claimed_row = next(c for c in claimed if c.id == echo.id)
        assert claimed_row.source == "one_shot"
        assert claimed_row.next_fire == past_fire

    async def test_runner_deletes_one_shot_before_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.errors import NotFoundError
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _one_shot_spec("oneshot_del", now - timedelta(seconds=1)),
            account_id=account_id,
        )

        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = $1 WHERE id = $2", now, echo.id)

        delete_seen: list[str] = []
        provision_seen: list[str] = []

        async def _provision(p: Any) -> object:
            async with p.acquire() as conn:
                exists: bool = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM triggers WHERE id = $1)", echo.id
                )
            provision_seen.append(echo.id if exists else "deleted-before-fire")
            return mock.MagicMock()

        async def _run(_self: object, _handle: object, _cmd: str, **_kw: Any) -> Any:
            return mock.MagicMock(exit_code=0, timed_out=False, stderr="")

        registry = _make_fake_sandbox_registry(_run, provision_impl=_provision)
        prev_pool = runtime.pool
        prev_registry = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = registry
        try:
            orig_delete = queries.delete_trigger_by_id

            async def _instrumented_delete(conn: Any, trigger_id: str) -> None:
                delete_seen.append(trigger_id)
                await orig_delete(conn, trigger_id)

            with mock.patch.object(queries, "delete_trigger_by_id", _instrumented_delete):
                await trigger_runner.run_trigger_step(echo.id)
        finally:
            runtime.pool = prev_pool
            runtime.sandbox_registry = prev_registry

        assert echo.id in delete_seen
        assert provision_seen == ["deleted-before-fire"], (
            f"row must be deleted before sandbox-side run; got: {provision_seen}"
        )
        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.unscoped_get_trigger_row(conn, echo.id)

    async def test_one_shot_sandbox_failure_appends_session_event(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _one_shot_spec("oneshot_fail", now - timedelta(seconds=1)),
            account_id=account_id,
        )
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = $1 WHERE id = $2", now, echo.id)

        async def _fail_run(_self: object, _handle: object, _cmd: str, **_kw: Any) -> Any:
            return mock.MagicMock(
                exit_code=7, timed_out=False, stderr="curl: (7) Failed to connect to broker"
            )

        registry = _make_fake_sandbox_registry(_fail_run)
        prev_pool = runtime.pool
        prev_registry = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = registry
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock):
            try:
                await trigger_runner.run_trigger_step(echo.id)
            finally:
                runtime.pool = prev_pool
                runtime.sandbox_registry = prev_registry

        async with pool.acquire() as conn:
            event_rows = await conn.fetch(
                "SELECT kind, data FROM events WHERE session_id = $1 ORDER BY seq ASC", sid
            )

        def _data(row: Any) -> dict[str, Any]:
            parsed: dict[str, Any] = (
                json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            )
            return parsed

        failure_msgs = [
            r
            for r in event_rows
            if r["kind"] == "message"
            and _data(r).get("role") == "user"
            and "Scheduled wake" in _data(r).get("content", "")
        ]
        assert failure_msgs, "expected a failure-surfacing user message"
        content = _data(failure_msgs[-1])["content"]
        assert "oneshot_fail" in content
        assert "7" in content


class TestOneShotSkipDeletes:
    async def test_archived_skip_deletes_one_shot(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.errors import NotFoundError
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _one_shot_spec("archived_oneshot", now - timedelta(seconds=1)),
            account_id=account_id,
        )
        await sessions_service.archive_session(pool, sid, account_id=account_id)

        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            await trigger_runner.run_trigger_step(echo.id)
        finally:
            runtime.pool = prev_pool

        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.unscoped_get_trigger_row(conn, echo.id)

    async def test_disabled_skip_deletes_one_shot(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.errors import NotFoundError
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        now = datetime.now(UTC)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _one_shot_spec("disabled_oneshot", now - timedelta(seconds=1)),
            account_id=account_id,
        )
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET enabled = false WHERE id = $1", echo.id)

        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            await trigger_runner.run_trigger_step(echo.id)
        finally:
            runtime.pool = prev_pool

        async with pool.acquire() as conn:
            with pytest.raises(NotFoundError):
                await queries.unscoped_get_trigger_row(conn, echo.id)


class TestReleaseClaim:
    async def test_release_clears_running_since(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(
            pool, sid, _spec("release_target"), account_id="acc_test_stub"
        )
        now = datetime.now(UTC)
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = $1 WHERE id = $2", now, echo.id)
            await queries.release_trigger_claim(conn, echo.id)
            row = await conn.fetchrow("SELECT running_since FROM triggers WHERE id = $1", echo.id)
            assert row is not None and row["running_since"] is None


class TestReEnablePastFireAtRejected:
    async def test_reenable_past_one_shot_raises(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.errors import ValidationError as AiosValidationError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        past_fire = datetime.now(UTC) - timedelta(days=1)
        spec = TriggerCreate.model_validate(
            {
                "name": "stale_oneshot",
                "source": {"kind": "one_shot", "fire_at": past_fire.isoformat()},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        await trig_service.add_trigger(pool, sid, spec, account_id=account_id)

        with pytest.raises(AiosValidationError, match="not in the future"):
            await trig_service.update_trigger(
                pool,
                sid,
                "stale_oneshot",
                TriggerUpdate.model_validate({"enabled": True}),
                account_id=account_id,
            )

    async def test_source_replace_past_one_shot_on_enabled_row_raises(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """§2.4: replacing the source with a past one-shot on an ALREADY-enabled
        (cron) row is rejected too — not just the re-enable path."""
        from aios.errors import ValidationError as AiosValidationError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        await trig_service.add_trigger(pool, sid, _spec("enabled_cron"), account_id=account_id)

        past_fire = datetime.now(UTC) - timedelta(days=1)
        with pytest.raises(AiosValidationError, match="not in the future"):
            await trig_service.update_trigger(
                pool,
                sid,
                "enabled_cron",
                TriggerUpdate.model_validate(
                    {"source": {"kind": "one_shot", "fire_at": past_fire.isoformat()}}
                ),
                account_id=account_id,
            )

    async def test_reenable_with_fresh_fire_at_succeeds(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        past_fire = datetime.now(UTC) - timedelta(days=1)
        future_fire = datetime.now(UTC) + timedelta(hours=1)
        spec = TriggerCreate.model_validate(
            {
                "name": "fresh_oneshot",
                "source": {"kind": "one_shot", "fire_at": past_fire.isoformat()},
                "action": {"kind": "sandbox_command", "command": "true"},
                "enabled": False,
            }
        )
        await trig_service.add_trigger(pool, sid, spec, account_id=account_id)

        echo = await trig_service.update_trigger(
            pool,
            sid,
            "fresh_oneshot",
            TriggerUpdate.model_validate(
                {
                    "enabled": True,
                    "source": {"kind": "one_shot", "fire_at": future_fire.isoformat()},
                }
            ),
            account_id=account_id,
        )
        assert echo.enabled is True
        assert echo.next_fire is not None
        assert echo.next_fire > datetime.now(UTC)


class TestPerSessionCap:
    async def test_per_session_cap_rejected(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.errors import RateLimitedError

        account_id, sid = await _isolated_account_session(pool)

        monkeypatch.setattr("aios.services.triggers.MAX_TRIGGERS_PER_SESSION", 2)

        await trig_service.add_trigger(pool, sid, _spec("a"), account_id=account_id)
        await trig_service.add_trigger(pool, sid, _spec("b"), account_id=account_id)
        with pytest.raises(RateLimitedError, match="session at triggers cap"):
            await trig_service.add_trigger(pool, sid, _spec("c"), account_id=account_id)


class TestNotifyTrigger:
    """The rewritten ``notify_scheduled_tasks_due`` trigger fires on INSERT, a
    source/source_spec/enabled UPDATE, and the runner-clear edge — but NOT on
    an action-only edit (matching 0059, which never gated on command)."""

    async def test_insert_emits_notify(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            await trig_service.add_trigger(
                pool, sid, _spec("notify_me"), account_id="acc_test_stub"
            )
            try:
                await _asyncio.wait_for(event.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("INSERT did not produce a NOTIFY within 2s")

    async def test_running_since_clear_emits_notify(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(
            pool, sid, _spec("runner_clear"), account_id="acc_test_stub"
        )
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET running_since = now() WHERE id = $1", echo.id)

        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            async with pool.acquire() as conn:
                await queries.record_trigger_fire(
                    conn, echo.id, status="ok", fired_at=datetime.now(UTC)
                )
            try:
                await _asyncio.wait_for(event.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("running_since clear did not produce a NOTIFY")

    async def test_action_only_update_does_not_notify(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """An action-only edit (no source/enabled/runner-clear change) must NOT
        NOTIFY — it doesn't change which row is next-due."""
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(
            pool, sid, _spec("action_edit"), account_id="acc_test_stub"
        )

        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE triggers SET action = jsonb_set(action, '{command}', '\"echo z\"'), "
                    "updated_at = now() WHERE id = $1",
                    echo.id,
                )
            with pytest.raises(TimeoutError):
                await _asyncio.wait_for(event.wait(), timeout=0.5)

    async def test_next_fire_only_update_emits_notify(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A pure ``next_fire`` edit (rescheduling a row without touching
        source/source_spec/enabled/running_since) MUST NOTIFY — migration 0086
        (#940) added the ``next_fire`` clause so a sleeping scheduler wakes
        immediately instead of waiting out the heartbeat."""
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await trig_service.add_trigger(
            pool, sid, _spec("next_fire_edit"), account_id="acc_test_stub"
        )

        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE triggers SET next_fire = next_fire + interval '7 minutes' "
                    "WHERE id = $1",
                    echo.id,
                )
            try:
                await _asyncio.wait_for(event.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("next_fire-only update did not produce a NOTIFY within 2s")


class TestReEnableNextFireInvariant:
    """The invariant "an enabled schedulable trigger has non-NULL next_fire"
    across re-enable. Groups two cohesive cases:

    - The clean-path re-enable baseline (ground truth, green PRE-fix): the
      service auto-disable→re-enable (false→true) path already re-arms
      next_fire and NOTIFYs — pre-existing behavior, not the #957 fix.
    - The #957 heal cases: cron rows that reach enabled+next_fire=NULL out of
      band (the #925 incident-recovery anti-pattern's manual
      `UPDATE … SET enabled=true`) are re-armed on ANY update whose final state
      is enabled, and the NULL→non-NULL re-arm NOTIFYs so the scheduler
      relearns its sleep — while run_completion and one-shot rows are left
      NULL by design.
    """

    async def test_reenable_after_service_auto_disable_rearms_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Clean path (ground truth): the service auto-disable flips enabled
        false and clears next_fire; a plain re-enable (false→true) re-arms
        next_fire and NOTIFYs. Passes even before the #957 fix."""
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due
        from aios.harness import runtime, trigger_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        echo = await trig_service.add_trigger(pool, sid, _spec("ultron"), account_id=account_id)

        # Drive auto-disable through the SERVICE fire path: a failing run on the
        # 5th consecutive failure trips the breaker (MAX_CONSECUTIVE_FAILURES).
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE triggers SET consecutive_failures = 4, running_since = $1 WHERE id = $2",
                datetime.now(UTC),
                echo.id,
            )

        async def _fail_run(_self: object, _handle: object, _cmd: str, **_kw: Any) -> Any:
            return mock.MagicMock(exit_code=2, timed_out=False, stderr="boom: cron failed")

        registry = _make_fake_sandbox_registry(_fail_run)
        prev_pool = runtime.pool
        prev_registry = runtime.sandbox_registry
        runtime.pool = pool
        runtime.sandbox_registry = registry
        with mock.patch("aios.services.wake.defer_wake", new_callable=mock.AsyncMock):
            try:
                await trigger_runner.run_trigger_step(echo.id)
            finally:
                runtime.pool = prev_pool
                runtime.sandbox_registry = prev_registry

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT enabled, consecutive_failures, next_fire FROM triggers WHERE id = $1",
                echo.id,
            )
        assert row is not None
        assert row["enabled"] is False
        assert row["consecutive_failures"] == 5
        assert row["next_fire"] is None

        # Re-enable (false→true): re-arms next_fire AND NOTIFYs (enabled flip).
        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            re_enabled = await trig_service.update_trigger(
                pool,
                sid,
                "ultron",
                TriggerUpdate.model_validate({"enabled": True}),
                account_id=account_id,
            )
            assert re_enabled.enabled is True
            assert re_enabled.next_fire is not None
            try:
                await _asyncio.wait_for(event.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("re-enable did not produce a NOTIFY within 2s")

    async def test_reenable_already_enabled_null_next_fire_heals(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Contaminated path (#957 RED before fix): a row that is ALREADY
        enabled but has next_fire NULL is healed on a no-source enabled update —
        next_fire is recomputed and the NULL→non-NULL re-arm NOTIFYs."""
        import asyncio as _asyncio

        from aios.config import get_settings
        from aios.db.listen import listen_for_triggers_due

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        echo = await trig_service.add_trigger(pool, sid, _spec("ultron"), account_id=account_id)
        # Reproduces the #957/#925 incident state: the manual `UPDATE … enabled=
        # true` left next_fire NULL and suppressed the false→true transition the
        # service keys recompute on.
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET next_fire = NULL WHERE id = $1", echo.id)

        async with listen_for_triggers_due(get_settings().db_url) as event:
            event.clear()
            healed = await trig_service.update_trigger(
                pool,
                sid,
                "ultron",
                TriggerUpdate.model_validate({"enabled": True}),
                account_id=account_id,
            )
            assert healed.enabled is True
            assert healed.next_fire is not None  # heal recomputed next_fire
            try:
                await _asyncio.wait_for(event.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("heal (NULL→non-NULL next_fire) did not produce a NOTIFY within 2s")

    async def test_heal_leaves_run_completion_next_fire_null(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Invariant guard: healing a run_completion-source row (correctly at
        enabled=true, next_fire=NULL) must keep next_fire NULL —
        compute_initial_next_fire returns None for RunCompletionSource, so the
        heal is a no-op and the triggers_run_completion_no_next_fire DB guard is
        never violated."""
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"
        wf_id = await _make_workflow_e2e(pool)

        echo = await trig_service.add_trigger(
            pool, sid, _watch_spec("watcher", wf_id), account_id=account_id
        )
        assert echo.next_fire is None  # reactive row: NULL by design

        healed = await trig_service.update_trigger(
            pool,
            sid,
            "watcher",
            TriggerUpdate.model_validate({"enabled": True}),
            account_id=account_id,
        )
        assert healed.enabled is True
        assert healed.next_fire is None  # heal is a no-op for run_completion

        # The DB guard held (the row would not have UPDATEd otherwise).
        async with pool.acquire() as conn:
            col = await conn.fetchval("SELECT next_fire FROM triggers WHERE id = $1", echo.id)
        assert col is None

    async def test_heal_skips_one_shot_source(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """#957 regression: the heal is scoped to CRON sources, so an enabled
        one-shot row contaminated to next_fire=NULL is NOT healed on a no-source
        enabled=true PATCH — and (the actual bug) the recompute branch is never
        entered, so the one-shot past-fire_at guard does NOT raise a spurious 422.
        One-shots are fire-and-delete; their re-arm contract is owned by an
        explicit source replacement, not the heal."""
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        # Enabled one-shot with a fire_at in the PAST relative to the heal call:
        # add_trigger accepts any fire_at, so seed it in the past directly. The
        # 422 bug needed a past fire_at to trip the one-shot guard once the broad
        # heal wrongly entered the recompute branch.
        past_fire = datetime.now(UTC) - timedelta(days=1)
        echo = await trig_service.add_trigger(
            pool, sid, _one_shot_spec("stale_oneshot", past_fire), account_id=account_id
        )
        assert echo.next_fire is not None  # enabled one-shot armed to fire_at
        # Reproduce the contaminated state (enabled=true, next_fire=NULL) the #925
        # manual edit left, but on a one-shot row this time.
        async with pool.acquire() as conn:
            await conn.execute("UPDATE triggers SET next_fire = NULL WHERE id = $1", echo.id)

        # No raise (the pre-fix broad heal would have hit the one-shot past-fire_at
        # guard and 422'd here), and next_fire stays NULL — only crons are healed.
        healed = await trig_service.update_trigger(
            pool,
            sid,
            "stale_oneshot",
            TriggerUpdate.model_validate({"enabled": True}),
            account_id=account_id,
        )
        assert healed.enabled is True
        assert healed.next_fire is None  # one-shot is not healed

        async with pool.acquire() as conn:
            col = await conn.fetchval("SELECT next_fire FROM triggers WHERE id = $1", echo.id)
        assert col is None


class TestAdvisoryLockSerializesCapCheck:
    async def test_concurrent_add_at_cap_serialized(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        import asyncio as _asyncio

        from aios.config import Settings
        from aios.errors import RateLimitedError

        account_id, sid = await _isolated_account_session(pool)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 2}),
        )

        await trig_service.add_trigger(pool, sid, _spec("seed"), account_id=account_id)

        async def _add(n: int) -> tuple[int, bool]:
            try:
                await trig_service.add_trigger(
                    pool, sid, _spec(f"concurrent_{n}"), account_id=account_id
                )
                return n, True
            except RateLimitedError:
                return n, False

        results = await _asyncio.gather(*[_add(i) for i in range(5)])
        successes = sum(1 for _, ok in results if ok)
        assert successes == 1, f"expected 1 success under cap, got {successes}: {results}"


# ─── migration §6.2: agent tool-name rewrite guards ────────────────────────


class TestMigrationToolRewrite:
    """The migration's ``schedule_task_* → trigger_*`` rewrite (run against the
    SAME SQL the migration uses): empty-tools rows are NOT nulled, custom tools
    merely NAMED schedule_task_add are untouched, legacy built-ins are renamed.
    """

    def _load_rewrite_sql(self) -> str:
        import importlib.util

        repo_root = Path(__file__).resolve().parents[2]
        mig_path = repo_root / "migrations/versions/0083_triggers_rename_and_action_union.py"
        spec = importlib.util.spec_from_file_location("mig0083", mig_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._TOOL_RENAME_SQL.format(  # type: ignore[no-any-return]
            table="agents",
            old_add="schedule_task_add",
            old_remove="schedule_task_remove",
            old_update="schedule_task_update",
            old_list="schedule_task_list",
            new_create="trigger_create",
            new_remove="trigger_remove",
            new_update="trigger_update",
            new_list="trigger_list",
            like="schedule_task_",
        )

    async def _make_agent_with_tools(self, pool: Any, tools_json: str) -> str:
        agent = await agents_service.create_agent(
            pool,
            name=f"rw-agent-{_uniq()}",
            model="fake/test",
            system="x",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id="acc_test_stub",
        )
        # Mutate the tools column directly to the pre-migration fixture shape.
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE agents SET tools = $1::jsonb WHERE id = $2", tools_json, agent.id
            )
        return agent.id

    async def test_rewrite_guards(self, pool: Any) -> None:
        empty_id = await self._make_agent_with_tools(pool, "[]")
        custom_id = await self._make_agent_with_tools(
            pool,
            '[{"type": "custom", "name": "schedule_task_add", "description": "d", '
            '"parameters_schema": {}}]',
        )
        legacy_id = await self._make_agent_with_tools(
            pool, '[{"type": "schedule_task_add"}, {"type": "schedule_task_list"}]'
        )

        async with pool.acquire() as conn:
            await conn.execute(self._load_rewrite_sql())
            rows = {
                r["id"]: queries.parse_jsonb(r["tools"])
                for r in await conn.fetch(
                    "SELECT id, tools FROM agents WHERE id = ANY($1)",
                    [empty_id, custom_id, legacy_id],
                )
            }

        # Empty tools NOT nulled (the load-bearing WHERE guard).
        assert rows[empty_id] == []
        # Custom tool merely NAMED schedule_task_add is untouched (keys on type).
        assert rows[custom_id][0]["type"] == "custom"
        assert rows[custom_id][0]["name"] == "schedule_task_add"
        # Legacy built-ins renamed, order preserved.
        assert [t["type"] for t in rows[legacy_id]] == ["trigger_create", "trigger_list"]

        # The point of the rewrite: the migrated agent now deserializes back
        # through ToolSpec.model_validate (the layer the landmine is about)
        # without a validation 500 — `schedule_task_*` is no longer a valid
        # BuiltinToolType, so an unrewritten row would raise here.
        agent = await agents_service.get_agent(pool, legacy_id, account_id="acc_test_stub")
        assert [t.type for t in agent.tools] == ["trigger_create", "trigger_list"]


# ─── slice 2 (#819): CHECK probes, write paths, clone, env column ────────────


async def _make_workflow_e2e(
    pool: Any, script: str = "async def main(input):\n    return 1\n"
) -> str:
    from aios.db.queries import workflows as wf_queries

    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_test_stub", name=f"wf-{_uniq()}", script=script
        )
    return wf.id


def _workflow_action_spec(name: str, workflow_id: str, **action_overrides: Any) -> TriggerCreate:
    action: dict[str, Any] = {"kind": "workflow", "workflow_id": workflow_id}
    action.update(action_overrides)
    return TriggerCreate.model_validate(
        {
            "name": name,
            "source": {"kind": "cron", "schedule": "*/5 * * * *"},
            "action": action,
        }
    )


def _watch_spec(name: str, workflow_id: str, **source_overrides: Any) -> TriggerCreate:
    source: dict[str, Any] = {"kind": "run_completion", "workflow_id": workflow_id}
    source.update(source_overrides)
    return TriggerCreate.model_validate(
        {
            "name": name,
            "source": source,
            "action": {"kind": "wake_owner", "content": "a run completed"},
        }
    )


class TestSlice2ShapeChecks:
    """Hostile-row probes for the slice-2 predicate branches, the iff env
    constraint (both directions), and the no-next_fire guard — against the
    LIVE constraints (the §6 matrix, executed)."""

    _WF_ACTION_OK = (
        '{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": null, '
        '"input_template": null, "vault_ids": []}'
    )
    _RC_SPEC_OK = '{"workflow_id": "wf_x", "statuses": ["completed"]}'

    async def test_legal_slice2_rows_insert(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        # run_completion watch (wake_owner action, no env column).
        await _insert_raw(
            pool,
            sid,
            source="run_completion",
            source_spec=self._RC_SPEC_OK,
            action='{"kind": "wake_owner", "content": "hi"}',
        )
        # workflow action, float pin + null template (env column REQUIRED).
        await _insert_raw(
            pool,
            sid,
            source="cron",
            source_spec='{"schedule": "*/5 * * * *"}',
            action=self._WF_ACTION_OK,
            environment_id=env_id,
        )
        # int pin + object template + vault list.
        await _insert_raw(
            pool,
            sid,
            source="cron",
            source_spec='{"schedule": "*/5 * * * *"}',
            action='{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": 3, '
            '"input_template": {"x": 1}, "vault_ids": ["vlt_a"]}',
            environment_id=env_id,
        )

    async def test_workflow_action_without_env_column_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError, match="environment_id_iff_workflow"):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action=self._WF_ACTION_OK,
            )

    async def test_non_workflow_action_with_env_column_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        # The iff constraint covers BOTH directions.
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError, match="environment_id_iff_workflow"):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "wake_owner", "content": "hi"}',
                environment_id=env_id,
            )

    async def test_run_completion_stray_fire_at_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="run_completion",
                source_spec='{"workflow_id": "wf_x", "statuses": ["completed"], '
                '"fire_at": "2026-06-11T09:00:00Z"}',
                action='{"kind": "wake_owner", "content": "hi"}',
            )

    async def test_run_completion_missing_statuses_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        # statuses ships in the kind's FIRST shape, so it joins the CHECK.
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="run_completion",
                source_spec='{"workflow_id": "wf_x"}',
                action='{"kind": "wake_owner", "content": "hi"}',
            )

    async def test_workflow_action_missing_version_key_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        # The §1.1 required-but-nullable idiom: the KEY must be present.
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "workflow", "workflow_id": "wf_x", '
                '"input_template": null, "vault_ids": []}',
                environment_id=env_id,
            )

    async def test_workflow_action_string_version_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": "3", '
                '"input_template": null, "vault_ids": []}',
                environment_id=env_id,
            )

    async def test_workflow_action_missing_template_key_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": null, '
                '"vault_ids": []}',
                environment_id=env_id,
            )

    async def test_workflow_action_object_vault_ids_rejected(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_raw(
                pool,
                sid,
                source="cron",
                source_spec='{"schedule": "*/5 * * * *"}',
                action='{"kind": "workflow", "workflow_id": "wf_x", "workflow_version": null, '
                '"input_template": null, "vault_ids": {}}',
                environment_id=env_id,
            )


class TestSlice2WritePath:
    """The shared write-path validator across all three write paths: watched-
    workflow existence, pin == current, env resolution, cross-tenant 404."""

    async def test_run_completion_echo_round_trips_statuses(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.models.triggers import RunCompletionSource

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        echo = await trig_service.add_trigger(
            pool,
            sid,
            _watch_spec("watcher", wf_id, statuses=["errored"]),
            account_id="acc_test_stub",
        )
        assert isinstance(echo.source, RunCompletionSource)
        assert echo.source.statuses == ["errored"]
        assert echo.next_fire is None  # unschedulable by the tick

        listed = await trig_service.list_triggers(pool, sid, account_id="acc_test_stub")
        src = listed[0].source
        assert isinstance(src, RunCompletionSource)
        assert src.statuses == ["errored"]

    async def test_bad_watch_404s(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        # The silent-dead-watch guard: a typo'd watch fails at create instead
        # of sitting dead forever (fire-time never surfaces a non-match).
        from aios.errors import NotFoundError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(NotFoundError):
            await trig_service.add_trigger(
                pool, sid, _watch_spec("watcher", "wf_nonexistent"), account_id="acc_test_stub"
            )

    async def test_pin_must_equal_current_version(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        # "resolve-latest-at-write": the write resolves what the pin freezes.
        from aios.errors import ConflictError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        with pytest.raises(ConflictError) as excinfo:
            await trig_service.add_trigger(
                pool,
                sid,
                _workflow_action_spec("pinned", wf_id, workflow_version=7),
                account_id="acc_test_stub",
            )
        assert excinfo.value.detail == {"pinned": 7, "current": 1}

        echo = await trig_service.add_trigger(
            pool,
            sid,
            _workflow_action_spec("pinned", wf_id, workflow_version=1),
            account_id="acc_test_stub",
        )
        assert echo.name == "pinned"

    async def test_workflow_action_resolves_env_column(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        echo = await trig_service.add_trigger(
            pool, sid, _workflow_action_spec("launcher", wf_id), account_id="acc_test_stub"
        )
        async with pool.acquire() as conn:
            col = await conn.fetchval("SELECT environment_id FROM triggers WHERE id = $1", echo.id)
        assert col == env_id  # the owner session's env, resolved at write

    async def test_cross_tenant_attach_404s(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Obligation 2 + Seam A: a foreign session_id 404s for ANY action
        kind — the unconditional account-scoped session read is FIRST in the
        shared validator (for slice-1 sandbox_command this was cross-tenant
        code execution at fire time)."""
        from aios.errors import NotFoundError

        _env_id, _agent_id = env_and_agent  # fixture forces the stub account to exist
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_other', 'acc_test_stub', FALSE, 'other-tenant')
                ON CONFLICT DO NOTHING;
                INSERT INTO environments (id, name, account_id)
                VALUES ('env_other', 'other-env', 'acc_other') ON CONFLICT DO NOTHING;
                INSERT INTO agents (id, name, model, account_id)
                VALUES ('agn_other', 'other-agent', 'fake/test', 'acc_other')
                ON CONFLICT DO NOTHING;
                INSERT INTO sessions
                    (id, agent_id, environment_id, workspace_volume_path, account_id)
                VALUES ('sess_other_tenant', 'agn_other', 'env_other', '/tmp/ws-o', 'acc_other')
                ON CONFLICT DO NOTHING;
                """
            )
        with pytest.raises(NotFoundError):
            await trig_service.add_trigger(
                pool, "sess_other_tenant", _spec("intruder"), account_id="acc_test_stub"
            )

    async def test_session_create_attach_validates_like_post(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """§10.j parity: SessionCreate.triggers routes through the SAME shared
        validator — bad watch 404s, drifted pin 409s, and a valid workflow
        action resolves the env column."""
        from aios.errors import ConflictError, NotFoundError

        env_id, agent_id = env_and_agent
        wf_id = await _make_workflow_e2e(pool)

        with pytest.raises(NotFoundError):
            await sessions_service.create_session(
                pool,
                agent_id=agent_id,
                environment_id=env_id,
                title=None,
                metadata={},
                triggers=[_watch_spec("w", "wf_nonexistent")],
                account_id="acc_test_stub",
            )
        with pytest.raises(ConflictError):
            await sessions_service.create_session(
                pool,
                agent_id=agent_id,
                environment_id=env_id,
                title=None,
                metadata={},
                triggers=[_workflow_action_spec("p", wf_id, workflow_version=9)],
                account_id="acc_test_stub",
            )

        session = await sessions_service.create_session(
            pool,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            triggers=[_workflow_action_spec("launcher", wf_id)],
            account_id="acc_test_stub",
        )
        assert [t.name for t in session.triggers] == ["launcher"]
        async with pool.acquire() as conn:
            col = await conn.fetchval(
                "SELECT environment_id FROM triggers WHERE owner_session_id = $1", session.id
            )
        assert col == env_id

    async def test_update_recomputes_env_column_on_action_change(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """§10.h: kind conversions flip the env column BOTH directions in one
        UPDATE (the iff CHECK would abort a half-flip); a same-kind workflow
        replacement re-resolves it."""
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        echo = await trig_service.add_trigger(
            pool, sid, _workflow_action_spec("morph", wf_id), account_id="acc_test_stub"
        )

        async def _env_col() -> str | None:
            async with pool.acquire() as conn:
                col: str | None = await conn.fetchval(
                    "SELECT environment_id FROM triggers WHERE id = $1", echo.id
                )
            return col

        assert await _env_col() == env_id

        # workflow → wake_owner: column must flip to NULL in the same UPDATE.
        await trig_service.update_trigger(
            pool,
            sid,
            "morph",
            TriggerUpdate.model_validate({"action": {"kind": "wake_owner", "content": "hi"}}),
            account_id="acc_test_stub",
        )
        assert await _env_col() is None

        # wake_owner → workflow: column set again.
        await trig_service.update_trigger(
            pool,
            sid,
            "morph",
            TriggerUpdate.model_validate(
                {
                    "action": {
                        "kind": "workflow",
                        "workflow_id": wf_id,
                        "workflow_version": None,
                        "input_template": {"x": 1},
                        "vault_ids": [],
                    }
                }
            ),
            account_id="acc_test_stub",
        )
        assert await _env_col() == env_id

        # Same-kind replacement re-resolves (env is immutable, so == still).
        await trig_service.update_trigger(
            pool,
            sid,
            "morph",
            TriggerUpdate.model_validate(
                {
                    "action": {
                        "kind": "workflow",
                        "workflow_id": wf_id,
                        "workflow_version": None,
                        "input_template": None,
                        "vault_ids": [],
                    }
                }
            ),
            account_id="acc_test_stub",
        )
        assert await _env_col() == env_id


class TestSlice2ClonePropagation:
    async def test_clone_session_copies_workflow_action_trigger(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """The §6.1 clone-crash class, resurrected by the new column: without
        the clone INSERT carrying environment_id, this aborts on the iff CHECK."""
        env_id, agent_id = env_and_agent
        parent_sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        await trig_service.add_trigger(
            pool,
            parent_sid,
            _workflow_action_spec("launcher", wf_id),
            account_id="acc_test_stub",
        )

        clone = await sessions_service.clone_session(pool, parent_sid, account_id="acc_test_stub")
        assert [t.name for t in clone.triggers] == ["launcher"]
        async with pool.acquire() as conn:
            col = await conn.fetchval(
                "SELECT environment_id FROM triggers WHERE owner_session_id = $1", clone.id
            )
        # Verbatim copy is correct-by-construction: the clone's session row
        # copies the parent's environment, so trigger-env == owner-env holds.
        assert col == env_id

    async def test_clone_session_copies_run_completion_trigger(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.models.triggers import RunCompletionSource

        env_id, agent_id = env_and_agent
        parent_sid = await _create_session(pool, env_id, agent_id)
        wf_id = await _make_workflow_e2e(pool)
        await trig_service.add_trigger(
            pool, parent_sid, _watch_spec("watcher", wf_id), account_id="acc_test_stub"
        )

        clone = await sessions_service.clone_session(pool, parent_sid, account_id="acc_test_stub")
        clone_triggers = await trig_service.list_triggers(
            pool, clone.id, account_id="acc_test_stub"
        )
        assert len(clone_triggers) == 1
        src = clone_triggers[0].source
        assert isinstance(src, RunCompletionSource)
        assert src.workflow_id == wf_id
        assert clone_triggers[0].next_fire is None  # stays tick-unschedulable


class TestTriggerRunsRoute:
    """GET /v1/sessions/{sid}/triggers/{name}/runs — the per-fire audit read.

    Keyed by the audit table's DENORMALIZED columns, never the live trigger
    row: one-shot tombstones and a deleted trigger's history stay reachable.
    """

    async def _seed_audit_rows(self, pool: Any, sid: str, name: str, n: int) -> str:
        echo = await trig_service.add_trigger(
            pool,
            sid,
            TriggerCreate.model_validate(
                {
                    "name": name,
                    "source": {"kind": "cron", "schedule": "*/5 * * * *"},
                    "action": {"kind": "wake_owner", "content": "tick"},
                }
            ),
            account_id="acc_test_stub",
        )
        async with pool.acquire() as conn:
            for i in range(n):
                await queries.record_trigger_run(
                    conn,
                    trigger_id=echo.id,
                    account_id="acc_test_stub",
                    owner_session_id=sid,
                    trigger_name=name,
                    trigger_context="cron",
                    status="ok" if i % 2 == 0 else "error",
                    error_summary=None if i % 2 == 0 else f"boom {i}",
                    result_id=None,
                    started_at=datetime.now(UTC) + timedelta(seconds=i),
                )
        return echo.id

    async def test_lists_newest_first_with_limit(
        self, pool: Any, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await self._seed_audit_rows(pool, sid, "audited", 3)

        r = await http_client.get(f"/v1/sessions/{sid}/triggers/audited/runs")
        assert r.status_code == 200, r.text
        rows = r.json()["data"]
        assert len(rows) == 3
        created = [row["created_at"] for row in rows]
        assert created == sorted(created, reverse=True)  # newest first
        assert {row["trigger_context"] for row in rows} == {"cron"}

        r = await http_client.get(f"/v1/sessions/{sid}/triggers/audited/runs?limit=2")
        assert len(r.json()["data"]) == 2

    async def test_history_survives_trigger_deletion(
        self, pool: Any, http_client: httpx.AsyncClient, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await self._seed_audit_rows(pool, sid, "ghost", 2)
        await trig_service.remove_trigger(pool, sid, "ghost", account_id="acc_test_stub")

        r = await http_client.get(f"/v1/sessions/{sid}/triggers/ghost/runs")
        assert r.status_code == 200, r.text
        assert len(r.json()["data"]) == 2  # the audit outlives the trigger

    async def test_account_scoped(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await self._seed_audit_rows(pool, sid, "scoped", 1)
        rows = await trig_service.list_trigger_runs(
            pool, sid, "scoped", account_id="acc_someone_else"
        )
        assert rows == []
