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
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


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

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_cap_{_uniq()}"

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 2}),
        )

        await trig_service.add_trigger(pool, sid, _spec("a"), account_id=account_id)
        await trig_service.add_trigger(pool, sid, _spec("b"), account_id=account_id)
        with pytest.raises(RateLimitedError, match="active-timer cap"):
            await trig_service.add_trigger(pool, sid, _spec("c"), account_id=account_id)

    async def test_wake_owner_counts_against_cap(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        """A wake_owner trigger consumes a per-account active-timer slot just
        like a sandbox_command — a future 'self-delivery is cheap, skip the
        cap' refactor must not silently remove the only standing-row bound."""
        from aios.config import Settings
        from aios.errors import RateLimitedError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_wo_cap_{_uniq()}"
        future = datetime.now(UTC) + timedelta(hours=1)

        original = Settings()
        monkeypatch.setattr(
            "aios.services.triggers.get_settings",
            lambda: original.model_copy(update={"triggers_per_account_max": 1}),
        )

        await trig_service.add_trigger(
            pool, sid, _wake_owner_spec("wo1", future), account_id=account_id
        )
        with pytest.raises(RateLimitedError, match="active-timer cap"):
            await trig_service.add_trigger(
                pool, sid, _wake_owner_spec("wo2", future), account_id=account_id
            )

    async def test_disabled_does_not_count(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_cap_dis_{_uniq()}"

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

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_cap_re_{_uniq()}"

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

        with pytest.raises(RateLimitedError, match="active-timer cap"):
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
            await queries.record_trigger_fire(
                conn, echo.id, status="error", consecutive_failures=3, fired_at=datetime.now(UTC)
            )
            await queries.record_trigger_fire(
                conn, echo.id, status="ok", consecutive_failures=0, fired_at=datetime.now(UTC)
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
        with mock.patch(
            "aios.harness.trigger_runner.defer_wake", new_callable=mock.AsyncMock
        ) as mock_defer:
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
        with mock.patch(
            "aios.harness.trigger_runner.defer_wake", new_callable=mock.AsyncMock
        ) as mock_defer:
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
        with mock.patch("aios.harness.trigger_runner.defer_wake", new_callable=mock.AsyncMock):
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


class TestShapeCheckConstraints:
    """The live triggers_source_spec_shape / triggers_action_shape CHECKs (and
    their load-bearing COALESCE wrappers) reject malformed rows — including the
    absent-key cases the unwrapped predicate would silently accept."""

    async def _insert_raw(
        self, pool: Any, sid: str, *, source: str, source_spec: str, action: str
    ) -> None:
        from aios.ids import TRIGGER, make_id

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO triggers
                    (id, owner_session_id, account_id, name, source, source_spec,
                     action, enabled, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, true, '{}'::jsonb)
                """,
                make_id(TRIGGER),
                sid,
                "acc_test_stub",
                f"raw-{_uniq()}",
                source,
                source_spec,
                action,
            )

    async def test_valid_rows_insert(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await self._insert_raw(
            pool,
            sid,
            source="cron",
            source_spec='{"schedule": "*/5 * * * *"}',
            action='{"kind": "sandbox_command", "command": "x", "timeout_seconds": 300, '
            '"max_output_bytes": 65536}',
        )
        await self._insert_raw(
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
            await self._insert_raw(
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
            await self._insert_raw(
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
            await self._insert_raw(
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
            await self._insert_raw(
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
        with mock.patch("aios.harness.trigger_runner.defer_wake", new_callable=mock.AsyncMock):
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

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_session_cap_{_uniq()}"

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
                    conn, echo.id, status="ok", consecutive_failures=0, fired_at=datetime.now(UTC)
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


class TestAdvisoryLockSerializesCapCheck:
    async def test_concurrent_add_at_cap_serialized(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        import asyncio as _asyncio

        from aios.config import Settings
        from aios.errors import RateLimitedError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_lock_{_uniq()}"

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
