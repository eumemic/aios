"""E2E tests for ``scheduled_tasks`` (#636).

Real Postgres (testcontainer) + real FastAPI app via ASGI transport.
Covers:

- Service layer: granular add/remove/update/list round-trip; ``next_fire``
  recomputation on schedule change / enable-flip; duplicate-name rejection.
- Tick claim: overlap-prevention via ``running_since``; stale-recovery;
  archived-session filter; ``next_fire`` advanced inside the claim
  transaction (the #636 correctness property the design hinges on).
- Auto-disable: ``MAX_CONSECUTIVE_FAILURES`` threshold flips ``enabled``
  to false and clears ``next_fire``.
- HTTP surface: POST / GET / DELETE / PUT round-trip.

No worker process and no Docker container are started here — the
fire-job handler exercises the sandbox, which is covered separately by
sandbox-tier tests. This file's job is the substrate's correctness at
the DB + service + HTTP layers.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock

import httpx
import pytest

from aios.db import queries
from aios.models.agents import ToolSpec
from aios.models.scheduled_tasks import ScheduledTaskCreate, ScheduledTaskUpdate
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import scheduled_tasks as st_service
from aios.services import sessions as sessions_service
from tests.helpers.connections import authed_client


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
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
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
        pool, name=f"sched-env-{_uniq()}", account_id=account_id
    )
    agent = await agents_service.create_agent(
        pool,
        name=f"sched-agent-{_uniq()}",
        model="fake/test",
        system="scheduled_tasks test",
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


def _spec(
    name: str, schedule: str = "*/5 * * * *", command: str = "echo hi"
) -> ScheduledTaskCreate:
    return ScheduledTaskCreate.model_validate(
        {"name": name, "schedule": schedule, "command": command}
    )


# ─── service-layer round-trip ─────────────────────────────────────────────


class TestServiceLayer:
    async def test_add_get_list_remove(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        echo = await st_service.add_task(pool, sid, _spec("poll"), account_id=account_id)
        assert echo.name == "poll"
        assert echo.id.startswith("sched_")
        assert echo.enabled is True
        assert echo.next_fire is not None
        assert echo.next_fire > datetime.now(UTC)
        assert echo.consecutive_failures == 0

        listed = await st_service.list_tasks(pool, sid, account_id=account_id)
        assert len(listed) == 1
        assert listed[0].name == "poll"

        await st_service.remove_task(pool, sid, "poll", account_id=account_id)
        listed = await st_service.list_tasks(pool, sid, account_id=account_id)
        assert listed == []

    async def test_add_disabled_has_null_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        spec = ScheduledTaskCreate.model_validate(
            {"name": "paused", "schedule": "*/5 * * * *", "command": "true", "enabled": False}
        )
        echo = await st_service.add_task(pool, sid, spec, account_id="acc_test_stub")
        assert echo.enabled is False
        assert echo.next_fire is None

    async def test_duplicate_name_rejected(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        from aios.errors import ConflictError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        await st_service.add_task(pool, sid, _spec("dup"), account_id=account_id)
        with pytest.raises(ConflictError):
            await st_service.add_task(pool, sid, _spec("dup"), account_id=account_id)

    async def test_remove_missing_raises(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        from aios.errors import NotFoundError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        with pytest.raises(NotFoundError):
            await st_service.remove_task(pool, sid, "nope", account_id="acc_test_stub")

    async def test_update_schedule_recomputes_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        initial = await st_service.add_task(
            pool, sid, _spec("p", schedule="*/5 * * * *"), account_id=account_id
        )
        assert initial.next_fire is not None
        before = initial.next_fire

        updated = await st_service.update_task(
            pool,
            sid,
            "p",
            ScheduledTaskUpdate.model_validate({"schedule": "0 9 * * *"}),
            account_id=account_id,
        )
        assert updated.schedule == "0 9 * * *"
        assert updated.next_fire is not None
        # New schedule yields a later daily fire vs the 5-minute cadence.
        assert updated.next_fire != before

    async def test_update_disable_clears_next_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = "acc_test_stub"

        await st_service.add_task(pool, sid, _spec("p"), account_id=account_id)
        updated = await st_service.update_task(
            pool,
            sid,
            "p",
            ScheduledTaskUpdate.model_validate({"enabled": False}),
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

        spec = ScheduledTaskCreate.model_validate(
            {"name": "p", "schedule": "*/5 * * * *", "command": "true", "enabled": False}
        )
        await st_service.add_task(pool, sid, spec, account_id=account_id)
        re_enabled = await st_service.update_task(
            pool,
            sid,
            "p",
            ScheduledTaskUpdate.model_validate({"enabled": True}),
            account_id=account_id,
        )
        assert re_enabled.enabled is True
        assert re_enabled.next_fire is not None


class TestPerAccountCap:
    """The ``scheduled_tasks_per_account_max`` cap shipped in Phase 3 of
    the schedule_wake unification."""

    async def test_add_at_cap_rejected(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings
        from aios.errors import RateLimitedError

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_cap_{_uniq()}"

        # Drop the cap to 2 for the test so we don't have to insert 100 rows.
        original = Settings()
        monkeypatch.setattr(
            "aios.services.scheduled_tasks.get_settings",
            lambda: original.model_copy(update={"scheduled_tasks_per_account_max": 2}),
        )

        await st_service.add_task(pool, sid, _spec("a"), account_id=account_id)
        await st_service.add_task(pool, sid, _spec("b"), account_id=account_id)
        with pytest.raises(RateLimitedError, match="active-timer cap"):
            await st_service.add_task(pool, sid, _spec("c"), account_id=account_id)

    async def test_disabled_does_not_count(
        self, pool: Any, env_and_agent: tuple[str, str], monkeypatch: Any
    ) -> None:
        from aios.config import Settings

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        account_id = f"acc_cap_dis_{_uniq()}"

        original = Settings()
        monkeypatch.setattr(
            "aios.services.scheduled_tasks.get_settings",
            lambda: original.model_copy(update={"scheduled_tasks_per_account_max": 1}),
        )

        # Fill the cap with one enabled row, then add many disabled — fine.
        await st_service.add_task(pool, sid, _spec("enabled-1"), account_id=account_id)
        disabled = ScheduledTaskCreate.model_validate(
            {"name": "paused", "schedule": "*/5 * * * *", "command": "true", "enabled": False}
        )
        await st_service.add_task(pool, sid, disabled, account_id=account_id)

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
            "aios.services.scheduled_tasks.get_settings",
            lambda: original.model_copy(update={"scheduled_tasks_per_account_max": 1}),
        )

        # One enabled (at cap), one disabled — re-enabling the second
        # should now hit the cap.
        await st_service.add_task(pool, sid, _spec("enabled"), account_id=account_id)
        disabled = ScheduledTaskCreate.model_validate(
            {"name": "paused", "schedule": "*/5 * * * *", "command": "true", "enabled": False}
        )
        await st_service.add_task(pool, sid, disabled, account_id=account_id)

        with pytest.raises(RateLimitedError, match="active-timer cap"):
            await st_service.update_task(
                pool,
                sid,
                "paused",
                ScheduledTaskUpdate.model_validate({"enabled": True}),
                account_id=account_id,
            )


# ─── tick claim semantics ─────────────────────────────────────────────────


class TestTickClaim:
    async def test_claims_due_row(self, pool: Any, env_and_agent: tuple[str, str]) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("now"), account_id="acc_test_stub")

        # Tick in the future so the row's next_fire is in the past.
        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(conn, now_utc=future)

        names = [c.name for c in claimed]
        assert "now" in names
        # The claim must have marked running_since and advanced next_fire
        # past the tick's now_utc — verify by re-reading the row.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT running_since, next_fire FROM session_scheduled_tasks WHERE name = $1",
                "now",
            )
        assert row is not None
        assert row["running_since"] == future
        assert row["next_fire"] > future

    async def test_does_not_claim_future_fire(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("later"), account_id="acc_test_stub")

        # Tick at "now": the just-added task's next_fire is in the future.
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(
                conn, now_utc=datetime.now(UTC)
            )
        assert all(c.name != "later" for c in claimed)

    async def test_overlap_skip_via_running_since(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A row with ``running_since`` set is excluded — overlap-prevention.

        This is the #636 correctness property: a fire that's still in
        flight must not be re-fired, even if its ``next_fire`` is in the
        past (which it usually is right after the tick advances it
        relative to ``now``).
        """
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("inflight"), account_id="acc_test_stub")

        # Mark in-flight (running_since=now, next_fire in the past).
        now = datetime.now(UTC)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE session_scheduled_tasks "
                "SET running_since = $1, next_fire = $2 WHERE name = $3",
                now,
                now - timedelta(minutes=1),
                "inflight",
            )

        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(
                conn, now_utc=now + timedelta(seconds=10)
            )
        assert all(c.name != "inflight" for c in claimed)

    async def test_stale_running_since_is_reclaimed(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A row whose ``running_since`` is older than the stale threshold
        gets picked up again — recovers from worker crashes mid-fire."""
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("stuck"), account_id="acc_test_stub")

        # running_since 3 hours ago (past the default 2h stale threshold).
        now = datetime.now(UTC)
        ancient = now - timedelta(hours=3)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE session_scheduled_tasks "
                "SET running_since = $1, next_fire = $2 WHERE name = $3",
                ancient,
                ancient,
                "stuck",
            )

        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(conn, now_utc=now)
        assert "stuck" in [c.name for c in claimed]

    async def test_archived_session_excluded(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """Once a session is archived, its scheduled_tasks must stop firing.

        Regression guard on the JOIN-against-sessions filter added in this PR.
        """
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("doomed"), account_id="acc_test_stub")
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(conn, now_utc=future)
        assert all(c.name != "doomed" for c in claimed)

    async def test_disabled_task_not_claimed(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        spec = ScheduledTaskCreate.model_validate(
            {"name": "off", "schedule": "*/5 * * * *", "command": "true", "enabled": False}
        )
        await st_service.add_task(pool, sid, spec, account_id="acc_test_stub")

        future = datetime.now(UTC) + timedelta(hours=1)
        async with pool.acquire() as conn, conn.transaction():
            claimed = await queries.fetch_and_claim_due_scheduled_tasks(conn, now_utc=future)
        assert all(c.name != "off" for c in claimed)


# ─── fire-recording + auto-disable ─────────────────────────────────────────


class TestRecordFire:
    async def test_ok_resets_consecutive_failures(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")

        # Bump failures then record ok.
        async with pool.acquire() as conn:
            await queries.record_scheduled_task_fire(
                conn,
                echo.id,
                status="error",
                consecutive_failures=3,
                fired_at=datetime.now(UTC),
            )
            await queries.record_scheduled_task_fire(
                conn,
                echo.id,
                status="ok",
                consecutive_failures=0,
                fired_at=datetime.now(UTC),
            )
            row = await conn.fetchrow(
                "SELECT consecutive_failures, last_fire_status, running_since "
                "FROM session_scheduled_tasks WHERE id = $1",
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
        echo = await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")

        async with pool.acquire() as conn:
            await queries.disable_scheduled_task(conn, echo.id)
            row = await conn.fetchrow(
                "SELECT enabled, next_fire FROM session_scheduled_tasks WHERE id = $1",
                echo.id,
            )
        assert row is not None
        assert row["enabled"] is False
        assert row["next_fire"] is None


# ─── HTTP surface ──────────────────────────────────────────────────────────


class TestHttp:
    async def _create_session_via_http(
        self, http_client: httpx.AsyncClient, agent_id: str, env_id: str
    ) -> str:
        r = await http_client.post(
            "/v1/sessions",
            json={"agent_id": agent_id, "environment_id": env_id},
        )
        assert r.status_code == 201, r.text
        return r.json()["id"]

    async def test_round_trip(
        self,
        pool: Any,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)

        # Initially empty.
        r = await http_client.get(f"/v1/sessions/{sid}/scheduled-tasks")
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []

        # Add.
        r = await http_client.post(
            f"/v1/sessions/{sid}/scheduled-tasks",
            json={"name": "poll", "schedule": "*/5 * * * *", "command": "echo hi"},
        )
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "poll"

        # List.
        r = await http_client.get(f"/v1/sessions/{sid}/scheduled-tasks")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 1

        # Update.
        r = await http_client.put(
            f"/v1/sessions/{sid}/scheduled-tasks/poll",
            json={"command": "echo updated"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["command"] == "echo updated"

        # Delete.
        r = await http_client.delete(f"/v1/sessions/{sid}/scheduled-tasks/poll")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/sessions/{sid}/scheduled-tasks")
        assert r.json()["data"] == []

    async def test_create_with_initial_scheduled_tasks(
        self,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
    ) -> None:
        env_id, agent_id = env_and_agent
        r = await http_client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "scheduled_tasks": [
                    {"name": "a", "schedule": "* * * * *", "command": "true"},
                    {"name": "b", "schedule": "0 9 * * *", "command": "echo b"},
                ],
            },
        )
        assert r.status_code == 201, r.text
        session = r.json()
        assert len(session["scheduled_tasks"]) == 2

    async def test_invalid_cron_rejected_at_boundary(
        self,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.post(
            f"/v1/sessions/{sid}/scheduled-tasks",
            json={"name": "bad", "schedule": "not a cron", "command": "true"},
        )
        assert r.status_code == 422, r.text

    async def test_delete_missing_returns_404(
        self,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.delete(f"/v1/sessions/{sid}/scheduled-tasks/nope")
        assert r.status_code == 404, r.text

    async def test_session_update_does_not_accept_scheduled_tasks(
        self,
        http_client: httpx.AsyncClient,
        env_and_agent: tuple[str, str],
    ) -> None:
        """SessionUpdate deliberately omits scheduled_tasks per #270 —
        regression guard."""
        env_id, agent_id = env_and_agent
        sid = await self._create_session_via_http(http_client, agent_id, env_id)
        r = await http_client.put(
            f"/v1/sessions/{sid}",
            json={"scheduled_tasks": [{"name": "x", "schedule": "* * * * *", "command": "true"}]},
        )
        # extra="forbid" on SessionUpdate → 422.
        assert r.status_code == 422, r.text


# ─── enrichment + clone propagation (post-review fixes) ────────────────────


class TestEnrichmentOnSessionMutations:
    """Regression guards on the code-review finding that ``clone``,
    ``update``, and ``archive`` were dropping ``scheduled_tasks`` from
    their response envelopes."""

    async def test_get_session_includes_scheduled_tasks(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.get_session(pool, sid, account_id="acc_test_stub")
        assert [t.name for t in s.scheduled_tasks] == ["p"]

    async def test_archive_session_enriches_scheduled_tasks(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")
        # Rows are preserved across archive; the enriched response must
        # reflect them (tick filter is what stops firing, not row removal).
        assert [t.name for t in s.scheduled_tasks] == ["p"]

    async def test_update_session_enriches_scheduled_tasks(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")
        s = await sessions_service.update_session(
            pool, sid, title="renamed", account_id="acc_test_stub"
        )
        assert [t.name for t in s.scheduled_tasks] == ["p"]

    async def test_clone_session_enriches_and_copies_scheduled_tasks(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """The clone must (1) DB-side copy the rows, and (2) enrich the
        response with the copied list. Pre-fix, neither happened."""
        env_id, agent_id = env_and_agent
        parent_sid = await _create_session(pool, env_id, agent_id)
        await st_service.add_task(
            pool,
            parent_sid,
            _spec("p", schedule="0 9 * * *"),
            account_id="acc_test_stub",
        )

        clone = await sessions_service.clone_session(pool, parent_sid, account_id="acc_test_stub")
        # Response is enriched.
        assert [t.name for t in clone.scheduled_tasks] == ["p"]

        # DB rows actually exist on the clone (independent IDs, fresh
        # runtime state, parent's next_fire preserved).
        clone_tasks = await st_service.list_tasks(pool, clone.id, account_id="acc_test_stub")
        assert len(clone_tasks) == 1
        parent_tasks = await st_service.list_tasks(pool, parent_sid, account_id="acc_test_stub")
        assert clone_tasks[0].id != parent_tasks[0].id  # fresh ULID
        assert clone_tasks[0].schedule == "0 9 * * *"
        assert clone_tasks[0].consecutive_failures == 0
        assert clone_tasks[0].last_fire_at is None
        # next_fire is preserved (the clone fires on the parent's cadence).
        assert clone_tasks[0].next_fire == parent_tasks[0].next_fire


class TestUpdatedAtAlwaysBumps:
    async def test_empty_patch_still_bumps_updated_at(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """A no-field PATCH (e.g. ``{}``) must still record an
        ``updated_at`` write — external pollers using
        ``updated_at > <since>`` mustn't miss the call."""
        from aios.models.scheduled_tasks import ScheduledTaskUpdate

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        initial = await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")

        # Sleep enough to let ``now()`` actually advance.
        import asyncio

        await asyncio.sleep(0.05)

        patched = await st_service.update_task(
            pool,
            sid,
            "p",
            ScheduledTaskUpdate.model_validate({}),
            account_id="acc_test_stub",
        )
        assert patched.updated_at > initial.updated_at


class TestArchiveRacePrevention:
    """The fire-handler must re-check ``session_archived_at`` between
    claim and execute — pre-fix, the tick filter held but the handler
    didn't, so an archive that landed in the race window would fire
    bash anyway."""

    async def test_archived_at_in_row_after_session_archive(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        async with pool.acquire() as conn:
            row = await queries.unscoped_get_scheduled_task_row(conn, echo.id)
        # Handler reads session_archived_at off the row and skips when
        # set — verify the JOIN actually surfaces it.
        assert row.session_archived_at is not None

    async def test_handler_skips_archived_session(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        """End-to-end of the archive-race fix: drive the handler with
        a sandbox_registry that would raise if entered. The handler
        must short-circuit before the sandbox call."""
        from unittest import mock

        from aios.harness import runtime, scheduled_task_runner

        env_id, agent_id = env_and_agent
        sid = await _create_session(pool, env_id, agent_id)
        echo = await st_service.add_task(pool, sid, _spec("p"), account_id="acc_test_stub")

        # Simulate the tick having claimed the row (running_since=now,
        # next_fire advanced) before archive.
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE session_scheduled_tasks SET running_since = now() WHERE id = $1",
                echo.id,
            )

        # Archive lands between claim and fire-handler-execute.
        await sessions_service.archive_session(pool, sid, account_id="acc_test_stub")

        # Sandbox that would explode if the handler ever calls into it.
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
            await scheduled_task_runner.run_scheduled_task_step(echo.id)
        finally:
            runtime.pool = prev_pool
            runtime.sandbox_registry = prev_sandbox

        # Post-skip, running_since must be cleared so the row doesn't
        # stay stuck if the session is later unarchived.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT running_since, last_fire_status FROM session_scheduled_tasks WHERE id = $1",
                echo.id,
            )
        assert row is not None
        assert row["running_since"] is None
        assert row["last_fire_status"] == "skipped"


class TestRowDeletedRace:
    """If the row is deleted between tick-claim and fire-handler-execute
    (e.g. agent calls schedule_task_remove), the handler must exit
    silently — no exception out of the procrastinate task."""

    async def test_handler_swallows_deleted_row(
        self, pool: Any, env_and_agent: tuple[str, str]
    ) -> None:
        from aios.harness import runtime, scheduled_task_runner
        from aios.ids import SCHEDULED_TASK, make_id

        prev_pool = runtime.pool
        runtime.pool = pool
        try:
            # An ID that doesn't exist in the DB.
            phantom = make_id(SCHEDULED_TASK)
            # Must not raise.
            await scheduled_task_runner.run_scheduled_task_step(phantom)
        finally:
            runtime.pool = prev_pool
