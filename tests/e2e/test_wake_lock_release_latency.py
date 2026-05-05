"""Lock-release wake-pickup latency must be sub-second (issue #237)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import asyncpg
import pytest
from procrastinate import PsycopgConnector

from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.conftest import wait_for_predicate

_TERMINAL_STATUSES = ("succeeded", "failed", "aborted", "cancelled")
_TERMINAL_EVENT_TYPES = ("succeeded", "failed", "aborted")


async def _has_doing_wake(pool: asyncpg.Pool[Any], session_id: str) -> bool:
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM procrastinate_jobs "
            "WHERE task_name = 'harness.wake_session' "
            "AND args->>'session_id' = $1 AND status = 'doing'",
            session_id,
        )
    return bool(n)


async def _both_wakes_terminal(pool: asyncpg.Pool[Any], session_id: str) -> bool:
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM procrastinate_jobs "
            "WHERE task_name = 'harness.wake_session' "
            "AND args->>'session_id' = $1 AND status = ANY($2)",
            session_id,
            list(_TERMINAL_STATUSES),
        )
    return int(n or 0) >= 2


@needs_docker
class TestWakeLockReleaseLatencyE2E:
    @pytest.mark.asyncio
    async def test_pickup_gap_after_lock_release_is_sub_second(
        self, real_wake_setup: asyncpg.Pool[Any], aios_env: dict[str, str]
    ) -> None:
        """When a lock-blocked wake becomes eligible, the worker must pick
        it up within a single LISTEN/NOTIFY round-trip (<200ms)."""
        from aios.harness.procrastinate_app import app as procrastinate_app
        from aios.harness.wake import defer_wake

        pool = real_wake_setup

        agent = await agents_service.create_agent(
            pool,
            name="lock-release-test-agent",
            model="fake/test",
            system="t",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(pool, name="lock-release-test-env")
        session = await sessions_service.create_session(
            pool, agent_id=agent.id, environment_id=env.id, title="lock-release-test", metadata={}
        )
        session_id = session.id

        first_call_done = asyncio.Event()

        async def fake_step(session_id: str, **kwargs: Any) -> None:
            # First call holds the lock for 1s — comfortably longer than
            # the 200ms assertion window. Subsequent calls return fast.
            if not first_call_done.is_set():
                first_call_done.set()
                await asyncio.sleep(1.0)

        conninfo = aios_env["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://")
        new_connector = PsycopgConnector(conninfo=conninfo)
        await new_connector.open_async()

        try:
            with (
                procrastinate_app.replace_connector(new_connector),
                mock.patch("aios.harness.loop.run_session_step", fake_step),
            ):
                procrastinate_app.perform_import_paths()
                # Construct the Worker directly so we can stop() it
                # explicitly. wait=True keeps it alive past the
                # initial-fetch drain so B (deferred mid-A) gets picked
                # up after A's lock releases.
                worker = procrastinate_app._worker(
                    queues=["sessions"],
                    concurrency=4,
                    wait=True,
                    install_signal_handlers=False,
                )

                await defer_wake(pool, session_id, cause="message")
                worker_task = asyncio.create_task(worker.run())

                await wait_for_predicate(
                    lambda: _has_doing_wake(pool, session_id),
                    max_wait_s=5.0,
                    interval_s=0.02,
                )
                await defer_wake(pool, session_id, cause="message")

                await wait_for_predicate(
                    lambda: _both_wakes_terminal(pool, session_id),
                    max_wait_s=15.0,
                    interval_s=0.02,
                )
                worker.stop()
                await asyncio.wait_for(worker_task, timeout=5.0)
        finally:
            await new_connector.close_async()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT j.id AS job_id, "
                "       (SELECT at FROM procrastinate_events e "
                "         WHERE e.job_id = j.id AND e.type = 'started') AS started_at, "
                "       (SELECT at FROM procrastinate_events e "
                "         WHERE e.job_id = j.id "
                "         AND e.type = ANY($2)) AS ended_at "
                "FROM procrastinate_jobs j "
                "WHERE j.task_name = 'harness.wake_session' "
                "AND j.args->>'session_id' = $1 "
                "ORDER BY j.id ASC",
                session_id,
                list(_TERMINAL_EVENT_TYPES),
            )

        assert len(rows) == 2, f"expected 2 wake_session jobs, got {len(rows)}: {rows}"
        a_ended = rows[0]["ended_at"]
        b_started = rows[1]["started_at"]
        assert a_ended is not None and b_started is not None, f"missing timestamps: {rows}"

        gap_s = (b_started - a_ended).total_seconds()
        assert gap_s < 0.2, (
            f"lock-release pickup gap is {gap_s:.3f}s, expected <0.2s — "
            f"trigger aios_jobs_notify_lock_released_v1 may not be firing."
        )
