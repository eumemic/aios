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

pytestmark = pytest.mark.docker


@needs_docker
class TestWakeLockReleaseLatencyE2E:
    @pytest.mark.asyncio
    async def test_pickup_gap_after_lock_release_is_sub_second(
        self, real_wake_setup: asyncpg.Pool[Any], aios_env: dict[str, str]
    ) -> None:
        """When a lock-blocked wake becomes eligible, the worker must pick
        it up within a single LISTEN/NOTIFY round-trip (<200ms)."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.jobs.app import app as procrastinate_app
        from aios.jobs.app import defer_wake

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
            account_id=account_id,
        )
        env = await environments_service.create_environment(
            pool, name="lock-release-test-env", account_id=account_id
        )
        session = await sessions_service.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="lock-release-test",
            metadata={},
            account_id=account_id,
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

                await defer_wake(pool, session_id, cause="message", account_id=account_id)
                worker_task = asyncio.create_task(worker.run())  # type: ignore[no-untyped-call]

                async def _count_wakes(statuses: tuple[str, ...]) -> int:
                    async with pool.acquire() as conn:
                        n = await conn.fetchval(
                            "SELECT count(*) FROM procrastinate_jobs "
                            "WHERE task_name = 'harness.wake_session' "
                            "AND args->>'session_id' = $1 AND status = ANY($2)",
                            session_id,
                            list(statuses),
                        )
                    return int(n or 0)

                async def _has_doing() -> bool:
                    return await _count_wakes(("doing",)) >= 1

                # Procrastinate's job-status enum has 'cancelled' but its
                # event-type enum does not.
                async def _both_terminal() -> bool:
                    return await _count_wakes(("succeeded", "failed", "aborted", "cancelled")) >= 2

                await wait_for_predicate(_has_doing, max_wait_s=5.0, interval_s=0.02)
                await defer_wake(pool, session_id, cause="message", account_id=account_id)
                await wait_for_predicate(_both_terminal, max_wait_s=15.0, interval_s=0.02)
                worker.stop()  # type: ignore[no-untyped-call]
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
                ["succeeded", "failed", "aborted"],
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
