"""End-to-end regression test for issue #192 — wake jobs run concurrently.

Exercises the full pipeline against a real Postgres testcontainer:

* Real ``defer_wake`` writes the procrastinate ``wake_session`` job
  (with per-session ``lock`` and ``queueing_lock``).
* A real procrastinate worker fetches and dispatches the jobs.
* The ``wake_session`` task body invokes ``run_session_step`` — patched
  here to a sleep + counter, since this test cares about *whether* the
  worker dispatches concurrently, not what happens inside a step.

Behavioural contract: deferring wake jobs for N distinct sessions in
burst, with the worker configured at ``concurrency=C``, must produce
peak observed concurrency equal to ``C`` (assuming N ≥ C). Implementation-
agnostic — the test never reads ``lock`` or ``queueing_lock`` columns
or any other procrastinate internals; it only observes the user-visible
parallelism.

If a future change re-introduces the #192 bug — by reverting per-call
locks, by adding cross-session serialization, by accidentally narrowing
the worker pool — peak will collapse to 1 and this test fails loudly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
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


@pytest.fixture
async def real_wake_setup(aios_env: dict[str, str]) -> AsyncIterator[asyncpg.Pool[Any]]:
    """Real Postgres pool + procrastinate schema, no harness mocks.

    The standard ``harness`` fixture installs a no-op ``defer_wake``
    so most e2e tests can drive steps deterministically. This test is
    the one place we *want* the real ``defer_wake`` to fire — it's
    exercising the procrastinate dispatch path end-to-end. So we
    build a bare pool here and bootstrap procrastinate's schema.
    """
    from procrastinate import App

    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=8)

    # Bootstrap procrastinate's schema if missing (mirrors the pattern
    # in ``tests/e2e/test_reap_stalled_jobs.py``).
    async with pool.acquire() as conn:
        present = await conn.fetchval("SELECT to_regclass('procrastinate_jobs')")
    if present is None:
        conninfo = aios_env["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://")
        tmp_app = App(connector=PsycopgConnector(conninfo=conninfo))
        await tmp_app.open_async()
        try:
            await tmp_app.schema_manager.apply_schema_async()
        finally:
            await tmp_app.close_async()

    try:
        yield pool
    finally:
        await pool.close()


async def _create_session(pool: asyncpg.Pool[Any], agent_id: str, env_id: str) -> str:
    session = await sessions_service.create_session(
        pool,
        agent_id=agent_id,
        environment_id=env_id,
        title="conc-test",
        metadata={},
    )
    return session.id


@needs_docker
class TestWorkerConcurrencyE2E:
    """Workers must dispatch wake jobs for distinct sessions concurrently."""

    async def test_n_sessions_dispatch_concurrently(
        self, real_wake_setup: asyncpg.Pool[Any], aios_env: dict[str, str]
    ) -> None:
        """8 sessions, concurrency=4 ⇒ peak observed concurrency == 4.

        Deferring wakes for 8 distinct ``session_id`` values and running
        a worker at ``concurrency=4`` must saturate the worker — peak
        observed ``run_session_step`` invocations in flight at once
        equals the configured concurrency. The strict-serial floor that
        #192 produced (peak == 1) is what this test guards against.

        Wall-clock isn't asserted: peak alone distinguishes the failure
        modes (peak == 1 is the bug; peak > 1 is correct dispatch). A
        hardware-dependent timing assertion would just add flake without
        adding signal.
        """
        from aios.harness.procrastinate_app import app as procrastinate_app
        from aios.harness.wake import defer_wake

        pool = real_wake_setup

        # Create one agent + environment + N sessions.
        agent = await agents_service.create_agent(
            pool,
            name="conc-test-agent",
            model="fake/test",
            system="t",
            tools=[ToolSpec(type="bash")],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(pool, name="conc-test-env")

        n_sessions = 8
        concurrency = 4
        per_step = 0.4

        session_ids = [await _create_session(pool, agent.id, env.id) for _ in range(n_sessions)]

        # Track in-flight steps via a shared counter. The wake_session
        # task imports run_session_step at call time, so patching the
        # module attribute affects every dispatched job.
        in_flight = 0
        peak = 0

        async def fake_step(session_id: str, **kwargs: Any) -> None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(per_step)
            finally:
                in_flight -= 1

        # Point the procrastinate_app singleton at the testcontainer.
        # Its connector was bound at import time to whatever DB was
        # active then; this swap is the standard pattern (see
        # tests/unit/test_wake_instrumentation.py).
        conninfo = aios_env["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://")
        new_connector = PsycopgConnector(conninfo=conninfo)
        await new_connector.open_async()

        try:
            with (
                procrastinate_app.replace_connector(new_connector),
                mock.patch("aios.harness.loop.run_session_step", fake_step),
            ):
                # Defer one wake per session via the production code path.
                for sid in session_ids:
                    await defer_wake(pool, sid, cause="message")

                # Run the worker until the queue drains. ``wait=False``
                # exits when ``fetch_job`` returns ``None``; the worker's
                # shutdown path awaits in-flight tasks before returning.
                await procrastinate_app.run_worker_async(
                    queues=["sessions"],
                    concurrency=concurrency,
                    wait=False,
                    install_signal_handlers=False,
                    fetch_job_polling_interval=0.1,
                )
        finally:
            await new_connector.close_async()

        assert peak == concurrency, (
            f"peak in-flight wake_session invocations = {peak}, expected "
            f"{concurrency} for {n_sessions} distinct sessions. The "
            f"worker is not dispatching concurrently — likely a "
            f"regression of issue #192 (per-session locks must come "
            f"from defer_wake's configure_task call, not from a "
            f"decorator-level template)."
        )
