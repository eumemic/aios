"""Wake jobs for distinct sessions must dispatch concurrently.

Exercises the full pipeline against a real Postgres testcontainer:
``defer_wake`` writes a procrastinate job, a real worker fetches it,
the ``wake_session`` task body invokes ``run_session_step`` (patched
here to a sleep + counter so the test focuses on dispatch behaviour
rather than inference).

Implementation-agnostic: never reads ``lock`` / ``queueing_lock`` or
any other procrastinate internals — only the user-visible parallelism
property. If a future change re-serializes wakes across sessions
(e.g. the bug that issue #192 fixed), peak collapses to 1 and this
test fails.
"""

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

pytestmark = pytest.mark.docker


async def _create_session(pool: asyncpg.Pool[Any], agent_id: str, env_id: str) -> str:
    account_id = "acc_test_stub"  # PR 3 scaffolding
    session = await sessions_service.create_session(
        pool,
        agent_id=agent_id,
        environment_id=env_id,
        title="conc-test",
        metadata={},
        account_id=account_id,
    )
    return session.id


@needs_docker
class TestWorkerConcurrencyE2E:
    async def test_n_sessions_dispatch_concurrently(
        self, real_wake_setup: asyncpg.Pool[Any], aios_env: dict[str, str]
    ) -> None:
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.harness.procrastinate_app import app as procrastinate_app
        from aios.services.wake import defer_wake

        pool = real_wake_setup
        n_sessions = 8
        concurrency = 4
        per_step = 0.4

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
            account_id=account_id,
        )
        env = await environments_service.create_environment(
            pool, name="conc-test-env", account_id=account_id
        )

        session_ids = await asyncio.gather(
            *(_create_session(pool, agent.id, env.id) for _ in range(n_sessions))
        )

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

        conninfo = aios_env["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://")
        new_connector = PsycopgConnector(conninfo=conninfo)
        await new_connector.open_async()

        try:
            with (
                procrastinate_app.replace_connector(new_connector),
                mock.patch("aios.harness.loop.run_session_step", fake_step),
            ):
                await asyncio.gather(
                    *(
                        defer_wake(pool, sid, cause="message", account_id=account_id)
                        for sid in session_ids
                    )
                )

                # ``wait=False`` exits when ``fetch_job`` returns ``None``;
                # the worker's shutdown path awaits in-flight tasks first,
                # so this resolves only after every job has run.
                await procrastinate_app.run_worker_async(
                    queues=["sessions"],
                    concurrency=concurrency,
                    wait=False,
                    install_signal_handlers=False,
                    fetch_job_polling_interval=0.1,
                )
        finally:
            await new_connector.close_async()

        assert peak == concurrency, f"peak={peak}, expected {concurrency}"
