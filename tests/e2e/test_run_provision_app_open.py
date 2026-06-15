"""E2E: provision-bearing run tests get an OPEN procrastinate app (#1169).

Two provision-bearing e2e tests (#1148, #1163) crashed at **setup** with
``procrastinate.exceptions.AppNotOpen`` — before exercising the behavior under
test — because the run-provision path enqueues a procrastinate task and the
test context never opened the procrastinate app.

The fix opens the app in the shared e2e fixture/conftest so every
provision-bearing e2e inherits it (``docker_harness`` depends on the
``open_procrastinate_app`` fixture). This test pins that contract directly:
``workflows_service.create_run`` ends with a real ``defer_run_wake`` —
``app.configure_task("harness.wake_workflow", ...).defer_async(...)`` — which
raises ``AppNotOpen`` against a closed app. Driven through ``docker_harness``
WITHOUT mocking ``defer_run_wake``, it must instead land a real
``procrastinate_jobs`` row.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db.queries import workflows as wf_queries
from aios.models.agents import ToolSpec
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.services import environments as environments_service
from aios.workflows import service as workflows_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_ALLOWED_HOST = "api.github.com"
_SCRIPT = """async def main(input):
    return "ok"
"""


@needs_docker
async def test_create_run_enqueues_wake_with_open_app(docker_harness: Harness) -> None:
    """The shared fixture opens the procrastinate app, so the run-provision
    path's real ``defer_run_wake`` enqueues instead of raising ``AppNotOpen``."""
    pool: asyncpg.Pool[Any] = docker_harness._pool

    env = await environments_service.create_environment(
        pool,
        account_id=_ACCOUNT_ID,
        name="run-provision-app-open-e2e",
        config=EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=[_ALLOWED_HOST]),
        ),
    )
    async with pool.acquire() as conn:
        workflow = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT_ID,
            name="run-provision-app-open-e2e",
            script=_SCRIPT,
            tools=[ToolSpec(type="bash")],
        )

    # No ``defer_run_wake`` mock here: with a closed app this line raises
    # ``procrastinate.AppNotOpen`` at the tail of ``create_run`` — the exact
    # setup-time crash #1148/#1163 hit on the provision path.
    run = await workflows_service.create_run(
        pool,
        account_id=_ACCOUNT_ID,
        workflow_id=workflow.id,
        environment_id=env.id,
    )

    async with pool.acquire() as conn:
        enqueued = await conn.fetchval(
            "SELECT count(*) FROM procrastinate_jobs "
            "WHERE task_name = 'harness.wake_workflow' "
            "AND args->>'run_id' = $1",
            run.id,
        )
    assert int(enqueued or 0) >= 1
