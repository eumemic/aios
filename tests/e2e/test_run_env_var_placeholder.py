"""E2E: workflow-run bash sees env-var credential placeholders, not secrets.

Restores the run-side acceptance leg from #882: a workflow run with an
``environment_variable`` vault credential bound provisions a real Docker sandbox,
materializes ``SECRET_NAME=<placeholder>``, and journals only that placeholder in the
bash result. The plaintext secret must not enter the model/session event surface or
the run journal.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest
from pydantic import SecretStr

from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.vaults import VaultCredentialCreate
from aios.services import environments as environments_service
from aios.services import vaults as vaults_service
from aios.services.vaults import resolve_run_env_var_credentials
from aios.workflows import run_tools
from aios.workflows import service as workflows_service
from aios.workflows.step import run_workflow_step
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_SECRET_NAME = "GITHUB_TOKEN"
_SENTINEL_SECRET = "ghp_RUN_SENTINEL_PLAINTEXT_DO_NOT_LEAK"
_ALLOWED_HOST = "api.github.com"
_SCRIPT = f"""async def main(input):
    return await tool('bash', {{"command": 'printf "%s" "${_SECRET_NAME}"'}})
"""


async def _drive_run_to_completion(pool: asyncpg.Pool[Any], run_id: str) -> None:
    for _ in range(10):
        await run_workflow_step(run_id)
        pending = [task for (rid, _), task in run_tools._INFLIGHT.items() if rid == run_id]
        if pending:
            await asyncio.gather(*pending, return_exceptions=False)
        async with pool.acquire() as conn:
            run = await wf_queries.get_run_for_step(conn, run_id)
        assert run is not None
        if run.status == "completed":
            return
    raise AssertionError("workflow run did not complete")


@needs_docker
async def test_run_bash_env_var_placeholder_round_trip(docker_harness: Harness) -> None:
    pool = docker_harness._pool
    crypto_box = runtime.require_crypto_box()

    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="run-envvar-e2e", metadata={}
    )
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=_ACCOUNT_ID,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name=_SECRET_NAME,
            secret_value=SecretStr(_SENTINEL_SECRET),
            allowed_hosts=[_ALLOWED_HOST],
        ),
    )
    env = await environments_service.create_environment(
        pool,
        account_id=_ACCOUNT_ID,
        name="run-envvar-e2e",
        config=EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=[_ALLOWED_HOST]),
        ),
    )
    async with pool.acquire() as conn:
        workflow = await wf_queries.insert_workflow(
            conn,
            account_id=_ACCOUNT_ID,
            name="run-envvar-e2e",
            script=_SCRIPT,
            tools=[ToolSpec(type="bash")],
        )

    with (
        mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
        mock.patch("aios.workflows.run_sandbox.defer_run_wake", new=AsyncMock()),
    ):
        run = await workflows_service.create_run(
            pool,
            account_id=_ACCOUNT_ID,
            workflow_id=workflow.id,
            environment_id=env.id,
            vault_ids=[vault.id],
        )
        async with pool.acquire() as conn:
            resolved = await resolve_run_env_var_credentials(
                conn, crypto_box, run.id, account_id=_ACCOUNT_ID
            )
        assert len(resolved) == 1
        placeholder = resolved[0].placeholder

        await _drive_run_to_completion(pool, run.id)

    async with pool.acquire() as conn:
        completed = await wf_queries.get_run_for_step(conn, run.id)
        run_event_rows = await conn.fetch(
            "SELECT row_to_json(e)::text AS raw FROM wf_run_events e WHERE run_id = $1 ORDER BY seq",
            run.id,
        )
        signal_rows = await conn.fetch(
            "SELECT row_to_json(s)::text AS raw FROM wf_run_signals s WHERE run_id = $1 ORDER BY delivered_at",
            run.id,
        )
        session_event_rows = await conn.fetch(
            """
            SELECT row_to_json(e)::text AS raw
              FROM events e
              JOIN sessions s ON s.id = e.session_id
             WHERE s.parent_run_id = $1 OR e.session_id = $1
             ORDER BY e.seq
            """,
            run.id,
        )

    assert completed is not None and completed.status == "completed"
    assert completed.output["stdout"] == placeholder
    assert completed.output["stdout"] != _SENTINEL_SECRET

    raw_session_events = "\n".join(row["raw"] for row in session_event_rows)
    raw_run_events = "\n".join(row["raw"] for row in run_event_rows)
    raw_run_signals = "\n".join(row["raw"] for row in signal_rows)
    assert _SENTINEL_SECRET not in raw_session_events
    assert _SENTINEL_SECRET not in raw_run_events
    assert _SENTINEL_SECRET not in raw_run_signals
    assert placeholder in raw_run_events
