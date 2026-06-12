"""E2E: run-side env-var credentials round-trip as placeholders.

Restores the run-path acceptance leg from #882: a workflow run with a
bound ``environment_variable`` vault credential invokes ``tool('bash', …)``
in a real Docker sandbox. The sandbox environment must contain the run's
minted placeholder, while neither the bash-executing session log nor the
run journal may contain the decrypted secret.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest import mock

import pytest
from pydantic import SecretStr

from aios.db import queries
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.vaults import VaultCredentialCreate
from aios.services import vaults as vaults_service
from aios.services.vaults import resolve_run_env_var_credentials
from aios.workflows import run_tools
from aios.workflows import service as wf_service
from aios.workflows.step import run_workflow_step
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_SECRET_NAME = "GITHUB_TOKEN"
_SENTINEL_SECRET = "ghp_RUN_SENTINEL_PLAINTEXT_DO_NOT_LEAK"
_ALLOWED_HOST = "api.github.com"
_SCRIPT = (
    "async def main(input):\n"
    f"    return await tool('bash', {{'command': 'printf \"%s\" \"${_SECRET_NAME}\"'}})\n"
)


async def _drain_run_tool_tasks() -> None:
    tasks = list(run_tools._INFLIGHT.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _raw_session_event_rows(pool: Any, session_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT row_to_json(events)::text AS raw FROM events "
            "WHERE session_id = $1 ORDER BY seq",
            session_id,
        )
    return [str(row["raw"]) for row in rows]


async def _raw_run_journal_rows(pool: Any, run_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT row_to_json(wf_run_events)::text AS raw FROM wf_run_events "
            "WHERE run_id = $1 ORDER BY seq",
            run_id,
        )
    return [str(row["raw"]) for row in rows]


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

    async with pool.acquire() as conn:
        env = await queries.insert_environment(
            conn,
            account_id=_ACCOUNT_ID,
            name="run-envvar-e2e",
            config=EnvironmentConfig(
                networking=LimitedNetworking(type="limited", allowed_hosts=[_ALLOWED_HOST])
            ),
        )
        wf = await queries.workflows.insert_workflow(
            conn,
            account_id=_ACCOUNT_ID,
            name="run-envvar-e2e",
            script=_SCRIPT,
            tools=[ToolSpec(type="bash")],
        )

    with mock.patch("aios.workflows.service.defer_run_wake", new=mock.AsyncMock()):
        run = await wf_service.create_run(
            pool,
            account_id=_ACCOUNT_ID,
            workflow_id=wf.id,
            environment_id=env.id,
            vault_ids=[vault.id],
        )

    async with pool.acquire() as conn:
        resolved = await resolve_run_env_var_credentials(
            conn, crypto_box, run.id, account_id=_ACCOUNT_ID
        )
    assert len(resolved) == 1
    placeholder = resolved[0].placeholder
    assert resolved[0].secret_name == _SECRET_NAME
    assert resolved[0].secret_value == _SENTINEL_SECRET

    with mock.patch("aios.workflows.run_sandbox.defer_run_wake", new=mock.AsyncMock()):
        await run_workflow_step(run.id)
        await _drain_run_tool_tasks()
        await run_workflow_step(run.id)

    async with pool.acquire() as conn:
        completed = await queries.workflows.get_run_for_step(conn, run.id)
        journal_events = await queries.workflows.list_run_events(conn, run.id)
        bash_session_id = await conn.fetchval(
            "SELECT id FROM sessions WHERE parent_run_id = $1 ORDER BY created_at DESC LIMIT 1",
            run.id,
        )
    assert completed is not None and completed.status == "completed"
    assert completed.output["stdout"] == placeholder
    assert _SENTINEL_SECRET not in json.dumps(completed.output, sort_keys=True)
    assert bash_session_id is not None

    session_rows = await _raw_session_event_rows(pool, bash_session_id)
    journal_rows = await _raw_run_journal_rows(pool, run.id)
    assert session_rows, "bash-executing session must have an event log to sweep"
    assert journal_events, "run must have journal events to sweep"
    assert all(_SENTINEL_SECRET not in row for row in session_rows)
    assert all(_SENTINEL_SECRET not in row for row in journal_rows)
    assert any(placeholder in row for row in session_rows)
    assert any(placeholder in row for row in journal_rows)
