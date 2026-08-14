"""End-to-end filesystem contract for workflow-spawned agent workspaces."""

from __future__ import annotations

import secrets
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.services import workflows as workflows_service
from aios.tools import workflow_completion
from aios.workflows.step import run_workflow_step
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = [needs_docker, pytest.mark.docker]


async def _child_id(pool: asyncpg.Pool[object], run_id: str, ordinal: int) -> str:
    async with pool.acquire() as conn:
        events = await wf_queries.list_run_events(conn, run_id)
    starts = [event for event in events if event.type == "call_started"]
    return str(starts[ordinal].payload["child_session_id"])


async def _finish_child(pool: asyncpg.Pool[object], child_id: str, value: str) -> None:
    async with pool.acquire() as conn:
        request_ids = await db_queries.get_open_request_ids(
            conn, child_id, account_id="acc_test_stub"
        )
    with mock.patch("aios.tools.workflow_completion.defer_run_wake", new=AsyncMock()):
        await workflow_completion.return_handler(
            child_id, {"request_id": request_ids[0], "value": value}
        )


async def test_real_workflow_agent_spawn_shares_workspace_and_fresh_isolates(
    docker_harness: Harness,
) -> None:
    """Exercise run creation -> agent() -> session provision, not hand-built mounts."""
    launcher = await docker_harness.start("workspace contract launcher")
    # `first`/`second` pass workspace="shared" EXPLICITLY: this test asserts sharing
    # BEHAVIOUR, not what the default happens to be. The default's value is pinned by
    # its own test (tests/integration/test_wf_host.py::
    # test_generic_agent_spec_includes_default_workspace), so a deliberate change to
    # the default fails that one obvious test instead of silently reddening this one.
    script = f"""async def main(input):
    first = await agent("first", agent_id={launcher.agent_id!r}, workspace="shared")
    second = await agent("second", agent_id={launcher.agent_id!r}, workspace="shared")
    isolated = await agent("fresh", agent_id={launcher.agent_id!r}, workspace="fresh")
    return [first, second, isolated]
"""
    async with docker_harness._pool.acquire() as conn:
        workflow = await wf_queries.insert_workflow(
            conn, account_id="acc_test_stub", name="workspace-contract", script=script
        )
    run = await workflows_service.create_run(
        docker_harness._pool,
        account_id="acc_test_stub",
        workflow_id=workflow.id,
        environment_id=launcher.environment_id,
        input=None,
    )

    registry = runtime.require_sandbox_registry()
    backend = registry._backend
    run_handle = await registry.get_or_provision_run(run.id)
    marker = secrets.token_hex(48)
    write = await backend.exec(
        run_handle,
        f"printf %s {marker} > marker.bin",
        timeout_seconds=30,
        max_output_bytes=10_000,
    )
    assert write.exit_code == 0

    with (
        mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
        mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
    ):
        await run_workflow_step(run.id)
        first_id = await _child_id(docker_harness._pool, run.id, 0)
        first = await registry.get_or_provision(first_id, pool=docker_harness._pool)
        read = await backend.exec(
            first,
            "cat marker.bin",
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert read.exit_code == 0 and read.stdout == marker
        sibling = secrets.token_hex(48)
        wrote = await backend.exec(
            first,
            f"printf %s {sibling} > sibling.bin",
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert wrote.exit_code == 0
        assert (run_handle.workspace_path / "sibling.bin").read_text() == sibling

        await _finish_child(docker_harness._pool, first_id, "first-ok")
        await run_workflow_step(run.id)
        second_id = await _child_id(docker_harness._pool, run.id, 1)
        second = await registry.get_or_provision(second_id, pool=docker_harness._pool)
        sees_both = await backend.exec(
            second,
            f'test "$(cat marker.bin)" = {marker} && test "$(cat sibling.bin)" = {sibling}',
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert sees_both.exit_code == 0
        second_marker = secrets.token_hex(48)
        wrote_back = await backend.exec(
            second,
            f"printf %s {second_marker} > second.bin",
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert wrote_back.exit_code == 0
        first_sees_second = await backend.exec(
            first,
            f'test "$(cat second.bin)" = {second_marker}',
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert first_sees_second.exit_code == 0

        await _finish_child(docker_harness._pool, second_id, "second-ok")
        await run_workflow_step(run.id)
        fresh_id = await _child_id(docker_harness._pool, run.id, 2)
        fresh = await registry.get_or_provision(fresh_id, pool=docker_harness._pool)
        isolated = await backend.exec(
            fresh,
            "test ! -e marker.bin && test ! -e sibling.bin",
            timeout_seconds=30,
            max_output_bytes=10_000,
        )
        assert isolated.exit_code == 0

    async with docker_harness._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT r.workspace_path, "
            "array_agg(s.workspace_volume_path ORDER BY s.id) FILTER (WHERE s.id <> $2) AS shared, "
            "(array_agg(s.workspace_volume_path) FILTER (WHERE s.id = $2))[1] AS fresh "
            "FROM wf_runs r JOIN sessions s ON s.parent_run_id = r.id "
            "WHERE r.id = $1 GROUP BY r.workspace_path",
            run.id,
            fresh_id,
        )
    assert row is not None
    assert row["shared"] == [row["workspace_path"], row["workspace_path"]]
    assert row["fresh"] != row["workspace_path"]

    await _finish_child(docker_harness._pool, fresh_id, "fresh-ok")
    with mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()):
        await run_workflow_step(run.id)
    async with docker_harness._pool.acquire() as conn:
        completed = await wf_queries.get_run_for_step(conn, run.id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.output == ["first-ok", "second-ok", "fresh-ok"]
