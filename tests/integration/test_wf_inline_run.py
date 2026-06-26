"""Inline-script run launch (T5, #1466) — end to end against a real Postgres.

A run has always pinned its own immutable ``script`` snapshot at launch; the
``workflows`` row was merely the *source* of that snapshot. T5 adds an
**inline-script arm** to ``create_run`` / ``call_workflow``: launch a one-shot
run directly from an inline ``{script, schemas, surface}`` body, with **NO
``workflows`` row created**.

Covered here:

* an inline run executes its snapshotted script with no ``workflows`` row created
  (the workflows row count is unchanged; the run's pinned snapshot holds the script);
* the inline run produces **identical execution semantics** to the equivalent
  register-then-run, minus the persisted definition;
* the inline script's declared surface is clamped to the launcher — a surface that
  exceeds the launcher raises ``ForbiddenError``; an operator (no launcher) launch
  binds the surface verbatim;
* exactly-one-arm enforcement: neither / both ``workflow_id`` + ``inline`` is a
  ``ValidationError``; ``version`` is rejected on the inline arm.

The ``created_by`` stamping (#4) + hide-by-default scoping (#5) acceptance bullet
is consumed here but defined there — those columns do not yet exist on ``wf_runs``,
so the "hidden from another session's list / visible with ?include_spawned" check
lands with #5. This child makes inline runs GC-eligible by carrying no definition
row; the ownership/visibility stamping rides #4/#5.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ForbiddenError, ValidationError
from aios.harness import runtime
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.workflows import run_tools, service
from aios.workflows.service import InlineScript
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_INLINE_SCRIPT = "async def main(input):\n    return {'doubled': input * 2}\n"


@pytest.fixture
async def wf_pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_inl', NULL, TRUE, 'inl-root')"
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_inl', 'inl-env', '{}'::jsonb, 'acc_inl')"
            )
        run_tools._INFLIGHT.clear()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=AsyncMock()),
        ):
            yield pool
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _workflows_count(pool: asyncpg.Pool[Any]) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT count(*) FROM workflows WHERE account_id = $1", "acc_inl"
        )


# ─── no workflows row + snapshot holds the script ────────────────────────────


async def test_inline_run_creates_no_workflows_row(wf_pool: asyncpg.Pool[Any]) -> None:
    before = await _workflows_count(wf_pool)

    run = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        inline=InlineScript(script=_INLINE_SCRIPT),
        environment_id="env_inl",
        input=21,
    )

    # No definition row was created; the run's pinned snapshot holds the script,
    # and the inline run carries NO workflow_id / source_version.
    assert await _workflows_count(wf_pool) == before
    assert run.workflow_id is None
    assert run.source_version is None
    assert run.script == _INLINE_SCRIPT
    assert run.script_sha == hashlib.sha256(_INLINE_SCRIPT.encode("utf-8")).hexdigest()


async def test_inline_run_executes_and_completes(wf_pool: asyncpg.Pool[Any]) -> None:
    run = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        inline=InlineScript(script=_INLINE_SCRIPT),
        environment_id="env_inl",
        input=21,
    )
    await run_workflow_step(run.id)
    async with wf_pool.acquire() as conn:
        done = await wf_queries.get_run_for_step(conn, run.id)
    assert done is not None
    assert done.status == "completed"
    assert done.output == {"doubled": 42}
    # Still no workflows row after a full execution.
    assert await _workflows_count(wf_pool) == 0


# ─── identical execution semantics to register-then-run ──────────────────────


async def test_inline_matches_register_then_run(wf_pool: asyncpg.Pool[Any]) -> None:
    """An inline run produces identical execution semantics to the equivalent
    register-then-run, minus the persisted definition."""
    # Register-then-run.
    async with wf_pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_inl", name="doubler", script=_INLINE_SCRIPT
        )
    registered = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        workflow_id=wf.id,
        environment_id="env_inl",
        input=21,
    )
    await run_workflow_step(registered.id)

    # The inline equivalent.
    inline = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        inline=InlineScript(script=_INLINE_SCRIPT),
        environment_id="env_inl",
        input=21,
    )
    await run_workflow_step(inline.id)

    async with wf_pool.acquire() as conn:
        reg_done = await wf_queries.get_run_for_step(conn, registered.id)
        inl_done = await wf_queries.get_run_for_step(conn, inline.id)
    assert reg_done is not None and inl_done is not None
    # Identical execution: same snapshotted script, same sha, same terminal status,
    # same output. The ONLY difference is the persisted definition (workflow_id +
    # source_version are set on the registered run, NULL on the inline run).
    assert inl_done.script == reg_done.script
    assert inl_done.script_sha == reg_done.script_sha
    assert inl_done.status == reg_done.status == "completed"
    assert inl_done.output == reg_done.output == {"doubled": 42}
    assert reg_done.workflow_id == wf.id and reg_done.source_version == 1
    assert inl_done.workflow_id is None and inl_done.source_version is None


# ─── surface clamp to the launcher ───────────────────────────────────────────


async def _agent_with_tools(pool: asyncpg.Pool[Any], tools: list[ToolSpec]) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id="acc_inl",
        name="surf-agent",
        model="test/dummy",
        system="surface test agent",
        tools=tools,
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    return agent.id


async def _session_for(pool: asyncpg.Pool[Any], agent_id: str) -> str:
    session = await sessions_service.create_session(
        pool,
        account_id="acc_inl",
        agent_id=agent_id,
        environment_id="env_inl",
        title=None,
        metadata={},
    )
    return session.id


async def test_inline_surface_clamped_to_launcher_ok(wf_pool: asyncpg.Pool[Any]) -> None:
    """An inline run whose surface is a subset of the launcher's is admitted and
    snapshots the declared surface."""
    agent_id = await _agent_with_tools(wf_pool, [ToolSpec(type="bash")])
    session_id = await _session_for(wf_pool, agent_id)
    script = "async def main(input):\n    await tool('bash')(command='echo hi')\n    return 'ok'\n"

    run = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        inline=InlineScript(script=script, tools=[ToolSpec(type="bash")]),
        environment_id="env_inl",
        launcher_session_id=session_id,
    )
    assert run.workflow_id is None
    assert [t.type for t in run.tools] == ["bash"]


async def test_inline_surface_exceeding_launcher_forbidden(wf_pool: asyncpg.Pool[Any]) -> None:
    """An inline run whose surface ⊄ the launcher's raises ForbiddenError — the same
    create-time clamp create_workflow uses."""
    agent_id = await _agent_with_tools(wf_pool, [])  # launcher holds no tools
    session_id = await _session_for(wf_pool, agent_id)
    script = "async def main(input):\n    await tool('bash')(command='echo hi')\n    return 'ok'\n"

    with pytest.raises(ForbiddenError):
        await service.create_run(
            wf_pool,
            account_id="acc_inl",
            inline=InlineScript(script=script, tools=[ToolSpec(type="bash")]),
            environment_id="env_inl",
            launcher_session_id=session_id,
        )


async def test_inline_operator_surface_verbatim(wf_pool: asyncpg.Pool[Any]) -> None:
    """The operator/HTTP path (no launcher) binds the inline surface verbatim — the
    lattice top, unattenuated."""
    script = "async def main(input):\n    await tool('bash')(command='echo hi')\n    return 'ok'\n"
    run = await service.create_run(
        wf_pool,
        account_id="acc_inl",
        inline=InlineScript(script=script, tools=[ToolSpec(type="bash")]),
        environment_id="env_inl",
    )
    assert [t.type for t in run.tools] == ["bash"]


# ─── exactly-one-arm enforcement ─────────────────────────────────────────────


async def test_neither_arm_is_validation_error(wf_pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError):
        await service.create_run(
            wf_pool, account_id="acc_inl", environment_id="env_inl"
        )


async def test_both_arms_is_validation_error(wf_pool: asyncpg.Pool[Any]) -> None:
    async with wf_pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(
            conn, account_id="acc_inl", name="both", script=_INLINE_SCRIPT
        )
    with pytest.raises(ValidationError):
        await service.create_run(
            wf_pool,
            account_id="acc_inl",
            workflow_id=wf.id,
            inline=InlineScript(script=_INLINE_SCRIPT),
            environment_id="env_inl",
        )


async def test_version_rejected_on_inline_arm(wf_pool: asyncpg.Pool[Any]) -> None:
    with pytest.raises(ValidationError):
        await service.create_run(
            wf_pool,
            account_id="acc_inl",
            inline=InlineScript(script=_INLINE_SCRIPT),
            environment_id="env_inl",
            version=2,
        )


async def test_inline_malformed_script_rejected(wf_pool: asyncpg.Pool[Any]) -> None:
    """A structurally invalid inline script fails as a clean ValidationError at launch
    (no missing-main run), the same gate create_workflow applies."""
    with pytest.raises(ValidationError):
        await service.create_run(
            wf_pool,
            account_id="acc_inl",
            inline=InlineScript(script="def not_main():\n    return 1\n"),
            environment_id="env_inl",
        )
