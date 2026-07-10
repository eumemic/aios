"""#1152 stage 1 — C1 caller-terminal cancel-cascade seeding, end to end on Postgres.

Drives REAL run terminals through ``run_workflow_step`` and asserts the durable DB
state the seeding path leaves behind (the markers/signals ARE the surface under
test — no mocks of the queries):

* §6(g) — a Tell (unawaited-edge) child of a terminal run is NEVER seeded a cancel
  marker nor archived by the seeding path: the ``awaited`` bit IS the ownership bit;
* §6(i) — FK-only children (``parent_run_id``/launcher FK, no request edge — the
  trigger-fired / ``create_run`` / WaM model-dispatch shapes) get NO cancel
  signal/marker: the trace-display ``awaited=True`` default is never load-bearing
  for a kill;
* the positive control — an open awaited Ask child IS seeded in the terminal's own
  transaction and promptly woken (``defer_wake(cause="cancel")``, the restored
  session-child prompt wake);
* §6(x) event-path parity — an ``engine_semantics_changed`` terminal closes the
  children's edges BEFORE the open-edge seeder runs, so
  ``_fail_child_requests_for_terminal_error`` must seed the markers itself; the
  child's own leaf then classifies the closed edge as REVOKED (kind ∈
  ``REVOCATION_KINDS``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import pytest

from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.models.attenuation import Surface
from aios.models.sessions import Err
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services.sessions import AskNewSession, TellNewSession
from aios.workflows import run_tools, service
from aios.workflows.child_id import child_session_id
from aios.workflows.step import run_workflow_step

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_cascade_seed"
_ENV = "env_cascade_seed"


@pytest.fixture
async def cascade_runtime(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], AsyncMock]]:
    """A pool installed on ``runtime.pool`` + a seeded tenant; every procrastinate
    deferral patched out. Yields the pool and the STEP module's ``defer_wake`` mock
    so tests can assert the post-commit session-child prompt wake."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'cascade-seed')",
                _ACCOUNT,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'cascade-env', '{}'::jsonb, $2)",
                _ENV,
                _ACCOUNT,
            )
        run_tools._INFLIGHT.clear()
        step_wake = AsyncMock()
        with (
            mock.patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.workflows.step.defer_wake", new=step_wake),
            mock.patch("aios.workflows.run_tools.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
            mock.patch("aios.services.sessions.defer_wake", new=AsyncMock()),
        ):
            yield pool, step_wake
    finally:
        run_tools._INFLIGHT.clear()
        runtime.pool = prev
        await pool.close()


async def _make_run(pool: asyncpg.Pool[Any], script: str, *, name: str = "w") -> str:
    async with pool.acquire() as conn:
        wf = await wf_queries.insert_workflow(conn, account_id=_ACCOUNT, name=name, script=script)
    run = await service.create_run(
        pool, account_id=_ACCOUNT, workflow_id=wf.id, environment_id=_ENV, input=None
    )
    return run.id


@pytest.fixture
async def child_agent_id(cascade_runtime: tuple[asyncpg.Pool[Any], AsyncMock]) -> str:
    pool, _ = cascade_runtime
    agent = await agents_service.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="cascade-child-agent",
        model="test/dummy",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    return agent.id


async def _spawn_edge_child(
    pool: asyncpg.Pool[Any],
    agent_id: str,
    run_id: str,
    *,
    request_id: str,
    awaited: bool,
) -> str:
    """Spawn a real Ask/Tell child of ``run_id`` through the public NewSession writer
    (which stamps ``parent_run_id`` + ``archive_when_idle=TRUE`` for BOTH arms and
    writes the ``request_opened`` edge with the given ``awaited`` bit)."""
    cid = child_session_id(run_id, request_id)
    stim: AskNewSession | TellNewSession
    if awaited:
        stim = AskNewSession(
            session_id=cid,
            agent_id=agent_id,
            environment_id=_ENV,
            agent_version=1,
            model=None,
            parent_run_id=run_id,
            surface=Surface([], [], []),
            vault_ids=[],
            request_id=request_id,
            input="hi",
        )
    else:
        stim = TellNewSession(
            session_id=cid,
            agent_id=agent_id,
            environment_id=_ENV,
            agent_version=1,
            model=None,
            parent_run_id=run_id,
            surface=Surface([], [], []),
            vault_ids=[],
            request_id=request_id,
            input="hi",
        )
    await sessions_service.create_child_session(pool, stim, account_id=_ACCOUNT)
    return cid


async def _run_status(pool: asyncpg.Pool[Any], run_id: str) -> str | None:
    async with pool.acquire() as conn:
        status = await conn.fetchval("SELECT status FROM wf_runs WHERE id = $1", run_id)
    return str(status) if status is not None else None


async def test_tell_child_not_seeded_ask_child_seeded_and_woken(
    cascade_runtime: tuple[asyncpg.Pool[Any], AsyncMock], child_agent_id: str
) -> None:
    """§6(g) + positive control: a terminal parent seeds ONLY the open awaited edge —
    the Tell child gets no marker and stays unarchived; the Ask child's marker commits
    with the terminal and the child is promptly woken with ``cause='cancel'``."""
    pool, step_wake = cascade_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    tell_id = await _spawn_edge_child(
        pool, child_agent_id, run_id, request_id="tell1", awaited=False
    )
    ask_id = await _spawn_edge_child(pool, child_agent_id, run_id, request_id="ask1", awaited=True)

    await run_workflow_step(run_id)
    assert await _run_status(pool, run_id) == "completed"

    async with pool.acquire() as conn:
        # the Ask child was seeded in the terminal transaction, unharvested
        ask_marker = await db_queries.get_session_cancel_marker(
            conn, session_id=ask_id, request_id="ask1"
        )
        assert ask_marker is not None and ask_marker.harvested_at is None
        # the Tell child: no marker at all, and NOT archived by the seeding path
        assert (
            await db_queries.get_session_cancel_marker(conn, session_id=tell_id, request_id="tell1")
            is None
        )
        tell_row = await conn.fetchrow("SELECT archived_at FROM sessions WHERE id = $1", tell_id)
        assert tell_row is not None and tell_row["archived_at"] is None

    # the restored session-child prompt wake: the Ask child only
    woken = {c.args[1] for c in step_wake.await_args_list}
    assert ask_id in woken
    assert tell_id not in woken
    for c in step_wake.await_args_list:
        if c.args[1] == ask_id:
            assert c.kwargs["cause"] == "cancel"


async def test_fk_only_children_get_no_cancel_signal(
    cascade_runtime: tuple[asyncpg.Pool[Any], AsyncMock], child_agent_id: str
) -> None:
    """§6(i): FK-only children — a run child with a bare ``parent_run_id`` and a
    session child with a bare ``parent_run_id`` (no request edge) — get no cancel
    signal/marker from a terminal parent: ``children_of``'s ``awaited=True`` display
    default is never load-bearing for a kill."""
    pool, step_wake = cascade_runtime
    parent_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    # FK-only RUN child (the create_run / trigger-fired / WaM shape: FK, no edge).
    fk_run_id = await _make_run(pool, "async def main(input):\n    return 2\n", name="w-child")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE wf_runs SET parent_run_id = $1 WHERE id = $2", parent_id, fk_run_id
        )
        # FK-only SESSION child (edgeless pre-#1123 legacy shape).
        fk_session = await db_queries.insert_session(
            conn,
            account_id=_ACCOUNT,
            agent_id=child_agent_id,
            environment_id=_ENV,
            agent_version=1,
            title=None,
            metadata={},
        )
        await conn.execute(
            "UPDATE sessions SET parent_run_id = $1 WHERE id = $2", parent_id, fk_session.id
        )

    await run_workflow_step(parent_id)
    assert await _run_status(pool, parent_id) == "completed"

    async with pool.acquire() as conn:
        # no cancel signal on the FK-only run child; it is untouched (still pending)
        signals = await wf_queries.list_run_signals(conn, fk_run_id)
        assert not any(s.kind == "cancel" for s in signals)
        assert (
            await conn.fetchval("SELECT status FROM wf_runs WHERE id = $1", fk_run_id) == "pending"
        )
        # no marker rows at all for the FK-only session child, and not archived
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM session_cancel_markers WHERE session_id = $1",
                fk_session.id,
            )
            == 0
        )
        assert (
            await conn.fetchval("SELECT archived_at FROM sessions WHERE id = $1", fk_session.id)
            is None
        )
    step_wake.assert_not_awaited()  # nothing was seeded → nothing to wake


async def test_engine_semantics_terminal_seeds_marker_in_same_transaction(
    cascade_runtime: tuple[asyncpg.Pool[Any], AsyncMock], child_agent_id: str
) -> None:
    """§6(x) event-path parity: an ``engine_semantics_changed`` terminal closes the
    child's edge BEFORE the open-edge seeder runs, so the closing path itself must
    seed the marker (same transaction) and the child is promptly woken; the child's
    leaf then classifies the closed edge REVOKED (kind ∈ REVOCATION_KINDS) and
    harvests."""
    pool, step_wake = cascade_runtime
    run_id = await _make_run(pool, "async def main(input):\n    return 1\n")
    child_id = await _spawn_edge_child(
        pool, child_agent_id, run_id, request_id="ask_es", awaited=True
    )
    async with pool.acquire() as conn:
        # age the run's engine epoch so the next wake takes the terminal-error path
        await conn.execute(
            "UPDATE wf_runs SET host_semantics_epoch = host_semantics_epoch - 1 WHERE id = $1",
            run_id,
        )

    await run_workflow_step(run_id)
    assert await _run_status(pool, run_id) == "errored"

    async with pool.acquire() as conn:
        # the child's edge was closed with the terminal kind…
        resolved = await db_queries.derive_response(
            conn, child_id, account_id=_ACCOUNT, request_id="ask_es"
        )
        assert isinstance(resolved, Err)
        assert resolved.error.get("kind") == "engine_semantics_changed"
        # …AND the cancel marker was seeded in the same transaction (not left to
        # the sweep): the open-edge seeder can no longer see this child.
        marker = await db_queries.get_session_cancel_marker(
            conn, session_id=child_id, request_id="ask_es"
        )
        assert marker is not None and marker.harvested_at is None
    # the prompt wake reached the marked child
    assert child_id in {c.args[1] for c in step_wake.await_args_list}

    # The child's Phase A classifies the already-closed edge as REVOKED
    # (engine_semantics_changed ∈ REVOCATION_KINDS). The marker remains durable
    # until the step-final Phase B archives the owned child and consumes it.
    decision = await sessions_service.harvest_session_cancel_markers(
        pool, child_id, account_id=_ACCOUNT
    )
    assert decision is not None and decision.teardown and decision.request_ids == ("ask_es",)
    async with pool.acquire() as conn:
        marker = await db_queries.get_session_cancel_marker(
            conn, session_id=child_id, request_id="ask_es"
        )
        assert marker is not None and marker.harvested_at is None
    assert await sessions_service.finalize_session_cancel_markers(
        pool, child_id, account_id=_ACCOUNT, teardown=True, request_ids=decision.request_ids
    )
    async with pool.acquire() as conn:
        marker = await db_queries.get_session_cancel_marker(
            conn, session_id=child_id, request_id="ask_es"
        )
        assert marker is not None and marker.harvested_at is not None
