"""Integration tests for the session-side cancel leaf + the C2 sweep clause (6e).

A cancel-marked session must (a) be selected by ``find_sessions_needing_inference`` even when
idle (so the sweep wakes it), and (b) when its step runs the leaf, answer each marked request
``cancelled`` and harvest the marker (so it does not hot-loop). Owned-session teardown +
recursive propagation are the deferred §4.1/§4.4 residuals — out of scope for this slice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.harness.sweep import find_sessions_needing_inference
from aios.harness.task_registry import TaskRegistry
from aios.services import invocations as invocations_service
from aios.services import sessions as service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_cancel_leaf"


@pytest.fixture(autouse=True)
def _mock_prompt_wakes() -> Iterator[None]:
    """``cancel_invocation`` / ``create_run`` fire prompt procrastinate wakes; there is no open
    App here, so mock them at their bound call sites. The durable seed (marker / signal +
    tombstone) and the C2 sweep — not these best-effort prompts — are what drive the cancel, and
    those are what we assert. ``defer_wake`` is late-imported (patch the source); the
    ``defer_run_wake`` bindings live in the modules that import them at load time."""
    with (
        patch("aios.services.wake.defer_wake", new=AsyncMock()),
        patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
        patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
    ):
        yield


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'cancel-leaf')",
                _ACCOUNT,
            )
        _agent, env, session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="cancel-leaf"
        )
        yield pool, session.id, env.id
    finally:
        await pool.close()


async def _open_request(
    pool: asyncpg.Pool[Any], session_id: str, env_id: str, *, request_id: str
) -> None:
    """Open an awaited api-caller request edge on the session (no caller wake needed)."""
    async with pool.acquire() as conn:
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "api", "id": _ACCOUNT},
            depth=0,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=True,
        )


async def test_cancel_marked_session_is_swept_then_leaf_answers_cancelled(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_c")
    async with pool.acquire() as conn:
        await queries.insert_session_cancel_marker(
            conn, session_id=session_id, request_id="req_c", account_id=_ACCOUNT
        )

    # C2: the marked session is selected even though it is otherwise idle (no unreacted msgs).
    needs = await find_sessions_needing_inference(pool, TaskRegistry(), session_id=session_id)
    assert session_id in needs

    # The leaf answers the request cancelled + harvests the marker.
    assert await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)

    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, session_id, account_id=_ACCOUNT, request_id="req_c"
        )
        assert resolved == {"result": None, "is_error": True, "error": {"kind": "cancelled"}}
        # The request is closed (answered), and the marker is harvested → no re-wake.
        assert await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT) == []
        marker = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_c"
        )
        assert marker is not None and marker.harvested_at is not None

    # Idempotent: with the marker harvested, a second leaf run is a no-op (no hot-loop).
    assert not await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    needs_again = await find_sessions_needing_inference(pool, TaskRegistry(), session_id=session_id)
    assert session_id not in needs_again


async def test_unmarked_session_runs_no_leaf(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A session with an open request but NO cancel-marker is untouched by the leaf."""
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_live")
    assert not await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        assert await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT) == [
            "req_live"
        ]


async def test_cancel_invocation_session_seeds_tombstone_and_marker_end_to_end(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``cancel_invocation`` on a session servicer writes the tombstone + the exit-marker; the
    session's own leaf then answers the request ``cancelled`` — the full operator cancel path."""
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_x")

    await invocations_service.cancel_invocation(
        pool,
        servicer_kind="session",
        servicer_id=session_id,
        request_id="req_x",
        account_id=_ACCOUNT,
    )
    async with pool.acquire() as conn:
        intent = await queries.get_cancel_intent(
            conn, servicer_kind="session", servicer_id=session_id, request_id="req_x"
        )
        assert intent is not None  # durable intent tombstone
        marker = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_x"
        )
        assert marker is not None and marker.harvested_at is None

    # The servicer's own leaf (woken by the cancel) answers the request cancelled.
    assert await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, session_id, account_id=_ACCOUNT, request_id="req_x"
        )
        assert resolved == {"result": None, "is_error": True, "error": {"kind": "cancelled"}}

    # Idempotent: re-cancelling the same edge is a no-op (ON CONFLICT).
    await invocations_service.cancel_invocation(
        pool,
        servicer_kind="session",
        servicer_id=session_id,
        request_id="req_x",
        account_id=_ACCOUNT,
    )


async def test_cancel_invocation_cross_tenant_404(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A foreign account cannot cancel another tenant's session invocation."""
    pool, session_id, _env = pool_and_session
    with pytest.raises(NotFoundError):
        await invocations_service.cancel_invocation(
            pool,
            servicer_kind="session",
            servicer_id=session_id,
            request_id="r",
            account_id="acc_other",
        )


async def test_cancel_invocation_run_seeds_signal_and_tombstone(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``cancel_invocation`` on a RUN servicer reuses ``cancel_run`` (seeds the cancel signal +
    wakes) and writes the tombstone; the run's own harvest finalizes ``cancelled``."""
    from aios.db.queries import workflows as wf_queries
    from aios.services import workflows as wf_service

    pool, _session_id, env_id = pool_and_session
    wf = await wf_service.create_workflow(
        pool,
        account_id=_ACCOUNT,
        name="cancel-run-wf",
        script="async def main(input):\n    return 1\n",
        description=None,
        tools=[],
    )
    run = await wf_service.create_run(
        pool, account_id=_ACCOUNT, workflow_id=wf.id, environment_id=env_id, input=None
    )

    await invocations_service.cancel_invocation(
        pool, servicer_kind="run", servicer_id=run.id, request_id="req_run", account_id=_ACCOUNT
    )
    async with pool.acquire() as conn:
        signals = await wf_queries.list_run_signals(conn, run.id)
        assert any(s.kind == "cancel" for s in signals)  # cancel_run seeded the signal
        intent = await queries.get_cancel_intent(
            conn, servicer_kind="run", servicer_id=run.id, request_id="req_run"
        )
        assert intent is not None  # durable intent tombstone
