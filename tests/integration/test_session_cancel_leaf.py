"""Integration tests for the session-side cancel leaf, the C2 sweep clause, and
``cancel_task`` (6e/6f/6g).

A cancel-marked session must (a) be selected by ``find_sessions_needing_inference`` even when
idle (so the sweep wakes it), (b) when its step runs the leaf, answer each marked request
``cancelled`` and harvest the marker (no hot-loop), and (c) when it owes no remaining inbound,
propagate a cancel-marker to each awaited child (the §2.3 recursive hop — owned-only, so a
shared session never over-cancels a surviving request's work). ``cancel_task`` seeds it
for both servicer kinds. Owned-session teardown (interrupt + archive, §4/C1) + §7 attributed_to
+ the §9 counter are the deferred residuals.
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
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import find_sessions_needing_inference
from aios.models.sessions import Err, Ok
from aios.services import sessions as service
from aios.services import tasks as tasks_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_cancel_leaf"


@pytest.fixture(autouse=True)
def _mock_prompt_wakes() -> Iterator[None]:
    """The cancel leaf + ``cancel_task``'s session arm fire prompt procrastinate wakes
    (no open App here). Since #1476 the deferral primitives live in ``aios.jobs.app`` and are
    bound at module load into the caller namespaces (``services.sessions`` for the cancel leaf,
    ``services.tasks`` for the session arm), so patch them where they are looked up. The durable
    seed (marker / signal + tombstone) and the C2 sweep, not these best-effort prompts, are what
    drive the cancel."""
    with (
        patch("aios.services.sessions.defer_wake", new=AsyncMock()),
        patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
        patch("aios.services.tasks.defer_wake", new=AsyncMock()),
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
    needs = await find_sessions_needing_inference(
        pool, InflightToolRegistry(), session_id=session_id
    )
    assert session_id in needs

    # The leaf answers the request cancelled + harvests the marker.
    assert await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)

    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, session_id, account_id=_ACCOUNT, request_id="req_c"
        )
        assert resolved == Err(error={"kind": "cancelled"})
        # The request is closed (answered), and the marker is harvested → no re-wake.
        assert await queries.get_open_request_ids(conn, session_id, account_id=_ACCOUNT) == []
        marker = await queries.get_session_cancel_marker(
            conn, session_id=session_id, request_id="req_c"
        )
        assert marker is not None and marker.harvested_at is not None

    # Idempotent: with the marker harvested, a second leaf run is a no-op (no hot-loop).
    assert not await service.harvest_session_cancel_markers(pool, session_id, account_id=_ACCOUNT)
    needs_again = await find_sessions_needing_inference(
        pool, InflightToolRegistry(), session_id=session_id
    )
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


async def test_cancel_task_session_seeds_tombstone_and_marker_end_to_end(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``cancel_task`` on a session servicer writes the tombstone + the exit-marker; the
    session's own leaf then answers the request ``cancelled`` — the full operator cancel path."""
    pool, session_id, env_id = pool_and_session
    await _open_request(pool, session_id, env_id, request_id="req_x")

    await tasks_service.cancel_task(
        pool,
        servicer_kind="session",
        servicer_id=session_id,
        request_id="req_x",
        account_id=_ACCOUNT,
    )
    async with pool.acquire() as conn:
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
        assert resolved == Err(error={"kind": "cancelled"})

    # Idempotent: re-cancelling the same edge is a no-op (ON CONFLICT).
    await tasks_service.cancel_task(
        pool,
        servicer_kind="session",
        servicer_id=session_id,
        request_id="req_x",
        account_id=_ACCOUNT,
    )


async def test_cancel_task_cross_tenant_404(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A foreign account cannot cancel another tenant's session task."""
    pool, session_id, _env = pool_and_session
    with pytest.raises(NotFoundError):
        await tasks_service.cancel_task(
            pool,
            servicer_kind="session",
            servicer_id=session_id,
            request_id="r",
            account_id="acc_other",
        )


async def test_cancel_task_run_seeds_signal_and_tombstone(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """``cancel_task`` on a RUN servicer reuses ``cancel_run`` (seeds the cancel signal +
    wakes) and writes the tombstone; the run's own harvest finalizes ``cancelled``."""
    from aios.db.queries import workflows as wf_queries
    from aios.services import workflows as wf_service

    pool, _session_id, env_id = pool_and_session
    # ``create_run`` / ``cancel_run`` bind ``defer_run_wake`` at module load — patch those
    # bindings here (the modules are imported above), no open procrastinate App in this tier.
    with (
        patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
        patch("aios.services.workflows.defer_run_wake", new=AsyncMock()),
    ):
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
        await tasks_service.cancel_task(
            pool,
            servicer_kind="run",
            servicer_id=run.id,
            request_id="req_run",
            account_id=_ACCOUNT,
        )
    async with pool.acquire() as conn:
        signals = await wf_queries.list_run_signals(conn, run.id)
        assert any(s.kind == "cancel" for s in signals)  # cancel_run seeded the signal


async def _open_child_session_edge(
    pool: asyncpg.Pool[Any], child_id: str, parent_id: str, env_id: str, *, request_id: str
) -> None:
    """Open an awaited request edge on ``child_id`` whose caller is the parent session."""
    async with pool.acquire() as conn:
        await queries.append_request_opened(
            conn,
            session_id=child_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller={"kind": "session", "id": parent_id},
            depth=1,
            environment_id=env_id,
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=True,
        )


async def test_owned_session_cancel_propagates_to_awaited_children(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """§2.3: when the cancelled request was an OWNED session's SOLE inbound, the leaf seeds
    a cancel-marker on each outbound awaited child — the recursive cascade hop. Ownership
    here is flag-declared (``archive_when_idle=TRUE``, the servicer class every edge-created
    child carries); a bare inbound Ask no longer classifies a session as owned."""
    pool, parent_id, env_id = pool_and_session
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET archive_when_idle = TRUE WHERE id = $1", parent_id)
    await _open_request(pool, parent_id, env_id, request_id="req_root")  # the parent's sole inbound
    _a, _e, child = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix="cancel-leaf-child"
    )
    await _open_child_session_edge(pool, child.id, parent_id, env_id, request_id="req_child")
    async with pool.acquire() as conn:
        await queries.insert_session_cancel_marker(
            conn, session_id=parent_id, request_id="req_root", account_id=_ACCOUNT
        )

    # Harvesting the parent's marker answers req_root cancelled AND (owned) markers the child.
    assert await service.harvest_session_cancel_markers(pool, parent_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        child_marker = await queries.get_session_cancel_marker(
            conn, session_id=child.id, request_id="req_child"
        )
        assert child_marker is not None and child_marker.harvested_at is None  # cascade reached it


async def test_self_owned_session_never_propagates(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """FATAL-2 wrong-kill immunity: a SELF-OWNED session (no creation edge, no
    ``archive_when_idle`` flag — the lieutenant/root shape) never auto-propagates on
    obligation withdrawal, even when its sole inbound was revoked. Receiving an
    awaited Ask must NOT classify a session as owned (the any-edge join defect)."""
    pool, parent_id, env_id = pool_and_session
    await _open_request(pool, parent_id, env_id, request_id="req_root")  # sole inbound
    _a, _e, child = await seed_agent_env_session(pool, account_id=_ACCOUNT, prefix="cancel-leaf-lt")
    await _open_child_session_edge(pool, child.id, parent_id, env_id, request_id="req_child")
    async with pool.acquire() as conn:
        await queries.insert_session_cancel_marker(
            conn, session_id=parent_id, request_id="req_root", account_id=_ACCOUNT
        )

    # The leaf answers req_root cancelled (revocation-context) and harvests…
    assert await service.harvest_session_cancel_markers(pool, parent_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        resolved = await queries.derive_response(
            conn, parent_id, account_id=_ACCOUNT, request_id="req_root"
        )
        assert resolved == Err(error={"kind": "cancelled"})
        # …but the live subtree is untouched: not owned ⇒ zero propagation.
        assert (
            await queries.get_session_cancel_marker(
                conn, session_id=child.id, request_id="req_child"
            )
            is None
        )


async def test_fulfilled_marker_set_does_not_propagate(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """§1 revoked/fulfilled asymmetry: a marker whose request the SERVICER already
    answered (``Ok``) classifies FULFILLED — an owned session with a purely-fulfilled
    marker set and zero surviving inbound must NOT propagate cancellation."""
    pool, parent_id, env_id = pool_and_session
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET archive_when_idle = TRUE WHERE id = $1", parent_id
        )  # owned via the flag — isolates the revocation-context gate
    await _open_request(pool, parent_id, env_id, request_id="req_done")  # sole inbound
    _a, _e, child = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix="cancel-leaf-fulfilled"
    )
    await _open_child_session_edge(pool, child.id, parent_id, env_id, request_id="req_child")
    async with pool.acquire() as conn:
        # the servicer answers FIRST (first-writer-wins latches the Ok)…
        assert await queries.write_response_if_absent(
            conn, parent_id, account_id=_ACCOUNT, request_id="req_done", outcome=Ok(result="done")
        )
        # …then a stray marker lands on the already-fulfilled edge
        await queries.insert_session_cancel_marker(
            conn, session_id=parent_id, request_id="req_done", account_id=_ACCOUNT
        )

    # The leaf applies the marker (harvests it; the cancelled write no-ops)…
    assert await service.harvest_session_cancel_markers(pool, parent_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        marker = await queries.get_session_cancel_marker(
            conn, session_id=parent_id, request_id="req_done"
        )
        assert marker is not None and marker.harvested_at is not None
        resolved = await queries.derive_response(
            conn, parent_id, account_id=_ACCOUNT, request_id="req_done"
        )
        assert resolved == Ok(result="done")  # the servicer's answer survived
        # …and does NOT cascade: fulfilled ⇒ no revocation-context ⇒ zero propagation.
        assert (
            await queries.get_session_cancel_marker(
                conn, session_id=child.id, request_id="req_child"
            )
            is None
        )


async def test_shared_session_cancel_does_not_over_cancel_children(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A still-multiply-inbound session does NOT propagate on a single request's cancel —
    over-cancelling a surviving request's child would be unsound (waits for §7 attributed_to)."""
    pool, parent_id, env_id = pool_and_session
    await _open_request(pool, parent_id, env_id, request_id="req_a")
    await _open_request(pool, parent_id, env_id, request_id="req_b")  # a SECOND live inbound
    _a, _e, child = await seed_agent_env_session(
        pool, account_id=_ACCOUNT, prefix="cancel-leaf-shared"
    )
    await _open_child_session_edge(pool, child.id, parent_id, env_id, request_id="req_child")
    async with pool.acquire() as conn:
        await queries.insert_session_cancel_marker(
            conn, session_id=parent_id, request_id="req_a", account_id=_ACCOUNT
        )

    assert await service.harvest_session_cancel_markers(pool, parent_id, account_id=_ACCOUNT)
    async with pool.acquire() as conn:
        # req_a is answered cancelled, but req_b survives → the child is NOT marked.
        assert (
            await queries.get_session_cancel_marker(
                conn, session_id=child.id, request_id="req_child"
            )
            is None
        )
        assert await queries.get_open_request_ids(conn, parent_id, account_id=_ACCOUNT) == ["req_b"]
