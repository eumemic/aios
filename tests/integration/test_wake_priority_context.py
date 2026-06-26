"""Integration tests for ``get_wake_priority_context`` (#1125).

The DB-backed half of the wake-priority re-key: ``defer_wake`` derives its
foreground/background demotion per-stimulus from the **triggering edge's
up-link** (#1123's ``request_opened`` ``caller``) rather than the run-only
``parent_run_id`` column, so every caller kind (api/session/run) demotes
uniformly when its ancestor is background.

Exercises ``get_wake_priority_context`` against real session + event rows for
each caller kind; the unit tier (``tests/unit/test_wake_priority.py``) covers
the ``defer_wake`` priority wiring with the query mocked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_wake_priority"


@pytest.fixture
async def pool_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, account_id, agent_id, environment_id)`` for a fresh tenant."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'wake-priority-test')",
                _ACCOUNT,
            )
        agent, env, _session = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="wake-priority"
        )
        yield pool, _ACCOUNT, agent.id, env.id
    finally:
        await pool.close()


async def _open_edge(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    request_id: str,
    caller: dict[str, Any],
    awaited: bool = True,
) -> None:
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_request_opened(
            conn,
            session_id=session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            caller=caller,
            depth=0,
            environment_id="env_x",
            frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
            vault_ids=[],
            awaited=awaited,
        )


async def _insert_session(
    pool: asyncpg.Pool[Any], *, agent_id: str, environment_id: str, origin: str
) -> str:
    """Insert a bare session with a chosen ``origin``, return its id."""
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=_ACCOUNT,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=1,
            title=None,
            metadata={},
        )
        await conn.execute("UPDATE sessions SET origin = $2 WHERE id = $1", session.id, origin)
    return session.id


async def _close_edge(pool: asyncpg.Pool[Any], *, session_id: str, request_id: str) -> None:
    """Answer a request (``request_response``), so its edge is no longer open."""
    async with pool.acquire() as conn:
        wrote = await queries.write_response_if_absent(
            conn,
            session_id,
            account_id=_ACCOUNT,
            request_id=request_id,
            is_error=False,
            result={"ok": True},
            error=None,
        )
    assert wrote


async def test_no_edge_is_foreground(pool_env: tuple[asyncpg.Pool[Any], str, str, str]) -> None:
    """An ordinary root / fg-user session (no ``request_opened`` edge) → foreground."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="foreground")
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is False


async def test_run_caller_is_background(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A run-launched request-serving child (``caller.kind='run'``) → background
    (the legacy ``parent_run_id`` run path, behavior-preserved)."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_run", caller={"kind": "run", "id": "wfr_1"}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is True


async def test_session_caller_background_root_is_background(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A session-launched child whose caller session is itself background-rooted
    (``origin='background'``) → background (the new behavior #1125 unlocks)."""
    pool, _account, agent_id, env_id = pool_env
    caller_sid = await _insert_session(
        pool, agent_id=agent_id, environment_id=env_id, origin="background"
    )
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_s", caller={"kind": "session", "id": caller_sid}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is True


async def test_session_caller_foreground_root_is_foreground(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A session-launched child whose caller session is foreground-rooted stays
    foreground — a fg-user session-invoke must not demote."""
    pool, _account, agent_id, env_id = pool_env
    caller_sid = await _insert_session(
        pool, agent_id=agent_id, environment_id=env_id, origin="foreground"
    )
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_s2", caller={"kind": "session", "id": caller_sid}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is False


async def test_api_caller_is_foreground(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """An api-launched (direct user) request-serving session → foreground."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="foreground")
    await _open_edge(
        pool, session_id=sid, request_id="req_api", caller={"kind": "api", "id": "k_1"}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is False


async def test_missing_session_is_none(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """Deleted-session race: a vanished row → ``None`` (the foreground fall-through)."""
    pool, _account, _agent_id, _env_id = pool_env
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, "ses_does_not_exist")
    assert ctx is None


# --- Multi-edge per-stimulus correctness (#1125 oldest-vs-latest-open) --------
#
# A session that has served more than one request carries several
# ``request_opened`` edges. The priority must reflect the **most-recently-opened
# still-open** edge (the current stimulus), not the oldest-ever one. These cases
# are the reachable inversions once ``POST /v1/tasks target_kind=session``
# (#1128) appends a second edge to a live session; the single-edge cases above
# cannot distinguish oldest from latest.


async def test_latest_open_edge_wins_over_older_open(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A background-rooted child (run edge) later invoked at foreground (api edge)
    wakes **foreground** — the latest open edge is the api stimulus, not the
    oldest run edge. The starvation inversion #1125 exists to prevent."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_run_old", caller={"kind": "run", "id": "wfr_1"}
    )
    await _open_edge(
        pool, session_id=sid, request_id="req_api_new", caller={"kind": "api", "id": "k_1"}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is False


async def test_latest_open_edge_demotes_when_newer_is_background(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """The symmetric inversion: a foreground-first session (api edge) later driven
    by a background run edge wakes **background** — the latest open edge is the run
    stimulus, so background fan-out can't compete with user messages."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_api_old", caller={"kind": "api", "id": "k_1"}
    )
    await _open_edge(
        pool, session_id=sid, request_id="req_run_new", caller={"kind": "run", "id": "wfr_1"}
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is True


async def test_answered_edge_is_excluded(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """An answered (``request_response``-closed) edge no longer counts: a session
    whose only request has been answered falls back to the foreground default,
    matching the docstring's **still-open** edge contract."""
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool, session_id=sid, request_id="req_run_done", caller={"kind": "run", "id": "wfr_1"}
    )
    await _close_edge(pool, session_id=sid, request_id="req_run_done")
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is False


async def test_unawaited_tell_edge_still_demotes_background(
    pool_env: tuple[asyncpg.Pool[Any], str, str, str],
) -> None:
    """A background-rooted session whose only open edge is an **unawaited**
    ``Tell(NewSession)`` spawn (``awaited=false``, ``caller.kind='run'``) still
    wakes **background**.

    Pins the intentional ``awaited_only=False`` choice in
    :func:`get_wake_priority_context`: the wake-priority demotion keys off *any*
    triggering up-link — a ``Tell``-spawned child of a background run is a fan-out
    descendant and must yield to user-facing sessions — so this query's open set
    is deliberately a **superset** of :func:`get_open_request_ids`'s (which would
    exclude this unawaited edge). A future "tidy" that adds the ``awaited`` filter
    to this reader would break this case loudly.
    """
    pool, _account, agent_id, env_id = pool_env
    sid = await _insert_session(pool, agent_id=agent_id, environment_id=env_id, origin="background")
    await _open_edge(
        pool,
        session_id=sid,
        request_id="req_tell",
        caller={"kind": "run", "id": "wfr_tell"},
        awaited=False,
    )
    async with pool.acquire() as conn:
        ctx = await queries.get_wake_priority_context(conn, sid)
    assert ctx is not None and ctx[1] is True
