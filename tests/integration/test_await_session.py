"""Integration tests for the session-backed awaits.

Two monotonic awaits over a session, now split across two services:

* **request correlation** is the session arm of the unified awaiter
  ``await_task`` — correlate a posted request to its response via
  ``derive_response``, resolving once a response (or a ``child_gone`` outcome) lands.
* **watermark quiescence** is ``await_session`` (the orthogonal session-only alias):
  block until ``last_reacted_seq >= watermark`` (defaulting to ``last_stimulus_seq``
  captured at call time).

DB-backed (testcontainer Postgres). Exercises the services (scope-check, LISTEN
subscribe, predicate, response build) end to end against real session rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.listen import open_listen_for_events
from aios.db.pool import create_pool
from aios.db.sse_lock import has_subscriber
from aios.errors import NotFoundError
from aios.models.tasks import AwaitResponse
from aios.services import sessions as service
from aios.services import tasks as tasks_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_and_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a fresh session."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_await_session', NULL, TRUE, 'await-session-test')
                """
            )
        _agent, _env, session = await seed_agent_env_session(
            pool, account_id="acc_await_session", prefix="await-session-test"
        )
        yield pool, "acc_await_session", session.id
    finally:
        await pool.close()


async def _await_request(
    pool: asyncpg.Pool[Any],
    db_url: str,
    session_id: str,
    *,
    account_id: str,
    request_id: str,
    timeout_seconds: float,
) -> AwaitResponse:
    """Drive the unified awaiter's session arm (request correlation)."""
    return await tasks_service.await_task(
        pool,
        db_url,
        servicer_kind="session",
        servicer_id=session_id,
        request_id=request_id,
        account_id=account_id,
        timeout_seconds=timeout_seconds,
    )


# ─── request correlation (await_task, session arm) ─────────────────────


async def test_request_already_responded_ok_with_result(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await queries.write_response_if_absent(
            conn,
            session_id,
            account_id=account_id,
            request_id="req1",
            is_error=False,
            result={"answer": 7},
            error=None,
        )

    resp = await _await_request(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="req1",
        timeout_seconds=5,
    )
    assert resp.outcome == "ok"
    assert resp.result == {"answer": 7}
    assert resp.error is None


async def test_request_pending_times_out_outcome_none(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    resp = await _await_request(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="reqX",
        timeout_seconds=0.1,
    )
    assert resp.outcome is None
    assert resp.result is None and resp.error is None


async def test_request_child_gone_is_errored(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await queries.archive_session(conn, session_id, account_id=account_id)

    resp = await _await_request(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="req_gone",
        timeout_seconds=5,
    )
    assert resp.outcome == "errored"
    assert resp.error == {"kind": "child_gone"}


async def test_request_wakes_on_response_during_wait(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session

    async def write_late() -> None:
        await asyncio.sleep(0.1)
        async with pool.acquire() as conn:
            await queries.write_response_if_absent(
                conn,
                session_id,
                account_id=account_id,
                request_id="req_race",
                is_error=False,
                result="hi",
                error=None,
            )

    resp, _ = await asyncio.gather(
        _await_request(
            pool,
            migrated_db_url,
            session_id,
            account_id=account_id,
            request_id="req_race",
            timeout_seconds=10,
        ),
        write_late(),
    )
    assert resp.outcome == "ok"
    assert resp.result == "hi"


async def test_request_cross_tenant_404_before_subscribe(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, _account_id, session_id = pool_and_session
    with pytest.raises(NotFoundError):
        await _await_request(
            pool,
            migrated_db_url,
            session_id,
            account_id="acc_other",
            request_id="x",
            timeout_seconds=1,
        )


# ─── watermark quiescence (await_session, the session-only alias) ─────────────


async def test_watermark_reacted_at_or_above_watermark_done(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # user (seq 1) → last_stimulus_seq = 1
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi"},
        )
        # assistant reacting_to=1 (seq 2) → last_reacted_seq = 1
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": "ok", "reacting_to": 1},
        )

    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        watermark=1,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.last_reacted_seq >= 1


async def test_watermark_default_watermark_blocks(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # only a user message: last_stimulus_seq=1 > last_reacted_seq=0
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "hi"},
        )

    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        watermark=None,
        timeout_seconds=0.1,
    )
    assert resp.done is False
    assert resp.last_reacted_seq == 0


async def test_watermark_explicit_below_reacted_immediate(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        # user (seq 1)
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "one"},
        )
        # assistant reacting_to=1 (seq 2) → last_reacted_seq = 1
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": "ack", "reacting_to": 1},
        )
        # user (seq 3)
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "user", "content": "two"},
        )
        # assistant reacting_to=3 (seq 4) → last_reacted_seq = 3 (>= 2)
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="message",
            data={"role": "assistant", "content": "ack2", "reacting_to": 3},
        )

    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        watermark=1,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.last_reacted_seq >= 2


# ─── subscriber lock: await poller must not toggle the streaming path ─────────


async def test_await_listen_does_not_acquire_subscriber_lock(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    """An await poller consumes terminal state only, so its LISTEN must NOT
    acquire the subscriber lock — otherwise has_subscriber() would force the
    awaited session's worker onto the streaming model path (issue #81).

    The lock lives on the listener's own dedicated connection; check
    has_subscriber via the pool (a separate connection).
    """
    pool, _account_id, session_id = pool_and_session

    # await-poller open: lock skipped → no subscriber observed.
    sub = await open_listen_for_events(migrated_db_url, session_id, on_connected=None)
    try:
        assert await has_subscriber(pool, session_id) is False
    finally:
        sub.terminate()

    # default open (SSE /stream, wait_for_events): lock held → subscriber observed.
    sub = await open_listen_for_events(migrated_db_url, session_id)
    try:
        assert await has_subscriber(pool, session_id) is True
    finally:
        sub.terminate()
