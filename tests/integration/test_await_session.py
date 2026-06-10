"""Integration tests for ``await_session`` — the session backing of the await primitive.

Two monotonic modes over one endpoint:

* **Mode 1 (request_id)**: correlate a posted request to its response via
  ``derive_response`` — ``done`` once a response (or a ``child_gone`` outcome) lands.
* **Mode 2 (watermark)**: block until ``last_reacted_seq >= watermark`` (watermark
  defaults to ``last_stimulus_seq`` captured at call time).

DB-backed (testcontainer Postgres). Exercises the service (scope-check, LISTEN
subscribe, predicate, response build) end to end against real session rows; the pure
loop behavior is covered in ``tests/unit/test_await_session_predicates.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError, ValidationError
from aios.services import sessions as service
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


# ─── Mode 1: request_id correlation ──────────────────────────────────────────


async def test_mode1_already_responded_done_with_result(
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

    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="req1",
        watermark=None,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.result == {"answer": 7}
    assert resp.is_error is False
    assert resp.error is None


async def test_mode1_pending_times_out_done_false(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="reqX",
        watermark=None,
        timeout_seconds=0.1,
    )
    assert resp.done is False
    assert resp.result is None
    assert resp.is_error is False


async def test_mode1_child_gone_is_error(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    async with pool.acquire() as conn:
        await queries.archive_session(conn, session_id, account_id=account_id)

    resp = await service.await_session(
        pool,
        migrated_db_url,
        session_id,
        account_id=account_id,
        request_id="req_gone",
        watermark=None,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.is_error is True
    assert resp.error == {"kind": "child_gone"}


async def test_mode1_wakes_on_response_during_wait(
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
        service.await_session(
            pool,
            migrated_db_url,
            session_id,
            account_id=account_id,
            request_id="req_race",
            watermark=None,
            timeout_seconds=10,
        ),
        write_late(),
    )
    assert resp.done is True
    assert resp.result == "hi"


# ─── Mode 2: watermark ───────────────────────────────────────────────────────


async def test_mode2_reacted_at_or_above_watermark_done(
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
        request_id=None,
        watermark=1,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.last_reacted_seq >= 1


async def test_mode2_default_watermark_blocks(
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
        request_id=None,
        watermark=None,
        timeout_seconds=0.1,
    )
    assert resp.done is False
    assert resp.last_reacted_seq == 0


async def test_mode2_explicit_watermark_below_reacted_immediate(
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
        request_id=None,
        watermark=1,
        timeout_seconds=5,
    )
    assert resp.done is True
    assert resp.last_reacted_seq >= 2


# ─── validation + scoping ────────────────────────────────────────────────────


async def test_both_params_rejected(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, account_id, session_id = pool_and_session
    with pytest.raises(ValidationError):
        await service.await_session(
            pool,
            migrated_db_url,
            session_id,
            account_id=account_id,
            request_id="r",
            watermark=1,
            timeout_seconds=1,
        )


async def test_cross_tenant_404_before_subscribe(
    pool_and_session: tuple[asyncpg.Pool[Any], str, str],
    migrated_db_url: str,
) -> None:
    pool, _account_id, session_id = pool_and_session
    with pytest.raises(NotFoundError):
        await service.await_session(
            pool,
            migrated_db_url,
            session_id,
            account_id="acc_other",
            request_id="x",
            watermark=None,
            timeout_seconds=1,
        )
