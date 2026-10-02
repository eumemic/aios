"""Concurrency tests for ``POST /v1/connectors/signal/unregister`` (#2322, F2).

Property: the irreversible ``unregister`` is never dispatched while any
non-archived signal connection the caller owns for the number (digits-only
match) has an active binding, and this holds UNDER CONCURRENCY.  When an
unregister races a binding-creating path, exactly one of them is refused.

"Dispatched" is observed the way the connector observes it: a committed
``pending_management_calls`` row.  A simulated connector polls for the row,
snapshots the number's live bindings the moment it sees it, then resolves
the call so ``submit_call`` returns.  Everything else is real: real pool,
real ``attach_connection`` / ``configure_per_chat``, real ``submit_call``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import asyncpg
import pytest

from aios.api.routers.connectors import SignalUnregisterRequest, post_signal_unregister
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import connections as connections_service
from aios.services import session_templates as session_templates_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

PHONE = "+16575274288"
DIGITS = "16575274288"
ACCOUNT = "acc_owner"

# Unwrapped reference: the simulated connector's snapshot must not count as
# one of the route's guard reads that the race tests wrap.
_REAL_LIST_ATTACHED = queries.list_attached_connections_for_phone


@pytest.fixture
async def detached_signal(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str]]:
    """Yield ``(pool, connection_id, session_id, template_id)`` — an UNBOUND
    signal connection for ``PHONE`` owned by ``acc_owner``, plus a live
    session and session template it could be bound to."""
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root',  NULL,       TRUE,  'root'),
                       ('acc_owner', 'acc_root', FALSE, 'owner'),
                       ('acc_other', 'acc_root', FALSE, 'other')
                """
            )
        agent, env, session = await seed_agent_env_session(
            pool, account_id=ACCOUNT, prefix="sig-race"
        )
        template = await session_templates_service.create_session_template(
            pool,
            account_id=ACCOUNT,
            name="sig-race-tpl",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                connector="signal",
                external_account_id=PHONE,
                metadata={},
                account_id=ACCOUNT,
            )
        yield pool, connection.id, session.id, template.id
    finally:
        await pool.close()


class _FakeConnector:
    """Polls for pending ``unregister`` rows (= dispatch), snapshots the
    number's live bindings at that instant, and resolves the call."""

    def __init__(self, pool: asyncpg.Pool[Any], *, resolve: bool = True) -> None:
        self.pool = pool
        self.resolve = resolve
        self.dispatched: list[str] = []
        self.live_at_dispatch: list[list[str]] = []
        self.seen = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id FROM pending_management_calls "
                    "WHERE method = 'unregister' AND status = 'pending'"
                )
                for row in rows:
                    if row["id"] in self.dispatched:
                        continue
                    live = await _REAL_LIST_ATTACHED(conn, "signal", DIGITS, account_id=ACCOUNT)
                    self.dispatched.append(row["id"])
                    self.live_at_dispatch.append([c.id for c in live])
                    self.seen.set()
                    if self.resolve:
                        await queries.mark_management_call_resolved(
                            conn,
                            account_id=ACCOUNT,
                            call_id=row["id"],
                            result={"external_account_id": PHONE},
                            is_error=False,
                        )
                        await queries.notify_management_call_result(conn, call_id=row["id"])
            await asyncio.sleep(0.02)

    async def __aenter__(self) -> _FakeConnector:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


async def _unregister(pool: asyncpg.Pool[Any], db_url: str) -> BaseException | None:
    try:
        await post_signal_unregister(
            SignalUnregisterRequest(external_account_id=PHONE), db_url, pool, ACCOUNT
        )
    except ConflictError as exc:
        return exc
    return None


async def _outcome(aw: Awaitable[Any]) -> BaseException | None:
    try:
        await aw
    except ConflictError as exc:
        return exc
    return None


def _wrap_attached_reads(
    monkeypatch: pytest.MonkeyPatch,
    after_read: Callable[[int], Awaitable[None]],
) -> None:
    """Wrap the real ``list_attached_connections_for_phone`` so ``after_read``
    runs after the N-th read returns (1-based), before its caller proceeds."""
    real = _REAL_LIST_ATTACHED
    calls = 0

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = await real(*args, **kwargs)
        calls += 1
        await after_read(calls)
        return result

    monkeypatch.setattr(queries, "list_attached_connections_for_phone", wrapped)


async def test_attach_committing_after_guard_read_is_never_unregistered(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's probe: the guard reads "nothing attached", then a real
    ``attach_connection`` commits, then unregister proceeds.  Must not
    dispatch with a live binding: the unregister is refused instead."""
    pool, connection_id, session_id, _ = detached_signal
    attach_result: list[BaseException | None] = []

    async def after_read(n: int) -> None:
        if n == 1:
            attach_result.append(
                await _outcome(
                    connections_service.attach_connection(
                        pool, connection_id, account_id=ACCOUNT, session_id=session_id
                    )
                )
            )

    _wrap_attached_reads(monkeypatch, after_read)
    async with _FakeConnector(pool) as connector:
        unregister_error = await _unregister(pool, migrated_db_url)
        await asyncio.sleep(0.2)  # give the connector a chance to see any row

    assert attach_result == [None], "the racing attach itself must succeed"
    live_dispatches = [live for live in connector.live_at_dispatch if live]
    assert not live_dispatches, f"unregister dispatched with live bindings {live_dispatches}"
    assert isinstance(unregister_error, ConflictError)
    assert unregister_error.detail["connection_ids"] == [connection_id]
    assert connector.dispatched == []


async def test_attach_while_unregister_in_flight_is_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """Reverse order: the unregister call is dispatched and still pending
    (connector has not answered).  Attach and configure_per_chat on the
    number are refused, so no binding appears under the unregister."""
    pool, connection_id, session_id, template_id = detached_signal
    async with _FakeConnector(pool, resolve=False) as connector:
        unregister = asyncio.create_task(_unregister(pool, migrated_db_url))
        await asyncio.wait_for(connector.seen.wait(), timeout=5)

        attach_error = await _outcome(
            connections_service.attach_connection(
                pool, connection_id, account_id=ACCOUNT, session_id=session_id
            )
        )
        per_chat_error = await _outcome(
            connections_service.configure_per_chat(
                pool, connection_id, account_id=ACCOUNT, session_template_id=template_id
            )
        )
        unregister.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await unregister

    assert connector.live_at_dispatch == [[]]
    assert isinstance(attach_error, ConflictError)
    assert attach_error.detail["reason"] == "number_unregister_pending"
    assert isinstance(per_chat_error, ConflictError)
    assert per_chat_error.detail["reason"] == "number_unregister_pending"
    async with pool.acquire() as conn:
        assert await queries.get_active_binding(conn, connection_id, account_id=ACCOUNT) is None


async def test_attach_racing_inside_unregister_critical_section_blocks_then_is_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverse order, tightest window: an attach starts AFTER unregister's
    authoritative check read and BEFORE its call row commits.  It must not
    commit in between (it blocks on the number lock), and once the call row
    is committed it is refused.  Exactly one side wins: the unregister."""
    pool, connection_id, session_id, _ = detached_signal
    attach_tasks: list[asyncio.Task[BaseException | None]] = []
    finished_inside_window: list[bool] = []

    async def after_read(n: int) -> None:
        # n == 1 is the route's fast-path read; n == 2 is the authoritative
        # read inside submit_call's transaction, under the number lock.
        if n == 2:
            task = asyncio.create_task(
                _outcome(
                    connections_service.attach_connection(
                        pool, connection_id, account_id=ACCOUNT, session_id=session_id
                    )
                )
            )
            attach_tasks.append(task)
            done, _ = await asyncio.wait({task}, timeout=0.5)
            finished_inside_window.append(bool(done))

    _wrap_attached_reads(monkeypatch, after_read)
    async with _FakeConnector(pool) as connector:
        unregister_error = await _unregister(pool, migrated_db_url)
        assert len(attach_tasks) == 1
        attach_error = await asyncio.wait_for(attach_tasks[0], timeout=5)

    live_dispatches = [live for live in connector.live_at_dispatch if live]
    assert not live_dispatches, f"unregister dispatched with live bindings {live_dispatches}"
    assert finished_inside_window == [False], "attach must block on the number lock"
    assert unregister_error is None
    assert len(connector.dispatched) == 1
    assert isinstance(attach_error, ConflictError)
    assert attach_error.detail["reason"] == "number_unregister_pending"


async def test_reparent_bound_connection_in_while_unregister_in_flight_is_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """Reparent carries an ACTIVE binding into the destination account, so it
    is a binding-creating path for the destination's number.  While
    ``acc_owner``'s unregister for the number is pending, moving another
    tenant's bound connection for the same number into ``acc_owner`` is
    refused."""
    pool, owner_connection_id, _, _ = detached_signal
    _a, _e, other_session = await seed_agent_env_session(
        pool, account_id="acc_other", prefix="sig-race-other"
    )
    async with pool.acquire() as conn:
        # Archive the owner's own row so the per-account partial-unique index
        # cannot be what refuses the move; only the number lock may.
        await queries.archive_connection(conn, owner_connection_id, account_id=ACCOUNT)
        other = await queries.insert_connection(
            conn,
            connector="signal",
            external_account_id="1 (657) 527-4288",
            metadata={},
            account_id="acc_other",
        )
    await connections_service.attach_connection(
        pool, other.id, account_id="acc_other", session_id=other_session.id
    )
    async with _FakeConnector(pool, resolve=False) as connector:
        unregister = asyncio.create_task(_unregister(pool, migrated_db_url))
        await asyncio.wait_for(connector.seen.wait(), timeout=5)
        reparent_error = await _outcome(
            connections_service.reparent_connection(
                pool,
                other.id,
                destination_account_id=ACCOUNT,
                requester_account_id="acc_root",
                crypto_box=CryptoBox(os.urandom(32)),
            )
        )
        unregister.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await unregister

    assert connector.live_at_dispatch == [[]]
    assert isinstance(reparent_error, ConflictError)
    assert reparent_error.detail["reason"] == "number_unregister_pending"
    async with pool.acquire() as conn:
        assert (
            await queries.list_attached_connections_for_phone(
                conn, "signal", DIGITS, account_id=ACCOUNT
            )
            == []
        )
