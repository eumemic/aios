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
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from aios.api.routers.connectors import (
    SignalUnregisterRequest,
    post_signal_unregister,
    post_signal_unregister_cancel,
)
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


async def test_attach_after_received_unregister_expires_is_still_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """F3: the connector has RECEIVED the unregister but not yet executed it
    (its management loop is serial; the call can sit behind a slow verify).
    The pending row's ``expires_at`` then passes.  The connector can still
    execute the call, so attach must STILL be refused: the guard clears only
    on a terminal status, never on wall-clock expiry."""
    pool, connection_id, session_id, _ = detached_signal
    async with _FakeConnector(pool, resolve=False) as connector:
        unregister = asyncio.create_task(_unregister(pool, migrated_db_url))
        await asyncio.wait_for(connector.seen.wait(), timeout=5)
        unregister.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await unregister
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE pending_management_calls SET expires_at = now() - interval '1 second' "
                "WHERE method = 'unregister'"
            )
        attach_after_expiry = await _outcome(
            connections_service.attach_connection(
                pool, connection_id, account_id=ACCOUNT, session_id=session_id
            )
        )

    assert isinstance(attach_after_expiry, ConflictError), (
        f"attach_after_expiry={attach_after_expiry!r}: a received unregister "
        "stopped guarding the number at wall-clock expiry"
    )
    assert attach_after_expiry.detail["reason"] == "number_unregister_pending"
    async with pool.acquire() as conn:
        assert await queries.get_active_binding(conn, connection_id, account_id=ACCOUNT) is None


async def _dispatch_and_hold_unregister(
    pool: asyncpg.Pool[Any], db_url: str, connector: _FakeConnector
) -> None:
    """Dispatch an unregister the connector receives but never resolves,
    then age its row past ``expires_at`` (the operator's 504 has fired)."""
    unregister = asyncio.create_task(_unregister(pool, db_url))
    await asyncio.wait_for(connector.seen.wait(), timeout=5)
    unregister.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await unregister
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pending_management_calls SET expires_at = now() - interval '1 second' "
            "WHERE method = 'unregister'"
        )


async def test_bind_chat_while_unregister_pending_is_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """#2322 N1: a bound chat routes inbound with no binding, so
    ``bind_chat_to_session`` is a make-it-routable path too and is refused
    under the same number lock while an unregister is non-terminal."""
    pool, connection_id, session_id, _ = detached_signal
    async with _FakeConnector(pool, resolve=False) as connector:
        await _dispatch_and_hold_unregister(pool, migrated_db_url, connector)
        bind_error = await _outcome(
            connections_service.bind_chat_to_session(
                pool,
                connection_id,
                account_id=ACCOUNT,
                chat_id="+15550001111",
                session_id=session_id,
            )
        )
    assert isinstance(bind_error, ConflictError)
    assert bind_error.detail["reason"] == "number_unregister_pending"
    async with pool.acquire() as conn:
        assert (
            await queries.list_chat_sessions_for_connection(conn, connection_id, account_id=ACCOUNT)
            == []
        )


async def test_expired_unregister_is_redelivered_to_a_restarted_connector(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """Connector crash / no-claim: the SSE backfill a (re)connecting
    connector reads still carries a non-terminal unregister past its
    ``expires_at``, so the connector drives it to a terminal status and the
    number does not stay un-bindable forever.  Other methods keep the old
    expiry filter."""
    pool, _, _, _ = detached_signal
    async with _FakeConnector(pool, resolve=False) as connector:
        await _dispatch_and_hold_unregister(pool, migrated_db_url, connector)
    async with pool.acquire() as conn:
        await queries.insert_management_call(
            conn,
            account_id=ACCOUNT,
            call_id="mgmt_expired_verify",
            connector="signal",
            method="verify",
            params={"external_account_id": PHONE, "code": "123456"},
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        backfill = await queries.list_pending_management_calls_for_connector(
            conn, "signal", account_id=ACCOUNT
        )
    assert [c["call_id"] for c in backfill] == connector.dispatched
    assert [c["method"] for c in backfill] == ["unregister"]


async def test_operator_cancel_terminalises_and_unblocks_attach(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
) -> None:
    """Escape hatch: ``POST /signal/unregister/cancel`` marks the pending
    unregister ``failed`` (cancelled_by_operator); attach then succeeds, the
    call is no longer redelivered, and a late connector result is ignored."""
    pool, connection_id, session_id, _ = detached_signal
    async with _FakeConnector(pool, resolve=False) as connector:
        await _dispatch_and_hold_unregister(pool, migrated_db_url, connector)
    (call_id,) = connector.dispatched

    # A variant spelling of the number cancels the same call.
    response = await post_signal_unregister_cancel(
        SignalUnregisterRequest(external_account_id="1 (657) 527-4288"), pool, ACCOUNT
    )
    assert response.cancelled_call_ids == [call_id]
    again = await post_signal_unregister_cancel(
        SignalUnregisterRequest(external_account_id=PHONE), pool, ACCOUNT
    )
    assert again.cancelled_call_ids == []

    assert (
        await _outcome(
            connections_service.attach_connection(
                pool, connection_id, account_id=ACCOUNT, session_id=session_id
            )
        )
        is None
    )
    async with pool.acquire() as conn:
        row = await queries.get_management_call(conn, call_id, account_id=ACCOUNT)
        assert row is not None
        assert row["status"] == "failed"
        assert row["result"]["code"] == "cancelled_by_operator"
        assert (
            await queries.list_pending_management_calls_for_connector(
                conn, "signal", account_id=ACCOUNT
            )
            == []
        )
        assert not await queries.mark_management_call_resolved(
            conn, account_id=ACCOUNT, call_id=call_id, result={}, is_error=False
        )


@pytest.mark.parametrize("aged_past_expiry", [False, True])
async def test_reparent_chat_bound_connection_in_while_unregister_pending_is_refused(
    detached_signal: tuple[asyncpg.Pool[Any], str, str, str],
    migrated_db_url: str,
    aged_past_expiry: bool,
) -> None:
    """#2322 F4: a connection with an operator-bound chat and NO binding is
    still "in use" (resolver tier 1 routes it).  Reparent moves its
    ``chat_sessions`` rows into the destination, so it must be refused under
    the number lock while the destination's unregister is non-terminal --
    whether or not the call row has passed ``expires_at``."""
    pool, owner_connection_id, _, _ = detached_signal
    _a, _e, other_session = await seed_agent_env_session(
        pool, account_id="acc_other", prefix="sig-race-chat-other"
    )
    async with pool.acquire() as conn:
        await queries.archive_connection(conn, owner_connection_id, account_id=ACCOUNT)
        other = await queries.insert_connection(
            conn,
            connector="signal",
            external_account_id="1 (657) 527-4288",
            metadata={},
            account_id="acc_other",
        )
    await connections_service.bind_chat_to_session(
        pool,
        other.id,
        account_id="acc_other",
        chat_id="+15550001111",
        session_id=other_session.id,
    )
    async with _FakeConnector(pool, resolve=False) as connector:
        unregister = asyncio.create_task(_unregister(pool, migrated_db_url))
        await asyncio.wait_for(connector.seen.wait(), timeout=5)
        unregister.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await unregister
        if aged_past_expiry:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE pending_management_calls "
                    "SET expires_at = now() - interval '1 second' "
                    "WHERE method = 'unregister'"
                )
        reparent_error = await _outcome(
            connections_service.reparent_connection(
                pool,
                other.id,
                destination_account_id=ACCOUNT,
                requester_account_id="acc_root",
                crypto_box=CryptoBox(os.urandom(32)),
            )
        )

    assert connector.live_at_dispatch == [[]]
    assert isinstance(reparent_error, ConflictError), (
        f"reparent_result={reparent_error!r}: reparent carried a routing bound "
        "chat into an account with a pending unregister"
    )
    assert reparent_error.detail["reason"] == "number_unregister_pending"
    async with pool.acquire() as conn:
        assert (
            await queries.list_attached_connections_for_phone(
                conn, "signal", DIGITS, account_id=ACCOUNT
            )
            == []
        )
