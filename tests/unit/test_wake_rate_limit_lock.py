"""Unit coverage for the per-``(source, target)`` advisory lock that guards the
``wake_session`` rate cap.

The concurrency bug this closes: :func:`deliver_cross_session_wake` used to read
the per-pair wake count on one pooled connection and append the lineage span +
user message on a *different* connection with no lock bridging the two, so a
concurrent burst of same-pair callers all read a stale count (READ COMMITTED)
and all appended — overshooting ``WAKE_SESSION_MAX_PER_HOUR``.  The fix takes a
transaction-scoped ``pg_advisory_xact_lock`` keyed on
``(root.source_id, target)`` and runs the count + append under that lock in ONE
transaction, mirroring the pattern ``outbound_tool_quota`` already uses.

These unit tests pin the *connection-shape* of the fix (the lock SQL, its key,
its position, its transactional scope) without a database.  The real-Postgres
concurrency regression — that a same-pair burst no longer overshoots — lives in
``tests/integration/test_wake_session.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.services.wake import (
    WAKE_LINEAGE_SPAN_EVENT,
    WAKE_SESSION_MAX_PER_HOUR,
    CrossSessionWakeRoot,
    WakeSessionRateLimitedError,
    deliver_cross_session_wake,
)


class _FakeTransaction:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    async def __aenter__(self) -> None:
        self.conn.in_transaction = True

    async def __aexit__(self, *_args: Any) -> None:
        self.conn.in_transaction = False


class _FakeConn:
    """Records every SQL statement (and the transactional state at the time) so
    the lock / count / append ordering and the lock key can be asserted.
    """

    def __init__(self, *, target_row: dict[str, Any] | None, count: int) -> None:
        self.target_row = target_row
        self.count = count
        self.in_transaction = False
        self.calls: list[tuple[str, str, tuple[Any, ...], bool]] = []
        # (method, sql, args, in_transaction_at_call_time)

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchrow", sql, args, self.in_transaction))
        return self.target_row

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append(("fetchval", sql, args, self.in_transaction))
        return self.count

    async def execute(self, sql: str, *args: Any) -> None:
        self.calls.append(("execute", sql, args, self.in_transaction))


class _FakeAcquire:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        self.pool.acquisitions += 1
        return self.pool.conn

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakePool:
    """Single shared conn across all ``acquire()`` blocks, matching the
    mock-pool shape the rest of the wake unit suite uses."""

    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn
        self.acquisitions = 0

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self)


_TARGET_ROW = {"account_id": "acc_test_stub", "archived_at": None}


def _root(source_id: str = "sess_01SOURCE", source_depth: int = 0) -> CrossSessionWakeRoot:
    return CrossSessionWakeRoot(source_id=source_id, source_depth=source_depth)


async def test_advisory_lock_runs_before_count_and_inside_transaction() -> None:
    """The per-pair advisory lock is the FIRST statement inside the append
    transaction, and the count read runs AFTER it (under the lock) — so every
    prior concurrent caller's committed span is visible to the count."""
    conn = _FakeConn(target_row=_TARGET_ROW, count=0)
    pool = _FakePool(conn)

    with (
        patch("aios.db.queries.append_event", AsyncMock()) as append,
        patch("aios.services.wake.defer_wake", AsyncMock()),
    ):
        await deliver_cross_session_wake(
            pool,
            target_session_id="sess_01TARGET",
            content="hi",
            account_id="acc_test_stub",
            root=_root(),
            cause="agent_wake",
        )

    methods = [m for m, _, _, _ in conn.calls]
    # First the target-row load (its own acquire, no transaction yet)...
    assert methods[0] == "fetchrow"
    assert conn.calls[0][3] is False  # target load is outside the append txn
    # ...then INSIDE the transaction: lock first, then count.
    assert methods[1] == "execute"
    assert "pg_advisory_xact_lock" in conn.calls[1][1]
    assert conn.calls[1][3] is True  # the lock is taken inside the transaction
    assert methods[2] == "fetchval"
    assert "count(*)" in conn.calls[2][1]
    assert conn.calls[2][3] is True  # the count is inside the (locked) transaction
    # Exactly two appends (span + message), issued AFTER the count under the
    # same outer transaction.
    assert append.await_count == 2
    assert conn.in_transaction is False  # transaction committed / released
    # Two pool acquisitions: target load + locked append txn.  Never held across
    # defer_wake (which runs after the transaction releases its connection).
    assert pool.acquisitions == 2


async def test_advisory_lock_key_is_per_source_target_pair() -> None:
    """The lock key is ``f"{source_id}:{target_id}"`` — scope-matched to the
    per-pair cap so concurrent wakes from OTHER sources to this target take a
    DIFFERENT lock and are unaffected."""
    conn = _FakeConn(target_row=_TARGET_ROW, count=0)
    pool = _FakePool(conn)

    with (
        patch("aios.db.queries.append_event", AsyncMock()),
        patch("aios.services.wake.defer_wake", AsyncMock()),
    ):
        await deliver_cross_session_wake(
            pool,
            target_session_id="sess_01TARGET",
            content="hi",
            account_id="acc_test_stub",
            root=_root(source_id="sess_42SRC"),
            cause="agent_wake",
        )

    lock_call = next(c for c in conn.calls if c[0] == "execute")
    assert lock_call[2] == ("sess_42SRC:sess_01TARGET",)


async def test_rate_limit_check_uses_locked_count_not_stale_pool_read() -> None:
    """At the cap, the locked count read refuses the append — and no append or
    defer fires (the refusal is side-effect-free, and the lock is released by
    the transaction rollback)."""
    conn = _FakeConn(target_row=_TARGET_ROW, count=WAKE_SESSION_MAX_PER_HOUR)
    pool = _FakePool(conn)

    with (
        patch("aios.db.queries.append_event", AsyncMock()) as append,
        patch("aios.services.wake.defer_wake", AsyncMock()) as defer,
        pytest.raises(WakeSessionRateLimitedError, match="rate limit"),
    ):
        await deliver_cross_session_wake(
            pool,
            target_session_id="sess_01TARGET",
            content="hi",
            account_id="acc_test_stub",
            root=_root(),
            cause="agent_wake",
        )

    # The lock runs before the count; the count (at-cap) refuses; nothing is
    # appended and no wake is deferred.
    assert any(c[0] == "execute" and "pg_advisory_xact_lock" in c[1] for c in conn.calls)
    assert any(c[0] == "fetchval" and "count(*)" in c[1] for c in conn.calls)
    append.assert_not_awaited()
    defer.assert_not_awaited()
    assert conn.in_transaction is False  # rolled back → lock released


async def test_distinct_pairs_take_distinct_lock_keys() -> None:
    """Two deliveries with different (source, target) pairs compute two
    distinct lock keys — the lock must not collapse distinct pairs onto one
    key (which would over-serialize unrelated caps)."""
    keys: list[str] = []
    for source_id, target_id in (
        ("sess_A", "sess_T1"),
        ("sess_A", "sess_T2"),
        ("sess_B", "sess_T1"),
    ):
        conn = _FakeConn(target_row=_TARGET_ROW, count=0)
        pool = _FakePool(conn)
        with (
            patch("aios.db.queries.append_event", AsyncMock()),
            patch("aios.services.wake.defer_wake", AsyncMock()),
        ):
            await deliver_cross_session_wake(
                pool,
                target_session_id=target_id,
                content="hi",
                account_id="acc_test_stub",
                root=_root(source_id=source_id),
                cause="agent_wake",
            )
        lock_call = next(c for c in conn.calls if c[0] == "execute")
        keys.append(lock_call[2][0])

    assert keys == ["sess_A:sess_T1", "sess_A:sess_T2", "sess_B:sess_T1"]
    assert len(set(keys)) == 3  # all distinct — per-pair, not per-target


async def test_count_sql_remains_scoped_to_trusted_span_lineage() -> None:
    """The locked count read must still filter to the system-owned
    ``kind='span'`` lineage event (issue #1083): forgeable user messages must
    move neither the count nor the cap.  Refactoring the count helper to run
    inside the transaction must not change the SQL shape."""
    conn = _FakeConn(target_row=_TARGET_ROW, count=0)
    pool = _FakePool(conn)

    with (
        patch("aios.db.queries.append_event", AsyncMock()),
        patch("aios.services.wake.defer_wake", AsyncMock()),
    ):
        await deliver_cross_session_wake(
            pool,
            target_session_id="sess_01TARGET",
            content="hi",
            account_id="acc_test_stub",
            root=_root(),
            cause="agent_wake",
        )

    count_call = next(c for c in conn.calls if c[0] == "fetchval")
    sql = count_call[1]
    args = count_call[2]
    assert "kind = 'span'" in sql
    assert "'role' = 'user'" not in sql
    assert "data->>'event' = $4" in sql
    assert "data->>'wake_source_session_id' = $3" in sql
    # Positional bind order is load-bearing (target, account, source, event).
    assert args == ("sess_01TARGET", "acc_test_stub", "sess_01SOURCE", WAKE_LINEAGE_SPAN_EVENT)
