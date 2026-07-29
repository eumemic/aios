"""Unit coverage for the durable outbound quota reservation service (#1903).

Real-Postgres semantics (atomic count+insert, concurrency bursts, rolling
expiry, constrained pools) are covered by
``tests/integration/test_outbound_tool_quota_db.py``; these tests pin the
pure/connection-shape behavior that needs no database.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.services import outbound_tool_quota


def _settings(quotas: dict[str, tuple[int, int]]) -> Any:
    return SimpleNamespace(outbound_tool_quotas=quotas)


class _FakeTransaction:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    async def __aenter__(self) -> None:
        self.conn.in_transaction = True

    async def __aexit__(self, *_args: Any) -> None:
        self.conn.in_transaction = False


class _FakeConn:
    """Records the SQL statements the reservation runs, in order."""

    def __init__(self, count: int = 0) -> None:
        self.count = count
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.in_transaction = False

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def execute(self, sql: str, *args: Any) -> None:
        self.calls.append((sql, args))

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        if "count(*)" in sql:
            return self.count
        if "INSERT INTO" in sql:
            return "res_1"
        raise AssertionError(f"unexpected fetchval: {sql}")


class _FakeAcquire:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        self.pool.held += 1
        self.pool.acquisitions += 1
        return self.pool.conn

    async def __aexit__(self, *_args: Any) -> None:
        self.pool.held -= 1


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn
        self.held = 0
        self.acquisitions = 0

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self)


async def test_disabled_quota_issues_no_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(outbound_tool_quota, "get_settings", lambda: _settings({}))
    pool = MagicMock()

    admission = await outbound_tool_quota.reserve_outbound_tool_quota(
        pool, "ses_1", "telegram_send"
    )

    assert admission.refusal is None
    assert admission.reservation_id is None
    pool.acquire.assert_not_called()


async def test_unconfigured_verb_is_disabled_without_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota, "get_settings", lambda: _settings({"matrix_invite": (3600, 20)})
    )
    pool = MagicMock()

    admission = await outbound_tool_quota.reserve_outbound_tool_quota(pool, "ses_1", "matrix_send")

    assert admission.refusal is None and admission.reservation_id is None
    pool.acquire.assert_not_called()


async def test_admission_inserts_reservation_under_lock(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota, "get_settings", lambda: _settings({"matrix_send": (3600, 5)})
    )
    conn = _FakeConn(count=2)
    pool = _FakePool(conn)

    admission = await outbound_tool_quota.reserve_outbound_tool_quota(pool, "ses_1", "matrix_send")

    assert admission.refusal is None
    assert admission.reservation_id == "res_1"
    sqls = [sql for sql, _ in conn.calls]
    assert "pg_advisory_xact_lock" in sqls[0]
    assert "DELETE FROM outbound_tool_reservations" in sqls[1]
    assert "count(*)" in sqls[2]
    assert "INSERT INTO outbound_tool_reservations" in sqls[3]
    # One pooled connection, released by the time the call returns — the
    # reservation must never be held across external I/O.
    assert pool.acquisitions == 1
    assert pool.held == 0
    assert conn.in_transaction is False


async def test_refusal_inserts_nothing_and_formats_message(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota, "get_settings", lambda: _settings({"matrix_invite": (3600, 20)})
    )
    conn = _FakeConn(count=20)
    pool = _FakePool(conn)

    admission = await outbound_tool_quota.reserve_outbound_tool_quota(
        pool, "ses_1", "matrix_invite"
    )

    assert admission.refusal == "quota_exceeded: matrix_invite 20/20 per hour"
    assert admission.reservation_id is None
    assert not any("INSERT" in sql for sql, _ in conn.calls)
    assert pool.held == 0


async def test_mcp_names_canonicalize_to_shared_verb(monkeypatch: Any) -> None:
    """Sibling effectors (``mcp__a__matrix_send``, ``mcp__b__matrix_send``)
    lock, purge, count, and insert under the SAME canonical verb — no
    per-server bypass of a shared quota."""
    monkeypatch.setattr(
        outbound_tool_quota, "get_settings", lambda: _settings({"matrix_send": (3600, 2)})
    )
    keys: list[tuple[str, tuple[Any, ...]]] = []
    for name in ("mcp__server_a__matrix_send", "mcp__server_b__matrix_send", "matrix_send"):
        conn = _FakeConn(count=0)
        pool = _FakePool(conn)
        admission = await outbound_tool_quota.reserve_outbound_tool_quota(pool, "ses_1", name)
        assert admission.reservation_id == "res_1"
        keys.extend(conn.calls)

    lock_keys = {args[0] for sql, args in keys if "pg_advisory_xact_lock" in sql}
    assert lock_keys == {"outbound-tool-quota:ses_1:matrix_send"}
    counted_verbs = {args[1] for sql, args in keys if "count(*)" in sql}
    inserted_verbs = {args[1] for sql, args in keys if "INSERT" in sql}
    assert counted_verbs == {"matrix_send"}
    assert inserted_verbs == {"matrix_send"}


async def test_completion_mark_failure_is_swallowed() -> None:
    """``mark_outbound_dispatch_completed`` is observability only: a failure
    leaves the row in the conservative ``admitted`` state and must never
    break result publication."""
    pool = MagicMock()
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(side_effect=RuntimeError("pool down"))
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire

    await outbound_tool_quota.mark_outbound_dispatch_completed(pool, "res_1")  # no raise


def test_display_window_wordings() -> None:
    assert outbound_tool_quota._display_window(3600) == "hour"
    assert outbound_tool_quota._display_window(60) == "minute"
    assert outbound_tool_quota._display_window(86400) == "day"
    assert outbound_tool_quota._display_window(90) == "90 seconds"
