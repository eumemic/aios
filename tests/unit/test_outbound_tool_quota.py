from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.services import outbound_tool_quota


async def test_disabled_quota_issues_no_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={}),
    )
    pool = MagicMock()

    assert (
        await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "telegram_send") is None
    )
    pool.acquire.assert_not_called()


async def test_quota_counts_session_and_verb_and_formats_refusal(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_invite": (3600, 20)}),
    )
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=20)
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = acquire

    refusal = await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_invite")

    assert refusal == "quota_exceeded: matrix_invite 20/20 per hour"
    sql, *args = conn.fetchval.await_args.args
    assert "data->>'event' = 'tool_execute_end'" in sql
    assert "data->>'is_error' = 'false'" in sql
    assert "role = 'tool'" not in sql
    assert "session_id = $1" in sql
    assert "tool_name = $2" in sql
    assert args == ["ses_1", "matrix_invite", 3600]


async def test_denied_retries_do_not_extend_rolling_window(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_invite": (60, 2)}),
    )
    now = 30
    dispatch_times = [0, 10]
    refusal_times: list[int] = []
    conn = MagicMock()

    async def count_dispatches(*_args: Any) -> int:
        return sum(dispatched_at > now - 60 for dispatched_at in dispatch_times)

    conn.fetchval = AsyncMock(side_effect=count_dispatches)
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = acquire

    for now in (30, 40, 50):
        refusal = await outbound_tool_quota.check_outbound_tool_quota(
            pool, "ses_1", "matrix_invite"
        )
        assert refusal is not None
        refusal_times.append(now)

    now = 71
    assert (
        await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_invite") is None
    )
    assert refusal_times == [30, 40, 50]


class _FakeAcquire:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def __aenter__(self) -> Any:
        return self.conn

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeTransaction:
    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock

    async def __aenter__(self) -> None:
        await self.lock.acquire()

    async def __aexit__(self, *_args: Any) -> None:
        self.lock.release()


class _FakeQuotaConnection:
    def __init__(self, successes: list[str]) -> None:
        self.successes = successes
        self.lock = asyncio.Lock()
        self.lock_keys: list[str] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self.lock)

    async def execute(self, sql: str, key: str) -> None:
        assert "pg_advisory_xact_lock" in sql
        self.lock_keys.append(key)

    async def fetchval(self, *_args: Any) -> int:
        return len(self.successes)


class _FakePool:
    def __init__(self, conn: _FakeQuotaConnection) -> None:
        self.conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


async def test_concurrent_same_key_dispatches_reserve_atomically(monkeypatch: Any) -> None:
    """Two simultaneous admissions at cap=1 publish at most one success."""
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_send": (3600, 1)}),
    )
    successes: list[str] = []
    conn = _FakeQuotaConnection(successes)
    pool = _FakePool(conn)
    entered = asyncio.Event()

    async def dispatch(call_id: str) -> str | None:
        async with outbound_tool_quota.outbound_tool_quota_reservation(
            pool, "ses_1", "matrix_send"
        ) as refusal:
            if refusal is None:
                entered.set()
                await asyncio.sleep(0)
                successes.append(call_id)  # successful result/span publication
            return refusal

    first = asyncio.create_task(dispatch("tc_1"))
    await entered.wait()
    second = asyncio.create_task(dispatch("tc_2"))
    refusals = await asyncio.gather(first, second)

    assert successes == ["tc_1"]
    assert refusals == [None, "quota_exceeded: matrix_send 1/1 per hour"]
    assert len(set(conn.lock_keys)) == 1


async def test_failed_publication_releases_reservation_without_consuming_quota(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_send": (3600, 1)}),
    )
    successes: list[str] = []
    pool = _FakePool(_FakeQuotaConnection(successes))

    with pytest.raises(RuntimeError, match="publish failed"):
        async with outbound_tool_quota.outbound_tool_quota_reservation(
            pool, "ses_1", "matrix_send"
        ) as refusal:
            assert refusal is None
            raise RuntimeError("publish failed")

    async with outbound_tool_quota.outbound_tool_quota_reservation(
        pool, "ses_1", "matrix_send"
    ) as refusal:
        assert refusal is None
        successes.append("tc_retry")

    assert successes == ["tc_retry"]


async def test_different_verb_is_disabled_without_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_invite": (3600, 20)}),
    )
    pool = MagicMock()

    assert await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_send") is None
    pool.acquire.assert_not_called()
