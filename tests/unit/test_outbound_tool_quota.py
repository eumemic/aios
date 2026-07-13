from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
        refusal_times.append(now)  # These persisted tool results are deliberately uncounted.

    now = 71
    assert (
        await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_invite") is None
    )
    assert refusal_times == [30, 40, 50]


async def test_different_verb_is_disabled_without_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_invite": (3600, 20)}),
    )
    pool = MagicMock()

    assert await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_send") is None
    pool.acquire.assert_not_called()
