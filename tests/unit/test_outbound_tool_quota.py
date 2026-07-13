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
    assert "role = 'tool'" in sql
    assert "session_id = $1" in sql
    assert "tool_name = $2" in sql
    assert args == ["ses_1", "matrix_invite", 3600]


async def test_different_verb_is_disabled_without_query(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_invite": (3600, 20)}),
    )
    pool = MagicMock()

    assert await outbound_tool_quota.check_outbound_tool_quota(pool, "ses_1", "matrix_send") is None
    pool.acquire.assert_not_called()
