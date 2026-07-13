"""Regression coverage for quota reservations across pooled connections."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aios.services import outbound_tool_quota


async def test_reservations_lock_the_same_key_on_distinct_connections(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        outbound_tool_quota,
        "get_settings",
        lambda: SimpleNamespace(outbound_tool_quotas={"matrix_send": (3600, 2)}),
    )
    connections = []
    for _ in range(2):
        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.fetchval = AsyncMock(return_value=0)
        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock(return_value=None)
        transaction.__aexit__ = AsyncMock(return_value=None)
        conn.transaction.return_value = transaction
        connections.append(conn)

    pool = MagicMock()
    acquisitions = []
    for conn in connections:
        acquisition = MagicMock()
        acquisition.__aenter__ = AsyncMock(return_value=conn)
        acquisition.__aexit__ = AsyncMock(return_value=None)
        acquisitions.append(acquisition)
    pool.acquire.side_effect = acquisitions

    for _ in range(2):
        async with outbound_tool_quota.outbound_tool_quota_reservation(
            pool, "ses_1", "mcp__matrix__matrix_send"
        ) as refusal:
            assert refusal is None

    keys = [conn.execute.await_args.args[1] for conn in connections]
    assert keys[0] == keys[1]
    assert "ses_1" in keys[0]
    assert "matrix_send" in keys[0]
