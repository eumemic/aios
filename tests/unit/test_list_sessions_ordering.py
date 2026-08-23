"""Session list ordering uses a deterministic timestamp/id keyset."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aios.db.queries import sessions as session_queries


class _CapturingConn:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.args: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.sql = sql
        self.args = args
        return []


@pytest.mark.parametrize(
    ("order_by", "expression"),
    [
        ("created_at", "sessions.created_at"),
        ("updated_at", "sessions.updated_at"),
        ("last_event_at", "sessions.last_event_at"),
    ],
)
async def test_list_sessions_orders_desc_with_id_tiebreaker(order_by: str, expression: str) -> None:
    conn = _CapturingConn()
    await session_queries.list_sessions(
        conn,
        account_id="acc_x",
        order_by=order_by,  # type: ignore[arg-type]
    )

    assert conn.sql is not None
    assert f"ORDER BY {expression} DESC NULLS LAST, sessions.id DESC" in conn.sql


async def test_list_sessions_applies_composite_cursor() -> None:
    conn = _CapturingConn()
    anchor = datetime(2026, 7, 12, 23, 25, tzinfo=UTC)
    await session_queries.list_sessions(
        conn,
        account_id="acc_x",
        order_by="last_event_at",
        after=(anchor, "sess_anchor"),
    )

    assert conn.sql is not None
    assert "last_event_at <" in conn.sql
    assert "last_event_at =" in conn.sql
    assert "last_event_at IS NULL" in conn.sql
    assert anchor in conn.args
    assert "sess_anchor" in conn.args


async def test_list_sessions_pages_within_null_last_event_group() -> None:
    conn = _CapturingConn()
    await session_queries.list_sessions(
        conn,
        account_id="acc_x",
        order_by="last_event_at",
        after=(None, "sess_anchor"),
    )

    assert conn.sql is not None
    assert "last_event_at IS NULL AND sessions.id <" in conn.sql
