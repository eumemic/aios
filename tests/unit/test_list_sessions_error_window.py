"""Session list error-window filtering is applied in SQL before pagination."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.db.queries import sessions as session_queries


class _CapturingConn:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.args: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.sql = sql
        self.args = args
        return []


async def test_list_sessions_filters_error_stop_reason_and_since_before_limit() -> None:
    conn = _CapturingConn()
    since = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    await session_queries.list_sessions(
        conn,
        account_id="acc_x",
        stop_reason="error",
        since=since,
        limit=10,
    )

    assert conn.sql is not None
    assert "stop_reason->>'type' = $2" in conn.sql
    assert "updated_at >= $3" in conn.sql
    assert conn.sql.index("updated_at >= $3") < conn.sql.index("LIMIT $4")
    assert conn.args == ("acc_x", "error", since, 10)
