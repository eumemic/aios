"""Terminal workflow runs are automatically archived only after their grace window."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aios.db.queries.prune import reconcile_terminal_archival_batch


@pytest.mark.asyncio
async def test_terminal_archival_is_bounded_aged_and_idempotent() -> None:
    conn = AsyncMock()
    conn.execute.return_value = "UPDATE 17"

    count = await reconcile_terminal_archival_batch(
        conn, grace_days=7, request_copy_grace_days=0, row_limit=123
    )

    assert count == 17
    sql, grace_days, row_limit, request_copy_grace_days = conn.execute.await_args.args
    assert "r.updated_at < now() - make_interval(days => LEAST($1::int, $3::int))" in sql
    # A request-copy run takes the short grace, every other run the default.
    assert "THEN $3::int ELSE $1::int END" in sql
    assert "r.request_ref_id IS NOT NULL" in sql
    assert "SET archived_at = COALESCE(r.archived_at, now())" in sql
    assert "r.archived_at IS NULL" in sql
    assert "LIMIT $2" in sql
    assert (grace_days, row_limit, request_copy_grace_days) == (7, 123, 0)
