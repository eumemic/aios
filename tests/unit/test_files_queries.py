"""Unit coverage for ``queries.get_file`` (#179 image-serve slice).

Query-layer test with a hand-rolled fake connection (no real DB needed —
same pattern as ``test_db_stats_queries.py``), pinning the scoping
contract: a hit returns the row, a miss/wrong-session/wrong-account all
collapse to the same 404 so a cross-account file id can't be distinguished
from a genuinely missing file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aios.db.queries import files as files_queries
from aios.errors import NotFoundError

NOW = datetime(2026, 8, 28, tzinfo=UTC)

_ROW = {
    "id": "file_abc",
    "session_id": "sess_1",
    "filename": "photo.png",
    "host_path": "/data/_uploads/sess_1/file_abc/photo.png",
    "in_sandbox_path": "/mnt/uploads/file_abc/photo.png",
    "size": 1234,
    "content_type": "image/png",
    "sha256": "deadbeef",
    "created_at": NOW,
}


class _Connection:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        return self._row


async def test_get_file_returns_row_scoped_by_session_and_account() -> None:
    conn = _Connection(_ROW)

    result = await files_queries.get_file(
        conn,
        "sess_1",
        "file_abc",
        account_id="acc_1",
    )

    assert result.id == "file_abc"
    assert result.filename == "photo.png"
    assert result.content_type == "image/png"
    assert result.host_path == _ROW["host_path"]
    # The query params carry all three scoping dimensions positionally.
    assert conn.calls[0][1] == ("file_abc", "sess_1", "acc_1")


async def test_get_file_missing_row_is_not_found() -> None:
    conn = _Connection(None)

    with pytest.raises(NotFoundError):
        await files_queries.get_file(
            conn,
            "sess_1",
            "file_missing",
            account_id="acc_1",
        )


async def test_get_file_wrong_session_or_account_is_indistinguishable_404() -> None:
    """A cross-account or cross-session lookup must 404 exactly like a
    genuinely nonexistent file id — the scoping is baked into the WHERE
    clause, not a separate ownership check after the fact."""
    conn = _Connection(None)  # simulates the WHERE excluding the row

    with pytest.raises(NotFoundError) as excinfo:
        await files_queries.get_file(
            conn,
            "sess_other",
            "file_abc",
            account_id="acc_other",
        )
    assert excinfo.value.detail == {"session_id": "sess_other", "file_id": "file_abc"}
