from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aios.services import sessions


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_args: object) -> None:
        return None


def _pool(conn: MagicMock) -> MagicMock:
    pool = MagicMock()
    pool.acquire.return_value = _AsyncContext(conn)
    conn.transaction.return_value = _AsyncContext(None)
    return pool


async def test_delete_session_removes_snapshot_and_all_session_directories(tmp_path: Path) -> None:
    session_id = "sess_delete"
    account_id = "acc_delete"
    workspace = tmp_path / account_id / session_id
    roots = [
        workspace,
        tmp_path / "_uploads" / session_id,
        tmp_path / "_attachments" / session_id,
        tmp_path / "_session_repos" / session_id,
    ]
    for root in roots:
        root.mkdir(parents=True)
        (root / "bytes").write_text("data")

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={"workspace_volume_path": str(workspace), "snapshot_ref": "snapshot-ref"}
    )
    pool = _pool(conn)
    store = MagicMock()
    store.remove = AsyncMock(return_value=True)

    with (
        patch("aios.services.sessions.get_snapshot_store", return_value=store),
        patch("aios.sandbox.volumes.get_settings") as settings,
        patch("aios.services.sessions.queries.delete_session", AsyncMock()),
        patch("aios.services.sessions.fail_open_child_requests_conn", AsyncMock(return_value=None)),
    ):
        settings.return_value.workspace_root = tmp_path
        await sessions.delete_session(pool, session_id, account_id=account_id)

    store.remove.assert_awaited_once_with("snapshot-ref")
    assert all(not root.exists() for root in roots)
