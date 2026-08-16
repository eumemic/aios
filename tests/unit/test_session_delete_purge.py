from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        patch(
            "aios.services.sessions.queries.unscoped_live_workspace_volume_paths",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.services.sessions.queries.acquire_workspace_hierarchy_advisory_xact_locks",
            AsyncMock(),
        ),
        patch("aios.services.sessions.fail_open_child_requests_conn", AsyncMock(return_value=None)),
    ):
        settings.return_value.workspace_root = tmp_path
        await sessions.delete_session(pool, session_id, account_id=account_id)

    store.remove.assert_awaited_once_with("snapshot-ref")
    assert all(not root.exists() for root in roots)


async def test_delete_session_out_of_jail_workspace_skips_purge_and_deletes_row(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    session_id = "sess_stale"
    account_id = "acc_delete"
    workspace_root = tmp_path / "current"
    stale_workspace = tmp_path / "old" / account_id / session_id
    stale_workspace.mkdir(parents=True)
    marker = stale_workspace / "bytes"
    marker.write_text("must survive")

    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={"workspace_volume_path": str(stale_workspace), "snapshot_ref": None}
    )
    pool = _pool(conn)
    delete_row = AsyncMock()

    with (
        patch("aios.services.sessions.get_settings") as service_settings,
        patch("aios.sandbox.volumes.get_settings") as volume_settings,
        patch("aios.services.sessions.queries.delete_session", delete_row),
        patch(
            "aios.services.sessions.queries.unscoped_live_workspace_volume_paths",
            AsyncMock(return_value=[]),
        ),
        patch("aios.services.sessions.fail_open_child_requests_conn", AsyncMock(return_value=None)),
    ):
        service_settings.return_value.workspace_root = workspace_root
        volume_settings.return_value.workspace_root = workspace_root
        await sessions.delete_session(pool, session_id, account_id=account_id)

    delete_row.assert_awaited_once_with(conn, session_id, account_id=account_id)
    assert marker.read_text() == "must survive"
    assert "skipping workspace hierarchy lock outside workspace_root" in caplog.text
    assert str(stale_workspace) in caplog.text
    assert session_id in caplog.text


async def test_delete_session_in_jail_still_takes_full_hierarchy_lock(tmp_path: Path) -> None:
    session_id = "sess_delete"
    account_id = "acc_delete"
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / account_id / session_id
    workspace.mkdir(parents=True)

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={"workspace_volume_path": str(workspace), "snapshot_ref": None}
    )
    pool = _pool(conn)
    hierarchy_lock = AsyncMock()

    with (
        patch("aios.services.sessions.get_settings") as settings,
        patch("aios.sandbox.volumes.get_settings", return_value=settings.return_value),
        patch("aios.services.sessions.queries.delete_session", AsyncMock()),
        patch(
            "aios.services.sessions.queries.unscoped_live_workspace_volume_paths",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.services.sessions.queries.acquire_workspace_hierarchy_advisory_xact_locks",
            hierarchy_lock,
        ),
        patch("aios.services.sessions.fail_open_child_requests_conn", AsyncMock(return_value=None)),
    ):
        settings.return_value.workspace_root = workspace_root
        await sessions.delete_session(pool, session_id, account_id=account_id)

    hierarchy_lock.assert_awaited_once_with(
        conn, str(workspace.resolve()), boundary=str(workspace_root)
    )
