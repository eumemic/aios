"""Workspace-root startup validation modes and full enumeration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aios.sandbox.workspace_root_startup import (
    WorkspaceRootValidationError,
    validate_workspace_root_against_sessions,
)


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def __aenter__(self) -> Any:
        return self.conn

    async def __aexit__(self, *args: Any) -> None:
        return None


class _Pool:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.conn = AsyncMock()
        self.conn.fetch.side_effect = [rows, []]

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


@pytest.mark.asyncio
async def test_warn_mode_enumerates_every_violation_without_aborting(tmp_path: Path) -> None:
    rows = [
        {"id": "sess_a", "account_id": "acc_a", "workspace_volume_path": "/wrong/a"},
        {"id": "sess_b", "account_id": "acc_b", "workspace_volume_path": "/wrong/b"},
    ]
    log = Mock()
    with (
        patch("aios.sandbox.workspace_root_startup.get_settings") as settings,
        patch("aios.sandbox.workspace_root_startup.get_logger", return_value=log),
    ):
        settings.return_value.workspace_root = tmp_path
        settings.return_value.workspace_scan_timeout_seconds = 30.0
        settings.return_value.workspace_scan_query_timeout_seconds = 10.0
        result = await validate_workspace_root_against_sessions(
            _Pool(rows), service="worker", mode="warn"
        )

    assert result.violation_count == 2
    assert log.warning.call_count == 2
    assert {call.kwargs["session_id"] for call in log.warning.call_args_list} == {
        "sess_a",
        "sess_b",
    }


@pytest.mark.asyncio
async def test_enforce_mode_scans_all_then_raises_once(tmp_path: Path) -> None:
    rows = [
        {"id": f"sess_{i}", "account_id": "acc", "workspace_volume_path": f"/wrong/{i}"}
        for i in range(12)
    ]
    log = Mock()
    with (
        patch("aios.sandbox.workspace_root_startup.get_settings") as settings,
        patch("aios.sandbox.workspace_root_startup.get_logger", return_value=log),
    ):
        settings.return_value.workspace_root = tmp_path
        settings.return_value.workspace_scan_timeout_seconds = 30.0
        settings.return_value.workspace_scan_query_timeout_seconds = 10.0
        with pytest.raises(WorkspaceRootValidationError) as caught:
            await validate_workspace_root_against_sessions(
                _Pool(rows), service="api", mode="enforce"
            )

    assert caught.value.violation_count == 12
    assert "12 violation(s)" in str(caught.value)
    assert "sess_0" in str(caught.value)
    assert "sess_11" not in str(caught.value)  # bounded sample
    assert log.warning.call_count == 12


@pytest.mark.asyncio
async def test_off_mode_does_not_scan() -> None:
    pool = _Pool([])
    result = await validate_workspace_root_against_sessions(pool, service="api", mode="off")
    assert result.violation_count == 0
    pool.conn.fetch.assert_not_awaited()
