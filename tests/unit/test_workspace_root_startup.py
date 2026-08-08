"""Startup validation for API/worker workspace-root agreement (#2064)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.config import get_settings
from aios.sandbox.workspace_root_startup import validate_workspace_root_against_sessions


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    return tmp_path


def _pool(rows: list[dict[str, str]]) -> MagicMock:
    conn = AsyncMock()
    conn.fetch.return_value = rows
    acquired = AsyncMock()
    acquired.__aenter__.return_value = conn
    pool = MagicMock()
    pool.acquire.return_value = acquired
    return pool


@pytest.mark.asyncio
async def test_accepts_canonical_account_scoped_rows(workspace_root: Path) -> None:
    row = {
        "id": "sess_ok",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "acc_a" / "sess_ok"),
    }
    await validate_workspace_root_against_sessions(_pool([row]), service="worker")


@pytest.mark.asyncio
async def test_rejects_root_drift_at_startup_with_full_diagnostic(workspace_root: Path) -> None:
    raw = "/srv/aios/workspaces/acc_a/sess_bad"
    row = {"id": "sess_bad", "account_id": "acc_a", "workspace_volume_path": raw}

    with pytest.raises(RuntimeError) as exc_info:
        await validate_workspace_root_against_sessions(_pool([row]), service="worker")

    message = str(exc_info.value)
    assert "workspace-root startup validation failed" in message
    assert "service='worker'" in message
    assert f"workspace_root={str(workspace_root)!r}" in message
    assert f"account_root={str(workspace_root / 'acc_a')!r}" in message
    assert f"raw_path={raw!r}" in message
    assert f"resolved_path={raw!r}" in message
    assert "account_id='acc_a'" in message
    assert "session_id='sess_bad'" in message


@pytest.mark.asyncio
async def test_cross_tenant_row_still_fails_closed(workspace_root: Path) -> None:
    row = {
        "id": "sess_a",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "acc_b" / "sess_b"),
    }
    with pytest.raises(RuntimeError):
        await validate_workspace_root_against_sessions(_pool([row]), service="api")


@pytest.mark.asyncio
async def test_absolute_legacy_row_remains_accepted(workspace_root: Path) -> None:
    row = {
        "id": "sess_legacy",
        "account_id": "acc_a",
        "workspace_volume_path": str(workspace_root / "sess_legacy"),
    }
    await validate_workspace_root_against_sessions(_pool([row]), service="worker")
