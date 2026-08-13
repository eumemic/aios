"""Exit codes for ``aios ops validate-workspace-roots``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from aios.cli.app import app
from aios.sandbox.workspace_root_startup import WorkspaceRootValidationResult

runner = CliRunner()


def _pool() -> AsyncMock:
    pool = AsyncMock()
    pool.close = AsyncMock()
    return pool


def test_preflight_exits_zero_when_clean() -> None:
    pool = _pool()
    with (
        patch("aios.config.get_settings", return_value=SimpleNamespace(db_url="postgresql://db")),
        patch("aios.db.pool.create_pool", AsyncMock(return_value=pool)),
        patch(
            "aios.sandbox.workspace_root_startup.validate_workspace_root_against_sessions",
            AsyncMock(return_value=WorkspaceRootValidationResult(violation_count=0)),
        ),
    ):
        result = runner.invoke(app, ["ops", "validate-workspace-roots"])

    assert result.exit_code == 0, result.output
    assert "0 violation(s)" in result.output
    pool.close.assert_awaited_once()


def test_preflight_exits_nonzero_on_violations() -> None:
    pool = _pool()
    with (
        patch("aios.config.get_settings", return_value=SimpleNamespace(db_url="postgresql://db")),
        patch("aios.db.pool.create_pool", AsyncMock(return_value=pool)),
        patch(
            "aios.sandbox.workspace_root_startup.validate_workspace_root_against_sessions",
            AsyncMock(return_value=WorkspaceRootValidationResult(violation_count=3)),
        ),
    ):
        result = runner.invoke(app, ["ops", "validate-workspace-roots"])

    assert result.exit_code == 1
    assert "3 violation(s)" in result.output
    pool.close.assert_awaited_once()
