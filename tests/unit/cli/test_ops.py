"""Tests for operator subcommands (api, worker, migrate)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_api_command_passes_proxy_headers_to_uvicorn(monkeypatch):
    """The `api` command must pass proxy_headers=True and forwarded_allow_ips="*"
    to uvicorn.run so that X-Forwarded-For headers are trusted when running
    behind a reverse proxy."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/test")
    # The shared-DB guard fires from a linked worktree (the mandated dev
    # workflow) because db name "test" is not aios_dev_*. Disable it here so we
    # exercise uvicorn wiring, not the guard. (CI runs from a main checkout
    # where the guard is a no-op, so it would never catch this regression.)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: False)

    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["api"])

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("proxy_headers") is True, "uvicorn.run must be called with proxy_headers=True"
    assert kwargs.get("forwarded_allow_ips") == "*", (
        'uvicorn.run must be called with forwarded_allow_ips="*"'
    )


def test_api_command_aborts_before_uvicorn_on_shared_worktree(monkeypatch):
    """`api` from a linked worktree on the shared DB aborts before uvicorn.run."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(app, ["api"])

    mock_run.assert_not_called()
    assert result.exit_code == 1


def test_worker_command_aborts_before_worker_main_on_shared_worktree(monkeypatch):
    """`worker` from a linked worktree on the shared DB aborts before worker_main."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("aios.harness.worker.worker_main") as mock_worker_main:
        result = runner.invoke(app, ["worker"])

    mock_worker_main.assert_not_called()
    assert result.exit_code == 1
