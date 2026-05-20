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

    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["api"])

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("proxy_headers") is True, "uvicorn.run must be called with proxy_headers=True"
    assert kwargs.get("forwarded_allow_ips") == "*", (
        'uvicorn.run must be called with forwarded_allow_ips="*"'
    )
