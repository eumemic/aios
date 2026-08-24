"""Tests for ``aios usage consumers``."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app
from tests.unit.cli.conftest import MockedCli

runner = CliRunner()


def test_consumers_forwards_window_metric_and_limit(mocked_cli: MockedCli) -> None:
    mocked_cli.queue_response(
        httpx.Response(
            200,
            json={
                "metric": "total_tokens",
                "window_seconds": 3600,
                "coverage_started_at": "2026-08-23T00:00:00Z",
                "total_rate_per_hour": 0,
                "items": [],
            },
        )
    )
    result = runner.invoke(
        app,
        [
            "usage",
            "consumers",
            "--window-seconds",
            "3600",
            "--metric",
            "total_tokens",
            "--limit",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/usage/consumers"
    assert mocked_cli.captured.query == {
        "window_seconds": ["3600"],
        "metric": ["total_tokens"],
        "limit": ["7"],
    }
    assert '"metric": "total_tokens"' in result.output
