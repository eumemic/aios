"""Tests for ``aios status`` — reachability/auth checks.

Regression: issue #126 P2 — when AIOS_URL points at something returning
HTML, ``status`` must not dump a raw ``json.decoder.JSONDecodeError``
traceback. It should report a friendly "non-JSON response" line and exit
non-zero.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_status_non_json_health_response_is_friendly(mocked_cli):
    """AIOS_URL points at an HTML landing page. The status command should
    degrade gracefully instead of raising a JSON decode traceback."""
    mocked_cli.queue_response(
        httpx.Response(
            200,
            content=b"<html>hi</html>",
            headers={"content-type": "text/html"},
        )
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "health" in result.output
    assert "non-JSON" in result.output
    assert "traceback" not in result.output.lower()


def test_status_non_json_health_response_json_format(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(
            200,
            content=b"<html>hi</html>",
            headers={"content-type": "text/html"},
        )
    )
    result = runner.invoke(app, ["--format", "json", "status"])
    assert result.exit_code == 1
    assert "non_json_response" in result.output


def test_status_happy_path_health_only(mocked_cli, monkeypatch):
    """Without an API key, status only probes /health and returns 0 on ok."""
    monkeypatch.delenv("AIOS_API_KEY", raising=False)
    mocked_cli.queue_response(httpx.Response(200, json={"ok": True}))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "health" in result.output
