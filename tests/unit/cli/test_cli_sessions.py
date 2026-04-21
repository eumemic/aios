"""Tests for ``aios sessions ...`` destructive verbs via the typer app.

Focuses on the ``delete`` guard (hard-delete requires ``--yes``) introduced
to line ``sessions delete`` up with the other hard-delete commands. Happy-
path CRUD is exercised elsewhere.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_delete_refuses_without_yes_and_makes_no_request(mocked_cli):
    result = runner.invoke(app, ["sessions", "delete", "sess_1"])
    assert result.exit_code == 2
    assert "--yes" in result.output
    assert "archive" in result.output  # remind about the soft alternative
    assert mocked_cli.captured.method == ""  # no HTTP call was made


def test_delete_is_hard_delete_with_yes(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["sessions", "delete", "sess_1", "--yes"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1"
