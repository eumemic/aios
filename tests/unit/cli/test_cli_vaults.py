"""Tests for ``aios vaults ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_list_with_pagination(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [], "has_more": False}))
    result = runner.invoke(app, ["vaults", "list", "--limit", "42"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.path == "/v1/vaults"
    assert mocked_cli.captured.query.get("limit") == ["42"]


def test_create_ergonomic_with_display_name(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "vlt_new"}))
    result = runner.invoke(
        app,
        ["vaults", "create", "--display-name", "prod-secrets"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/vaults"
    assert mocked_cli.captured.body == {"display_name": "prod-secrets"}


def test_create_ergonomic_with_metadata(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "vlt_new"}))
    runner.invoke(
        app,
        [
            "vaults",
            "create",
            "--display-name",
            "prod",
            "--metadata-json",
            '{"env": "prod"}',
        ],
    )
    assert mocked_cli.captured.body == {
        "display_name": "prod",
        "metadata": {"env": "prod"},
    }


def test_create_requires_display_name_when_ergonomic(mocked_cli):
    result = runner.invoke(
        app,
        ["vaults", "create", "--metadata-json", '{"env":"x"}'],
    )
    assert result.exit_code == 64
    assert "--display-name" in result.output


def test_archive_uses_post(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "vlt_1", "archived_at": "t"}))
    result = runner.invoke(app, ["vaults", "archive", "vlt_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/archive"


def test_delete_is_hard_delete_with_yes(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["vaults", "delete", "vlt_1", "--yes"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1"
    assert "deleted" in result.output
    assert "vlt_1" in result.output


def test_delete_refuses_without_yes_and_makes_no_request(mocked_cli):
    result = runner.invoke(app, ["vaults", "delete", "vlt_1"])
    assert result.exit_code == 2
    assert "--yes" in result.output
    assert "archive" in result.output  # remind about the soft alternative
    assert mocked_cli.captured.method == ""  # no HTTP call was made
