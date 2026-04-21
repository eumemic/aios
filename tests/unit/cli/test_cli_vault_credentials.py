"""Tests for ``aios vaults credentials ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_list_scoped_to_vault(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [], "has_more": False}))
    result = runner.invoke(
        app,
        ["vaults", "credentials", "list", "vlt_1", "--limit", "10"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/credentials"
    assert mocked_cli.captured.query.get("limit") == ["10"]


def test_get_resource_path(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "cred_1"}))
    runner.invoke(app, ["vaults", "credentials", "get", "vlt_1", "cred_1"])
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/credentials/cred_1"


def test_create_via_body_file(mocked_cli, tmp_path):
    body = tmp_path / "cred.json"
    body.write_text(
        '{"display_name": "my-creds", "mcp_server_url": "http://x",'
        ' "auth_type": "static_bearer", "token": "t"}'
    )
    mocked_cli.queue_response(httpx.Response(201, json={"id": "cred_new"}))
    result = runner.invoke(
        app,
        [
            "vaults",
            "credentials",
            "create",
            "vlt_1",
            "--body-file",
            str(body),
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/credentials"
    assert mocked_cli.captured.body["display_name"] == "my-creds"
    assert mocked_cli.captured.body["token"] == "t"


def test_create_via_file_alias(mocked_cli, tmp_path):
    body = tmp_path / "cred.json"
    body.write_text(
        '{"display_name": "x", "mcp_server_url": "http://y", "auth_type": "static_bearer", "token": "t"}'
    )
    mocked_cli.queue_response(httpx.Response(201, json={"id": "cred_new"}))
    result = runner.invoke(
        app,
        ["vaults", "credentials", "create", "vlt_1", "--file", str(body)],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body["display_name"] == "x"


def test_archive_uses_post(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "cred_1"}))
    result = runner.invoke(app, ["vaults", "credentials", "archive", "vlt_1", "cred_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/credentials/cred_1/archive"


def test_delete_is_hard_delete_with_yes(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["vaults", "credentials", "delete", "vlt_1", "cred_1", "--yes"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/vaults/vlt_1/credentials/cred_1"


def test_delete_refuses_without_yes_and_makes_no_request(mocked_cli):
    result = runner.invoke(app, ["vaults", "credentials", "delete", "vlt_1", "cred_1"])
    assert result.exit_code == 2
    assert "irreversible" in result.output
    assert mocked_cli.captured.method == ""  # no HTTP call was made
