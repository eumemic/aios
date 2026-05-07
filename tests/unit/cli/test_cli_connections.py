"""Tests for ``aios connections ...`` via the typer app.

Connector redesign (#200): create is detached-only; mode binding moves
to ``attach`` / ``configure-per-chat`` subcommands.  No more
``--mcp-url`` / ``--vault-id`` / inbound subcommand.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app
from tests.unit.cli.conftest import connection_response

runner = CliRunner()


def test_list_with_pagination(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": False, "next_after": None})
    )
    result = runner.invoke(app, ["connections", "list", "--limit", "25", "--after", "conn_x"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/connections"
    assert mocked_cli.captured.query.get("limit") == ["25"]
    assert mocked_cli.captured.query.get("after") == ["conn_x"]


def test_list_filters_pass_through(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": False, "next_after": None})
    )
    runner.invoke(
        app,
        [
            "connections",
            "list",
            "--connector",
            "signal",
            "--session-id",
            "sess_1",
        ],
    )
    assert mocked_cli.captured.query.get("connector") == ["signal"]
    assert mocked_cli.captured.query.get("session_id") == ["sess_1"]


def test_create_ergonomic(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json=connection_response(id="conn_new")))
    result = runner.invoke(
        app,
        [
            "connections",
            "create",
            "--connector",
            "signal",
            "--account",
            "acct-123",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections"
    assert mocked_cli.captured.body == {
        "connector": "signal",
        "account": "acct-123",
    }


def test_create_ergonomic_with_metadata_json(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json=connection_response(id="conn_new")))
    result = runner.invoke(
        app,
        [
            "connections",
            "create",
            "--connector",
            "signal",
            "--account",
            "acct-123",
            "--metadata-json",
            '{"region": "us-east"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body["metadata"] == {"region": "us-east"}


def test_create_missing_ergonomic_flag(mocked_cli):
    result = runner.invoke(
        app,
        ["connections", "create", "--connector", "signal"],
    )
    assert result.exit_code == 64
    assert "--account" in result.output


def test_create_rejects_mixed_sources(mocked_cli):
    result = runner.invoke(
        app,
        [
            "connections",
            "create",
            "--connector",
            "signal",
            "--account",
            "acct-1",
            "--data",
            "{}",
        ],
    )
    assert result.exit_code == 64
    assert "not both" in result.output


def test_get_resource_path(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "conn_01"}))
    runner.invoke(app, ["connections", "get", "conn_01"])
    assert mocked_cli.captured.path == "/v1/connections/conn_01"


def test_archive_uses_delete(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["connections", "archive", "conn_01"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/connections/conn_01"
    assert "archived" in result.output
    assert "conn_01" in result.output


def test_attach_posts_session_id(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=connection_response()))
    result = runner.invoke(
        app,
        ["connections", "attach", "conn_01", "--session-id", "sess_1"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/attach"
    assert mocked_cli.captured.body == {"session_id": "sess_1"}


def test_detach_posts_no_body(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "conn_01"}))
    runner.invoke(app, ["connections", "detach", "conn_01"])
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/detach"


def test_configure_per_chat_passes_template(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "conn_01"}))
    runner.invoke(
        app,
        ["connections", "configure-per-chat", "conn_01", "--template", "stpl_1"],
    )
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/configure-per-chat"
    assert mocked_cli.captured.body == {"session_template_id": "stpl_1"}


def test_unconfigure_posts_no_body(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "conn_01"}))
    runner.invoke(app, ["connections", "unconfigure", "conn_01"])
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/unconfigure"
