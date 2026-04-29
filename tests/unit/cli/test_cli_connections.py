"""Tests for ``aios connections ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

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


def test_create_ergonomic(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "conn_new"}))
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


def test_create_ergonomic_with_legacy_mcp_projection(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "conn_new"}))
    result = runner.invoke(
        app,
        [
            "connections",
            "create",
            "--connector",
            "signal",
            "--account",
            "acct-123",
            "--mcp-url",
            "http://mcp.example:9000",
            "--vault-id",
            "vlt_1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections"
    assert mocked_cli.captured.body == {
        "connector": "signal",
        "account": "acct-123",
        "mcp_url": "http://mcp.example:9000",
        "vault_id": "vlt_1",
    }


def test_create_ergonomic_with_metadata_json(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "conn_new"}))
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
    # Success line on stdout so scripts + humans get a visible ack.
    assert "archived" in result.output
    assert "conn_01" in result.output


def test_inbound_posts_to_messages(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(201, json={"session_id": "sess_new", "event_id": "evt_1"})
    )
    result = runner.invoke(
        app,
        [
            "connections",
            "inbound",
            "conn_01",
            "--path",
            "dm/abc",
            "--content",
            "hello",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/messages"
    assert mocked_cli.captured.body == {"path": "dm/abc", "content": "hello"}
