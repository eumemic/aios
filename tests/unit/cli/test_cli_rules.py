"""Tests for ``aios rules ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_list_scoped_to_connection(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [], "has_more": False}))
    result = runner.invoke(app, ["rules", "list", "conn_01", "--limit", "5"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.path == "/v1/connections/conn_01/routing-rules"
    assert mocked_cli.captured.query.get("limit") == ["5"]


def test_create_ergonomic_sends_prefix_target_and_empty_session_params(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "rule_new"}))
    result = runner.invoke(
        app,
        [
            "rules",
            "create",
            "conn_01",
            "--prefix",
            "dm/",
            "--target",
            "agent:claude",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/routing-rules"
    assert mocked_cli.captured.body == {
        "prefix": "dm/",
        "target": "agent:claude",
        "session_params": {},
    }


def test_create_ergonomic_with_session_params_json(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "rule_new"}))
    runner.invoke(
        app,
        [
            "rules",
            "create",
            "conn_01",
            "--prefix",
            "",
            "--target",
            "session:sess_1",
            "--session-params-json",
            '{"title": "fallback"}',
        ],
    )
    assert mocked_cli.captured.body == {
        "prefix": "",
        "target": "session:sess_1",
        "session_params": {"title": "fallback"},
    }


def test_create_requires_both_ergonomic_flags(mocked_cli):
    result = runner.invoke(
        app,
        ["rules", "create", "conn_01", "--prefix", "dm/"],
    )
    assert result.exit_code == 64
    assert "--target" in result.output


def test_create_rejects_mixed_sources(mocked_cli):
    result = runner.invoke(
        app,
        ["rules", "create", "conn_01", "--prefix", "x", "--target", "y", "--data", "{}"],
    )
    assert result.exit_code == 64
    assert "not both" in result.output


def test_get_scoped_path(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "rule_01"}))
    runner.invoke(app, ["rules", "get", "conn_01", "rule_01"])
    assert mocked_cli.captured.path == "/v1/connections/conn_01/routing-rules/rule_01"


def test_archive_uses_delete(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["rules", "archive", "conn_01", "rule_01"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/connections/conn_01/routing-rules/rule_01"


def test_update_via_file(mocked_cli, tmp_path):
    body = tmp_path / "upd.json"
    body.write_text('{"target": "agent:other"}')
    mocked_cli.queue_response(httpx.Response(200, json={"id": "rule_01"}))
    result = runner.invoke(
        app,
        ["rules", "update", "conn_01", "rule_01", "--file", str(body)],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "PUT"
    assert mocked_cli.captured.body == {"target": "agent:other"}
