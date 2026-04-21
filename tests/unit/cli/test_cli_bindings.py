"""Tests for ``aios bindings ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_list_sends_filters_and_pagination(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": False, "next_after": None})
    )
    result = runner.invoke(
        app,
        ["bindings", "list", "--session-id", "sess_target", "--limit", "10", "--after", "cbn_x"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/channel-bindings"
    assert mocked_cli.captured.query.get("session_id") == ["sess_target"]
    assert mocked_cli.captured.query.get("limit") == ["10"]
    assert mocked_cli.captured.query.get("after") == ["cbn_x"]


def test_list_prints_rows(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "cbn_01",
                        "address": "signal/+15550001/group/abc",
                        "session_id": "sess_01",
                        "created_at": "2026-04-20T00:00:00Z",
                    }
                ],
                "has_more": False,
            },
        )
    )
    result = runner.invoke(app, ["bindings", "list"])
    assert result.exit_code == 0, result.output
    assert "cbn_01" in result.output
    assert "signal/+15550001/group/abc" in result.output


def test_create_from_ergonomic_flags(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "cbn_new", "session_id": "sess_01"}))
    result = runner.invoke(
        app,
        [
            "bindings",
            "create",
            "--address",
            "signal/+1/group/abc",
            "--session-id",
            "sess_01",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/channel-bindings"
    assert mocked_cli.captured.body == {
        "address": "signal/+1/group/abc",
        "session_id": "sess_01",
    }


def test_create_from_data_json(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json={"id": "cbn_new"}))
    result = runner.invoke(
        app,
        [
            "bindings",
            "create",
            "--data",
            '{"address": "x/y/z", "session_id": "sess_1"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body == {"address": "x/y/z", "session_id": "sess_1"}


def test_create_rejects_both_ergonomic_and_file(mocked_cli):
    result = runner.invoke(
        app,
        [
            "bindings",
            "create",
            "--address",
            "x",
            "--session-id",
            "s",
            "--data",
            "{}",
        ],
    )
    assert result.exit_code == 64
    assert "not both" in result.output


def test_get_hits_resource_path(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"id": "cbn_01"}))
    result = runner.invoke(app, ["bindings", "get", "cbn_01"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.path == "/v1/channel-bindings/cbn_01"


def test_archive_uses_delete(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["bindings", "archive", "cbn_01"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/channel-bindings/cbn_01"
    # Success line on stdout so scripts + humans get a visible ack.
    assert "archived" in result.output
    assert "cbn_01" in result.output


def test_http_error_nonzero_exit(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(500, json={"error": {"type": "internal_error", "message": "boom"}})
    )
    result = runner.invoke(app, ["bindings", "list"])
    assert result.exit_code == 1
    assert "boom" in result.output
