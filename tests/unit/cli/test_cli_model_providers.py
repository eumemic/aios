"""Tests for ``aios model-providers ...`` via the typer app."""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app
from tests.unit.cli.conftest import resource_response

runner = CliRunner()


def test_list_with_pagination(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [], "has_more": False}))
    result = runner.invoke(app, ["model-providers", "list", "--limit", "42"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.path == "/v1/model-providers"
    assert mocked_cli.captured.query.get("limit") == ["42"]


def test_list_filters_by_provider(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [], "has_more": False}))
    result = runner.invoke(app, ["model-providers", "list", "--provider", "anthropic"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.query.get("provider") == ["anthropic"]


def test_get(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=resource_response("model_provider")))
    result = runner.invoke(app, ["model-providers", "get", "mp_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/model-providers/mp_1"


def test_create(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(201, json=resource_response("model_provider", id="mp_new"))
    )
    result = runner.invoke(
        app,
        [
            "model-providers",
            "create",
            "--provider",
            "anthropic",
            "--api-key",
            "sk-real",
            "--api-base",
            "https://proxy.example",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/model-providers"
    assert mocked_cli.captured.body == {
        "provider": "anthropic",
        "api_key": "sk-real",
        "api_base": "https://proxy.example",
    }


def test_create_without_api_base(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json=resource_response("model_provider")))
    result = runner.invoke(
        app,
        ["model-providers", "create", "--provider", "openai", "--api-key", "sk-real"],
    )
    assert result.exit_code == 0, result.output
    # api_base omitted entirely (Unset), not sent as null.
    assert "api_base" not in mocked_cli.captured.body


def test_update_rotates_key_only(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=resource_response("model_provider")))
    result = runner.invoke(app, ["model-providers", "update", "mp_1", "--api-key", "sk-new"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "PUT"
    assert mocked_cli.captured.path == "/v1/model-providers/mp_1"
    assert mocked_cli.captured.body == {"api_key": "sk-new"}


def test_update_clears_api_base(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=resource_response("model_provider")))
    result = runner.invoke(app, ["model-providers", "update", "mp_1", "--clear-api-base"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body == {"api_base": None}


def test_update_sets_api_base(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=resource_response("model_provider")))
    result = runner.invoke(
        app,
        ["model-providers", "update", "mp_1", "--api-base", "https://new-proxy.example"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body == {"api_base": "https://new-proxy.example"}


def test_archive_is_delete(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["model-providers", "archive", "mp_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/model-providers/mp_1"
    assert "archived" in result.output
    assert "mp_1" in result.output
