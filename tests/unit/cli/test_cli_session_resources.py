"""Tests for ``aios sessions resources …`` (granular add/remove/list/rotate, #270).

Each test points the CLI at an ``httpx.MockTransport`` (the ``mocked_cli``
fixture) and asserts the request shape the sub-app builds — method, path,
and body — for the four commands that mirror the triggers sub-app.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()

_GH_ECHO = {
    "type": "github_repository",
    "id": "ghrepo_01HQR2K7VXBZ9MNPL3WYCT8F00",
    "url": "https://x/a.git",
    "mount_path": "/mnt/a",
    "created_at": "2024-01-01T00:00:00+00:00",
    "updated_at": "2024-01-01T00:00:00+00:00",
}
_MEM_ECHO = {
    "type": "memory_store",
    "memory_store_id": "memstore_01HQR2K7VXBZ9MNPL3WYCT8F00",
    "access": "read_write",
    "instructions": "",
    "name": "notes",
    "description": "d",
    "mount_path": "/mnt/memory/notes",
}


def test_resources_list_gets_collection(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": []}))
    result = runner.invoke(app, ["sessions", "resources", "list", "sess_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/resources"


def test_resources_add_memory_posts_collection(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json=_MEM_ECHO))
    result = runner.invoke(
        app,
        ["sessions", "resources", "add", "sess_1", "--memory-store-id", "memstore_x"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/resources"
    assert mocked_cli.captured.body == {
        "type": "memory_store",
        "memory_store_id": "memstore_x",
        "access": "read_write",
        "instructions": "",
    }


def test_resources_add_github_posts_collection(mocked_cli):
    mocked_cli.queue_response(httpx.Response(201, json=_GH_ECHO))
    result = runner.invoke(
        app,
        [
            "sessions",
            "resources",
            "add",
            "sess_1",
            "--url",
            "https://x/a.git",
            "--mount-path",
            "/mnt/a",
            "--token",
            "ghp_secret",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/resources"
    assert mocked_cli.captured.body == {
        "type": "github_repository",
        "url": "https://x/a.git",
        "mount_path": "/mnt/a",
        "authorization_token": "ghp_secret",
    }


def test_resources_add_github_requires_all_three(mocked_cli):
    result = runner.invoke(
        app, ["sessions", "resources", "add", "sess_1", "--url", "https://x/a.git"]
    )
    assert result.exit_code == 64
    assert mocked_cli.captured.method == ""  # no HTTP call


def test_resources_add_requires_a_selector(mocked_cli):
    result = runner.invoke(app, ["sessions", "resources", "add", "sess_1"])
    assert result.exit_code == 64
    assert mocked_cli.captured.method == ""


def test_resources_remove_deletes_by_id(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["sessions", "resources", "remove", "sess_1", "ghrepo_x"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/resources/ghrepo_x"


def test_resources_rotate_puts_by_id(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=_GH_ECHO))
    result = runner.invoke(
        app,
        ["sessions", "resources", "rotate", "sess_1", "ghrepo_x", "--token", "ghp_new"],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "PUT"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/resources/ghrepo_x"
    assert mocked_cli.captured.body == {"authorization_token": "ghp_new"}


def test_resources_rotate_requires_token_or_payload(mocked_cli):
    result = runner.invoke(app, ["sessions", "resources", "rotate", "sess_1", "ghrepo_x"])
    assert result.exit_code == 64
    assert mocked_cli.captured.method == ""
