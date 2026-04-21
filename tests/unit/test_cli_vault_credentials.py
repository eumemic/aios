"""Unit tests for ``aios vault-credentials <verb>`` CLI (progresses #35 item 4).

Covers ``list`` and ``create`` (#108 + this PR).  ``create`` takes
``--body-file`` (or ``-`` for stdin) rather than per-field flags: the
credential schema branches on ``auth_type`` with several ``SecretStr``
fields, and passing secrets as shell args leaks them into history.
``--body-file`` is the ``kubectl create -f`` pattern — operator hands
the CLI a JSON file they prepared locally.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.vault_credentials import run_async


def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIOS_API_KEY", "test-key")
    monkeypatch.setenv("AIOS_API_URL", "http://test.server:8090")


def _mock_response(status_code: int, body: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body)
    resp.text = json.dumps(body)
    return resp


def _mock_async_client(method: str, response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    setattr(client, method, AsyncMock(return_value=response))
    return client


class TestListVaultCredentials:
    async def test_prints_json_array_and_hits_nested_route(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        payload = {
            "data": [
                {
                    "id": "vcr_01",
                    "vault_id": "vlt_01",
                    "display_name": "Signal MCP bearer",
                    "mcp_server_url": "http://m",
                    "auth_type": "static_bearer",
                    "metadata": {},
                    "created_at": "2026-04-20T00:00:00Z",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "archived_at": None,
                }
            ],
            "has_more": False,
            "next_after": None,
        }
        client = _mock_async_client("get", _mock_response(200, payload))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["list", "--vault-id", "vlt_01"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "vcr_01" in out
        assert "Signal MCP bearer" in out
        # Nested route under the vault id
        call = client.get.await_args
        assert call.args[0].endswith("/v1/vaults/vlt_01/credentials")

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(500, {"error": "boom"}))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["list", "--vault-id", "vlt_01"])

        assert rc != 0
        assert "500" in capsys.readouterr().err

    async def test_missing_api_key_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("AIOS_API_KEY", raising=False)
        monkeypatch.setenv("AIOS_API_URL", "http://test.server:8090")
        rc = await run_async(["list", "--vault-id", "vlt_01"])
        assert rc != 0
        assert "AIOS_API_KEY" in capsys.readouterr().err

    async def test_missing_vault_id_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["list"])
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


_CREATED_CREDENTIAL: dict[str, Any] = {
    "id": "vcr_new",
    "vault_id": "vlt_01",
    "display_name": "Signal MCP bearer",
    "mcp_server_url": "http://mcp.signal.local",
    "auth_type": "static_bearer",
    "metadata": {},
    "created_at": "2026-04-20T00:00:00Z",
    "updated_at": "2026-04-20T00:00:00Z",
    "archived_at": None,
}


class TestCreateVaultCredential:
    async def test_posts_body_read_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch)
        body = {
            "display_name": "Signal MCP bearer",
            "mcp_server_url": "http://mcp.signal.local",
            "auth_type": "static_bearer",
            "bearer_token": "tok_secret",
        }
        body_file = tmp_path / "cred.json"
        body_file.write_text(json.dumps(body))
        client = _mock_async_client("post", _mock_response(201, _CREATED_CREDENTIAL))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--vault-id",
                    "vlt_01",
                    "--body-file",
                    str(body_file),
                ]
            )

        assert rc == 0
        client.post.assert_awaited_once()
        call = client.post.await_args
        assert call.args[0].endswith("/v1/vaults/vlt_01/credentials")
        assert call.kwargs["json"] == body

    async def test_posts_body_read_from_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        body = {
            "display_name": "piped",
            "mcp_server_url": "http://m",
            "auth_type": "static_bearer",
            "bearer_token": "tok_secret",
        }
        client = _mock_async_client("post", _mock_response(201, _CREATED_CREDENTIAL))
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(body)))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["create", "--vault-id", "vlt_01", "--body-file", "-"])

        assert rc == 0
        assert client.post.await_args.kwargs["json"] == body

    async def test_missing_body_file_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(
            [
                "create",
                "--vault-id",
                "vlt_01",
                "--body-file",
                "/nonexistent/path/cred.json",
            ]
        )
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "body-file" in err or "no such" in err or "cannot" in err

    async def test_invalid_json_body_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        body_file = tmp_path / "bad.json"
        body_file.write_text("not-json{{")
        rc = await run_async(["create", "--vault-id", "vlt_01", "--body-file", str(body_file)])
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "json" in err

    async def test_non_object_json_body_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        body_file = tmp_path / "arr.json"
        body_file.write_text("[]")
        rc = await run_async(["create", "--vault-id", "vlt_01", "--body-file", str(body_file)])
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "object" in err or "json" in err

    async def test_missing_required_flag_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["create", "--vault-id", "vlt_01"])  # no --body-file
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


class TestGetVaultCredential:
    async def test_prints_single_resource(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(200, _CREATED_CREDENTIAL))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["get", "vcr_new", "--vault-id", "vlt_01"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "vcr_new" in out
        assert client.get.await_args.args[0].endswith("/v1/vaults/vlt_01/credentials/vcr_new")

    async def test_missing_vault_id_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["get", "vcr_01"])
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


class TestArchiveVaultCredential:
    async def test_archives_via_post_archive_endpoint_and_prints_resource(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        archived = {**_CREATED_CREDENTIAL, "archived_at": "2026-04-20T00:01:00Z"}
        client = _mock_async_client("post", _mock_response(200, archived))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["archive", "vcr_new", "--vault-id", "vlt_01"])

        assert rc == 0
        assert client.post.await_args.args[0].endswith(
            "/v1/vaults/vlt_01/credentials/vcr_new/archive"
        )
        out = capsys.readouterr().out
        assert "vcr_new" in out
        assert "archived_at" in out


class TestUpdateVaultCredential:
    async def test_puts_body_read_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _setup_env(monkeypatch)
        body = {"display_name": "renamed", "token": "tok_rotated"}
        body_file = tmp_path / "update.json"
        body_file.write_text(json.dumps(body))
        client = _mock_async_client("put", _mock_response(200, _CREATED_CREDENTIAL))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(
                [
                    "update",
                    "vcr_new",
                    "--vault-id",
                    "vlt_01",
                    "--body-file",
                    str(body_file),
                ]
            )

        assert rc == 0
        call = client.put.await_args
        assert call.args[0].endswith("/v1/vaults/vlt_01/credentials/vcr_new")
        assert call.kwargs["json"] == body

    async def test_missing_vault_id_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["update", "vcr_01", "--body-file", "/tmp/anything"])
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


class TestDeleteVaultCredential:
    async def test_deletes_when_yes_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("delete", _mock_response(204, None))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["delete", "vcr_01", "--vault-id", "vlt_01", "--yes"])

        assert rc == 0
        assert client.delete.await_args.args[0].endswith("/v1/vaults/vlt_01/credentials/vcr_01")

    async def test_refuses_without_yes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("delete", _mock_response(204, None))

        with patch("aios.cli.vault_credentials.async_client", return_value=client):
            rc = await run_async(["delete", "vcr_01", "--vault-id", "vlt_01"])

        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "--yes" in err or "confirm" in err
        client.delete.assert_not_awaited()


class TestDispatch:
    async def test_unknown_verb_prints_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = await run_async(["bogus-verb"])
        assert rc == 2
        assert "usage" in capsys.readouterr().err.lower()

    async def test_no_verb_prints_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = await run_async([])
        assert rc == 2
        assert "usage" in capsys.readouterr().err.lower()

    async def test_help_prints_usage_and_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rc = await run_async(["--help"])
        assert rc == 0
