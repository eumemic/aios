"""Unit tests for ``aios vaults <verb>`` CLI (progresses #35 item 4).

Vaults have a nested ``metadata`` object in ``VaultCreate``; CLI
accepts it as ``--metadata-json`` (optional, default ``{}``), same
shape as ``aios rules --session-params-json``.

Tests exercise ``run_async`` directly and patch ``async_client``
through the per-module binding introduced in #105.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.vaults import run_async


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


class TestListVaults:
    async def test_prints_json_array(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        payload = {
            "data": [
                {
                    "id": "vlt_01",
                    "display_name": "Signal secrets",
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

        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(["list"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "vlt_01" in out
        assert "Signal secrets" in out

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(500, {"error": "boom"}))

        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(["list"])

        assert rc != 0
        assert "500" in capsys.readouterr().err

    async def test_missing_api_key_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("AIOS_API_KEY", raising=False)
        monkeypatch.setenv("AIOS_API_URL", "http://test.server:8090")
        rc = await run_async(["list"])
        assert rc != 0
        assert "AIOS_API_KEY" in capsys.readouterr().err


class TestCreateVault:
    async def test_posts_expected_body_with_default_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "vlt_new",
            "display_name": "My vault",
            "metadata": {},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
        }
        client = _mock_async_client("post", _mock_response(201, created))

        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(["create", "--display-name", "My vault"])

        assert rc == 0
        client.post.assert_awaited_once()
        call = client.post.await_args
        assert call.args[0].endswith("/v1/vaults")
        assert call.kwargs["json"] == {"display_name": "My vault", "metadata": {}}

    async def test_posts_with_metadata_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "vlt_new",
            "display_name": "Tagged vault",
            "metadata": {"env": "prod", "owner": "ops"},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
        }
        client = _mock_async_client("post", _mock_response(201, created))

        md = json.dumps({"env": "prod", "owner": "ops"})
        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--display-name",
                    "Tagged vault",
                    "--metadata-json",
                    md,
                ]
            )

        assert rc == 0
        call = client.post.await_args
        assert call.kwargs["json"] == {
            "display_name": "Tagged vault",
            "metadata": {"env": "prod", "owner": "ops"},
        }

    async def test_invalid_metadata_json_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["create", "--display-name", "x", "--metadata-json", "not-json{{"])
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "json" in err or "metadata" in err

    async def test_missing_required_flag_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["create"])  # missing --display-name
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


class TestGetVault:
    async def test_prints_single_resource(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        vault = {
            "id": "vlt_01",
            "display_name": "Signal secrets",
            "metadata": {},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
        }
        client = _mock_async_client("get", _mock_response(200, vault))

        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(["get", "vlt_01"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "vlt_01" in out
        assert "Signal secrets" in out
        assert client.get.await_args.args[0].endswith("/v1/vaults/vlt_01")

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(404, {"error": "not found"}))

        with patch("aios.cli.vaults.async_client", return_value=client):
            rc = await run_async(["get", "vlt_missing"])

        assert rc != 0
        assert "404" in capsys.readouterr().err


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
