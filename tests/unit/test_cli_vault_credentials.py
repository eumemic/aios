"""Unit tests for ``aios vault-credentials <verb>`` CLI (progresses #35 item 4).

Scope: ``list`` only — credential *creation* has an ``auth_type``-
dependent schema branch with several ``SecretStr`` fields; that
warrants its own PR with a deliberate CLI flag design.
"""

from __future__ import annotations

import json
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
