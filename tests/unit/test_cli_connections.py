"""Unit tests for ``aios connections <verb>`` CLI (progresses #35 item 4).

The HTTP client layer is mocked; these tests focus on argv parsing,
request-body shaping, error propagation, and output formatting.

Tests exercise ``run_async`` directly so pytest-asyncio owns the event
loop — avoids the ``asyncio.run`` → closed-loop interaction with other
tests in the suite that use ``asyncio.get_event_loop()``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.connections import run_async


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


class TestListConnections:
    async def test_prints_json_array(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        payload = {
            "data": [
                {
                    "id": "conn_01",
                    "connector": "signal",
                    "account": "+15550001",
                    "mcp_url": "http://m",
                    "vault_id": "vlt_01",
                    "archived_at": None,
                }
            ],
            "has_more": False,
            "next_after": None,
        }
        client = _mock_async_client("get", _mock_response(200, payload))

        with patch("aios.cli.connections.async_client", return_value=client):
            rc = await run_async(["list"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "conn_01" in out
        assert "signal" in out
        assert "+15550001" in out

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(500, {"error": "boom"}))

        with patch("aios.cli.connections.async_client", return_value=client):
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


class TestCreateConnection:
    async def test_posts_expected_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "conn_new",
            "connector": "signal",
            "account": "+15550001",
            "mcp_url": "http://m",
            "vault_id": "vlt_01",
            "archived_at": None,
        }
        client = _mock_async_client("post", _mock_response(201, created))

        with patch("aios.cli.connections.async_client", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--connector",
                    "signal",
                    "--account",
                    "+15550001",
                    "--mcp-url",
                    "http://m",
                    "--vault-id",
                    "vlt_01",
                ]
            )

        assert rc == 0
        client.post.assert_awaited_once()
        call = client.post.await_args
        assert call.args[0].endswith("/v1/connections")
        assert call.kwargs["json"] == {
            "connector": "signal",
            "account": "+15550001",
            "mcp_url": "http://m",
            "vault_id": "vlt_01",
        }

    async def test_missing_required_flag_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["create", "--connector", "signal"])
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
