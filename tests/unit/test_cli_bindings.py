"""Unit tests for ``aios bindings <verb>`` CLI (progresses #35 item 4).

The HTTP client layer is mocked; these tests focus on argv parsing,
request-body / query-param shaping, error propagation, and output
formatting.

Tests exercise ``run_async`` directly so pytest-asyncio owns the event
loop — same rationale as ``test_cli_connections.py``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.bindings import run_async


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


class TestListBindings:
    async def test_prints_json_array(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        payload = {
            "data": [
                {
                    "id": "cbn_01",
                    "connection_id": "conn_01",
                    "path": "group/abc",
                    "address": "signal/+15550001/group/abc",
                    "session_id": "sess_01",
                    "created_at": "2026-04-20T00:00:00Z",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "archived_at": None,
                    "notification_mode": "focal_candidate",
                }
            ],
            "has_more": False,
            "next_after": None,
        }
        client = _mock_async_client("get", _mock_response(200, payload))

        with patch("aios.cli.bindings.httpx.AsyncClient", return_value=client):
            rc = await run_async(["list"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "cbn_01" in out
        assert "signal/+15550001/group/abc" in out
        assert "sess_01" in out

    async def test_filters_by_session_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        payload = {"data": [], "has_more": False, "next_after": None}
        client = _mock_async_client("get", _mock_response(200, payload))

        with patch("aios.cli.bindings.httpx.AsyncClient", return_value=client):
            rc = await run_async(["list", "--session-id", "sess_target"])

        assert rc == 0
        client.get.assert_awaited_once()
        call = client.get.await_args
        # Accept either ?session_id=... on the URL, or passed via params=
        url = call.args[0]
        params = call.kwargs.get("params") or {}
        assert "session_id" in url or params.get("session_id") == "sess_target"

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(500, {"error": "boom"}))

        with patch("aios.cli.bindings.httpx.AsyncClient", return_value=client):
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


class TestCreateBinding:
    async def test_posts_expected_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "cbn_new",
            "connection_id": "conn_01",
            "path": "group/abc",
            "address": "signal/+15550001/group/abc",
            "session_id": "sess_01",
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
            "notification_mode": "focal_candidate",
        }
        client = _mock_async_client("post", _mock_response(201, created))

        with patch("aios.cli.bindings.httpx.AsyncClient", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--address",
                    "signal/+15550001/group/abc",
                    "--session-id",
                    "sess_01",
                ]
            )

        assert rc == 0
        client.post.assert_awaited_once()
        call = client.post.await_args
        assert call.args[0].endswith("/v1/channel-bindings")
        assert call.kwargs["json"] == {
            "address": "signal/+15550001/group/abc",
            "session_id": "sess_01",
        }

    async def test_missing_required_flag_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["create", "--address", "signal/+15550001/group/abc"])
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
