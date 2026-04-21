"""Unit tests for ``aios rules <verb>`` CLI (progresses #35 item 4).

Routing rules are per-connection (nested resource), so both verbs take
``--connection-id``.  ``create`` accepts ``--session-params-json`` for
the nested ``SessionParams`` block; default is an empty object which
the API treats as a session_params-less rule.

Tests exercise ``run_async`` directly — same pytest-asyncio rationale
as ``test_cli_connections.py``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.cli.rules import run_async


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


class TestListRules:
    async def test_prints_json_array_and_hits_nested_route(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        payload = {
            "data": [
                {
                    "id": "rul_01",
                    "connection_id": "conn_01",
                    "prefix": "group",
                    "target": "agent:claude-sonnet-4",
                    "session_params": {
                        "environment_id": None,
                        "vault_ids": [],
                        "title": None,
                        "metadata": {},
                    },
                    "created_at": "2026-04-20T00:00:00Z",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "archived_at": None,
                }
            ],
            "has_more": False,
            "next_after": None,
        }
        client = _mock_async_client("get", _mock_response(200, payload))

        with patch("aios.cli.rules.async_client", return_value=client):
            rc = await run_async(["list", "--connection-id", "conn_01"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "rul_01" in out
        assert "group" in out
        # Nested route under the connection id
        call = client.get.await_args
        assert call.args[0].endswith("/v1/connections/conn_01/routing-rules")

    async def test_http_error_returns_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        client = _mock_async_client("get", _mock_response(500, {"error": "boom"}))

        with patch("aios.cli.rules.async_client", return_value=client):
            rc = await run_async(["list", "--connection-id", "conn_01"])

        assert rc != 0
        assert "500" in capsys.readouterr().err

    async def test_missing_api_key_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("AIOS_API_KEY", raising=False)
        monkeypatch.setenv("AIOS_API_URL", "http://test.server:8090")
        rc = await run_async(["list", "--connection-id", "conn_01"])
        assert rc != 0
        assert "AIOS_API_KEY" in capsys.readouterr().err

    async def test_missing_connection_id_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(["list"])
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()


class TestCreateRule:
    async def test_posts_expected_body_with_default_session_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "rul_new",
            "connection_id": "conn_01",
            "prefix": "group",
            "target": "agent:claude-sonnet-4",
            "session_params": {
                "environment_id": None,
                "vault_ids": [],
                "title": None,
                "metadata": {},
            },
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
        }
        client = _mock_async_client("post", _mock_response(201, created))

        with patch("aios.cli.rules.async_client", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--connection-id",
                    "conn_01",
                    "--prefix",
                    "group",
                    "--target",
                    "agent:claude-sonnet-4",
                ]
            )

        assert rc == 0
        client.post.assert_awaited_once()
        call = client.post.await_args
        assert call.args[0].endswith("/v1/connections/conn_01/routing-rules")
        assert call.kwargs["json"] == {
            "prefix": "group",
            "target": "agent:claude-sonnet-4",
            "session_params": {},
        }

    async def test_posts_with_session_params_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch)
        created = {
            "id": "rul_new",
            "connection_id": "conn_01",
            "prefix": "",
            "target": "agent:claude-sonnet-4",
            "session_params": {
                "environment_id": "env_01",
                "vault_ids": ["vlt_01"],
                "title": "chat {address}",
                "metadata": {},
            },
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
            "archived_at": None,
        }
        client = _mock_async_client("post", _mock_response(201, created))

        sp_json = json.dumps(
            {
                "environment_id": "env_01",
                "vault_ids": ["vlt_01"],
                "title": "chat {address}",
            }
        )
        with patch("aios.cli.rules.async_client", return_value=client):
            rc = await run_async(
                [
                    "create",
                    "--connection-id",
                    "conn_01",
                    "--prefix",
                    "",
                    "--target",
                    "agent:claude-sonnet-4",
                    "--session-params-json",
                    sp_json,
                ]
            )

        assert rc == 0
        call = client.post.await_args
        assert call.kwargs["json"] == {
            "prefix": "",
            "target": "agent:claude-sonnet-4",
            "session_params": {
                "environment_id": "env_01",
                "vault_ids": ["vlt_01"],
                "title": "chat {address}",
            },
        }

    async def test_missing_required_flag_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(
            ["create", "--connection-id", "conn_01", "--prefix", "group"]
        )  # missing --target
        assert rc != 0
        assert "required" in capsys.readouterr().err.lower()

    async def test_invalid_session_params_json_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _setup_env(monkeypatch)
        rc = await run_async(
            [
                "create",
                "--connection-id",
                "conn_01",
                "--prefix",
                "group",
                "--target",
                "agent:claude-sonnet-4",
                "--session-params-json",
                "not-json{{",
            ]
        )
        assert rc != 0
        err = capsys.readouterr().err.lower()
        assert "json" in err or "session-params" in err


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
