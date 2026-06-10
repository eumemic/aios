"""Tests for ``aios accounts bootstrap`` via the typer app.

The bootstrap command mints the root account + its first API key on a
fresh DB. Its defining behaviour: the ``AIOS_BOOTSTRAP_TOKEN`` env var is
sent as the ``Authorization: Bearer`` header (NOT the client api_key),
because the endpoint is gated by that token, and the minted plaintext key
must reach stdout so ``| python3`` consumers can extract it.
"""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()

_BOOTSTRAP_RESPONSE = {
    "account_id": "acc_1",
    "key_id": "acckey_1",
    "plaintext_key": "aios_minted_xyz",
}


def test_bootstrap_posts_to_bootstrap_endpoint(mocked_cli, monkeypatch):
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["accounts", "bootstrap"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/accounts/bootstrap"


def test_bootstrap_sends_bootstrap_token_as_bearer(mocked_cli, monkeypatch):
    # CORE regression guard: the bootstrap token gates the endpoint and must
    # be the bearer, NOT the client AIOS_API_KEY (which is empty on a fresh DB).
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    monkeypatch.setenv("AIOS_API_KEY", "client-key")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["accounts", "bootstrap"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.headers.get("authorization") == "Bearer boot-tok"


def test_bootstrap_body_default_display_name(mocked_cli, monkeypatch):
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["accounts", "bootstrap"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body == {"display_name": "root"}


def test_bootstrap_custom_display_name(mocked_cli, monkeypatch):
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["accounts", "bootstrap", "-n", "ops"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.body == {"display_name": "ops"}


def test_bootstrap_prints_plaintext_key(mocked_cli, monkeypatch):
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["accounts", "bootstrap"])
    assert result.exit_code == 0, result.output
    assert "aios_minted_xyz" in result.output


def test_bootstrap_format_json_emits_full_payload(mocked_cli, monkeypatch):
    monkeypatch.setenv("AIOS_BOOTSTRAP_TOKEN", "boot-tok")
    mocked_cli.queue_response(httpx.Response(201, json=_BOOTSTRAP_RESPONSE))
    result = runner.invoke(app, ["-f", "json", "accounts", "bootstrap"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["account_id"] == "acc_1"
    assert payload["key_id"] == "acckey_1"
    assert payload["plaintext_key"] == "aios_minted_xyz"


def test_bootstrap_missing_token_errors(mocked_cli, monkeypatch):
    monkeypatch.delenv("AIOS_BOOTSTRAP_TOKEN", raising=False)
    result = runner.invoke(app, ["accounts", "bootstrap"])
    assert result.exit_code == 2
    assert "AIOS_BOOTSTRAP_TOKEN" in result.output
    # No response should have been consumed off the queue.
    assert len(mocked_cli.response_queue) == 0
