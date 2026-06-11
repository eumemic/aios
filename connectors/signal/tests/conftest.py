"""Shared test fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

# HttpConnector reads AIOS_URL / AIOS_RUNTIME_TOKEN at __init__ time.
os.environ.setdefault("AIOS_URL", "http://test")
os.environ.setdefault("AIOS_RUNTIME_TOKEN", "aios_runtime_test")

from aios_signal import connector as _connector_module
from aios_signal.config import Settings
from aios_signal.connector import SignalConnector, _SignalConnectionState


@pytest.fixture(autouse=True)
def _fast_echo_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the group-send self-echo deadline so tests don't pay 2s
    per group ``signal_send`` while waiting for an echo that the mocked
    daemon never emits.  Tests that exercise the echo path explicitly
    set the future themselves; everyone else relies on this autouse to
    hit the timeout in ~10ms instead of ~2s."""
    monkeypatch.setattr(_connector_module, "_ECHO_WAIT_S", 0.01)


FIXTURES_DIR = Path(__file__).parent / "fixtures"

BOT_UUID = "99999999-8888-7777-6666-555555555555"
PHONE = "+15550001"
CONNECTION_ID = "conn_test"

# Sample identities used across send/delete/groups tests.
ALICE_UUID = "11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB_UUID = "22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
GROUP_CHAT_ID = "abcXYZ123_-"  # URL-safe base64; not a UUID
# decode_chat_id reverses URL-safe → standard base64 before handing the
# raw form to signal-cli.
GROUP_RAW_ID = "abcXYZ123/+"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES_DIR / name).read_text())
    return data


@pytest.fixture
def connector(tmp_path: Path) -> SignalConnector:
    """SignalConnector with a stubbed daemon + one pre-registered connection.

    The connection state is pre-populated under ``CONNECTION_ID`` so
    tool tests can call ``dispatch_call`` (or invoke tool methods
    directly) without running ``serve_connection`` first.  RPC calls
    are captured on the mock; tests assert the right phone landed in
    the params.
    """
    cfg = Settings(
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    c = SignalConnector(cfg)
    c._daemon = type(
        "Daemon",
        (),
        {
            "rpc": type("Rpc", (), {"call": AsyncMock(return_value=None)})(),
            "list_groups": AsyncMock(return_value=[]),
            "verify_phone": AsyncMock(return_value="bot-uuid"),
            # ``subprocess_alive`` / ``listener`` are consulted by the
            # inbound dispatcher's reconnect path; tests that exercise it
            # override these (the listener stub + alive flag), but the
            # attributes must exist so ``monkeypatch.setattr`` can patch
            # them and so a stray dispatcher read doesn't AttributeError.
            "subprocess_alive": lambda self: True,
            "listener": None,
        },
    )()
    c.state[CONNECTION_ID] = _SignalConnectionState(
        phone=PHONE,
        bot_uuid="bot-uuid",
        contact_names={},
        groups=[],
    )
    return c


@pytest.fixture
def bot_uuid() -> str:
    return BOT_UUID


@pytest.fixture
def envelope_text_dm() -> dict[str, Any]:
    return _load("text_dm.json")


@pytest.fixture
def envelope_text_group() -> dict[str, Any]:
    return _load("text_group.json")


@pytest.fixture
def envelope_reaction() -> dict[str, Any]:
    return _load("reaction.json")


@pytest.fixture
def envelope_reply() -> dict[str, Any]:
    return _load("reply.json")


@pytest.fixture
def envelope_attachment_only() -> dict[str, Any]:
    return _load("attachment_only.json")


@pytest.fixture
def envelope_attachment_no_file_field() -> dict[str, Any]:
    return _load("attachment_no_file_field.json")


@pytest.fixture
def envelope_receipt() -> dict[str, Any]:
    return _load("receipt.json")


@pytest.fixture
def envelope_typing() -> dict[str, Any]:
    return _load("typing.json")


@pytest.fixture
def envelope_self() -> dict[str, Any]:
    return _load("self_message.json")


@pytest.fixture
def envelope_source_less_receipt() -> dict[str, Any]:
    return _load("source_less_receipt.json")


@pytest.fixture
def envelope_missing_server_guid() -> dict[str, Any]:
    return _load("missing_server_guid.json")
