"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aios_connector.base import ToolDescriptor

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BOT_UUID = "99999999-8888-7777-6666-555555555555"

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
    """SignalConnector with a stubbed daemon — tests drive RPC via the mock."""
    cfg = Settings(
        phones=["+15550001"],
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    c = SignalConnector(cfg)
    c._uuid_to_phone = {"bot-uuid": "+15550001"}
    c._daemon = type(
        "Daemon",
        (),
        {
            "rpc": type("Rpc", (), {"call": AsyncMock(return_value=None)})(),
            # signal_create_group refreshes the roster cache via list_groups
            # post-create; tests that don't care can leave the default ``[]``.
            "list_groups": AsyncMock(return_value=[]),
        },
    )()
    return c


def descriptor(connector: SignalConnector, name: str) -> ToolDescriptor:
    """Look up a tool descriptor by name on the connector."""
    return next(d for d in connector._tools if d.name == name)


def stub_focal(connector: SignalConnector, value: str | None) -> None:
    """Override the connector's focal-channel resolver for a single test."""
    connector._focal_from_request_meta = lambda: value  # type: ignore[method-assign]


def decode_tool_result(content_list: list[Any]) -> dict[str, Any]:
    """Parse the JSON payload from a tool's MCP-style content list."""
    return dict(json.loads(content_list[0].text))


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
