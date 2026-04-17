"""Unit tests for pure helpers in daemon.py.

The subprocess + TCP integration is exercised by test_integration.py; this
file covers the match-by-phone logic via a lightweight SignalDaemon stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aios_signal.daemon import SignalDaemon
from aios_signal.errors import BotAccountNotFoundError


def _daemon_with_accounts(phone: str, accounts: list[dict[str, Any]]) -> SignalDaemon:
    d = SignalDaemon(
        phone=phone,
        config_dir=Path("/tmp"),
        cli_bin="signal-cli",
        host="127.0.0.1",
        port=0,
    )
    d._accounts = accounts
    return d


async def test_discover_bot_uuid_match() -> None:
    d = _daemon_with_accounts(
        "+15552222222",
        [
            {"number": "+15551111111", "uuid": "uuid-a"},
            {"number": "+15552222222", "uuid": "uuid-b"},
        ],
    )
    assert await d.discover_bot_uuid() == "uuid-b"


async def test_discover_bot_uuid_normalizes_whitespace() -> None:
    d = _daemon_with_accounts(
        "+15551234567",
        [{"number": "  +15551234567 ", "uuid": "abc"}],
    )
    assert await d.discover_bot_uuid() == "abc"


async def test_discover_bot_uuid_no_match_raises() -> None:
    d = _daemon_with_accounts(
        "+15559999999",
        [{"number": "+15551111111", "uuid": "uuid-a"}],
    )
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()


async def test_discover_bot_uuid_empty_list_raises() -> None:
    d = _daemon_with_accounts("+15550000000", [])
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()
