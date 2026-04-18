"""Unit tests for pure helpers in daemon.py.

``discover_bot_uuid`` reads ``accounts.json`` from the config dir
(signal-cli's on-disk account index) rather than RPC-ing
``listAccounts`` — the latter is not implemented in account-scoped
daemon mode, see the Signal smoke test findings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aios_signal.daemon import SignalDaemon
from aios_signal.errors import BotAccountNotFoundError


def _write_accounts(config_dir: Path, accounts: list[dict[str, Any]]) -> None:
    (config_dir / "data").mkdir(parents=True, exist_ok=True)
    (config_dir / "data" / "accounts.json").write_text(json.dumps({"accounts": accounts}))


def _daemon(phone: str, config_dir: Path) -> SignalDaemon:
    return SignalDaemon(
        phone=phone,
        config_dir=config_dir,
        cli_bin="signal-cli",
        host="127.0.0.1",
        port=0,
    )


async def test_discover_bot_uuid_match(tmp_path: Path) -> None:
    _write_accounts(
        tmp_path,
        [
            {"number": "+15551111111", "uuid": "uuid-a"},
            {"number": "+15552222222", "uuid": "uuid-b"},
        ],
    )
    d = _daemon("+15552222222", tmp_path)
    assert await d.discover_bot_uuid() == "uuid-b"


async def test_discover_bot_uuid_normalizes_whitespace(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [{"number": "  +15551234567 ", "uuid": "abc"}])
    d = _daemon("+15551234567", tmp_path)
    assert await d.discover_bot_uuid() == "abc"


async def test_discover_bot_uuid_no_match_raises(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [{"number": "+15551111111", "uuid": "uuid-a"}])
    d = _daemon("+15559999999", tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()


async def test_discover_bot_uuid_empty_list_raises(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [])
    d = _daemon("+15550000000", tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()


async def test_discover_bot_uuid_missing_accounts_file_raises(tmp_path: Path) -> None:
    # No accounts.json at all.
    d = _daemon("+15550000000", tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()


async def test_discover_bot_uuid_malformed_accounts_file_raises(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "accounts.json").write_text("{not valid json")
    d = _daemon("+15550000000", tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuid()
