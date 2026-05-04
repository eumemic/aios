"""Unit tests for pure helpers in daemon.py.

``discover_bot_uuids`` reads ``accounts.json`` from the config dir
(signal-cli's on-disk account index) rather than RPC-ing
``listAccounts`` — both work in multi-account daemon mode but
accounts.json avoids a startup network round-trip.

Multi-account: returns a ``{phone: uuid}`` map for every configured
phone; raises if any phone is unregistered (operator must register
each before launching the connector).
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


def _daemon(phones: list[str], config_dir: Path) -> SignalDaemon:
    return SignalDaemon(
        phones=phones,
        config_dir=config_dir,
        cli_bin="signal-cli",
        host="127.0.0.1",
        port=0,
    )


async def test_discover_bot_uuids_single_phone(tmp_path: Path) -> None:
    _write_accounts(
        tmp_path,
        [
            {"number": "+15551111111", "uuid": "uuid-a"},
            {"number": "+15552222222", "uuid": "uuid-b"},
        ],
    )
    d = _daemon(["+15552222222"], tmp_path)
    assert await d.discover_bot_uuids() == {"+15552222222": "uuid-b"}


async def test_discover_bot_uuids_multiple_phones(tmp_path: Path) -> None:
    _write_accounts(
        tmp_path,
        [
            {"number": "+15551111111", "uuid": "uuid-a"},
            {"number": "+15552222222", "uuid": "uuid-b"},
            {"number": "+15553333333", "uuid": "uuid-c"},
        ],
    )
    d = _daemon(["+15551111111", "+15553333333"], tmp_path)
    assert await d.discover_bot_uuids() == {
        "+15551111111": "uuid-a",
        "+15553333333": "uuid-c",
    }


async def test_discover_bot_uuids_normalizes_whitespace(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [{"number": "  +15551234567 ", "uuid": "abc"}])
    d = _daemon(["+15551234567"], tmp_path)
    assert await d.discover_bot_uuids() == {"+15551234567": "abc"}


async def test_discover_bot_uuids_missing_phone_raises(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [{"number": "+15551111111", "uuid": "uuid-a"}])
    d = _daemon(["+15559999999"], tmp_path)
    with pytest.raises(BotAccountNotFoundError, match=r"\+15559999999"):
        await d.discover_bot_uuids()


async def test_discover_bot_uuids_partial_match_raises(tmp_path: Path) -> None:
    """One missing phone fails the whole call — surfaces all missing at once."""
    _write_accounts(tmp_path, [{"number": "+15551111111", "uuid": "uuid-a"}])
    d = _daemon(["+15551111111", "+15559999999"], tmp_path)
    with pytest.raises(BotAccountNotFoundError, match=r"\+15559999999"):
        await d.discover_bot_uuids()


async def test_discover_bot_uuids_empty_accounts_raises(tmp_path: Path) -> None:
    _write_accounts(tmp_path, [])
    d = _daemon(["+15550000000"], tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuids()


async def test_discover_bot_uuids_missing_accounts_file_raises(tmp_path: Path) -> None:
    # No accounts.json at all.
    d = _daemon(["+15550000000"], tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuids()


async def test_discover_bot_uuids_malformed_accounts_file_raises(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "accounts.json").write_text("{not valid json")
    d = _daemon(["+15550000000"], tmp_path)
    with pytest.raises(BotAccountNotFoundError):
        await d.discover_bot_uuids()
