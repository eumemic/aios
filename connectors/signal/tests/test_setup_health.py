"""Regression for #223 §3: signal connector startup must surface
listGroups failure (typically "account not registered") rather than
marking the account ready with empty contacts/groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector


def _stub_daemon() -> MagicMock:
    daemon = MagicMock()
    daemon.discover_bot_uuids = AsyncMock()
    daemon.list_contacts = AsyncMock()
    daemon.list_groups = AsyncMock()
    daemon.__aenter__ = AsyncMock(return_value=daemon)
    daemon.__aexit__ = AsyncMock(return_value=None)
    return daemon


@pytest.fixture
def connector(tmp_path: Path) -> SignalConnector:
    cfg = Settings(
        phones=["+15551111111"],
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    return SignalConnector(cfg)


async def test_setup_refuses_when_only_account_unavailable(
    connector: SignalConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One configured phone whose listGroups raises → setup must raise.
    Operator sees the connector fail to start instead of green-but-dead."""
    daemon = _stub_daemon()
    daemon.discover_bot_uuids.return_value = {"+15551111111": "bot-uuid"}
    daemon.list_contacts.return_value = {}
    daemon.list_groups.side_effect = RuntimeError("Specified account does not exist")
    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )

    with pytest.raises(RuntimeError, match="no configured account is healthy"):
        await connector.setup()


async def test_setup_drops_unavailable_account_when_others_healthy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-account setup: one bad account, one good → connector starts,
    bad account dropped from discover_accounts."""
    cfg = Settings(
        phones=["+15551111111", "+15552222222"],
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    connector = SignalConnector(cfg)

    daemon = _stub_daemon()
    daemon.discover_bot_uuids.return_value = {
        "+15551111111": "bot-bad",
        "+15552222222": "bot-good",
    }
    daemon.list_contacts.side_effect = lambda account: (
        {"bot-good": "Good Bot"} if account == "+15552222222" else {}
    )

    async def _list_groups(account: str) -> list[Any]:
        if account == "+15551111111":
            raise RuntimeError("not registered")
        return []

    daemon.list_groups.side_effect = _list_groups
    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )

    await connector.setup()

    accounts = await connector.discover_accounts()
    assert len(accounts) == 1
    assert accounts[0]["id"] == "bot-good"
    assert "+15551111111" in connector._unavailable_accounts
