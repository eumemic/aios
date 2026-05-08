"""Regression for #223 §3: signal connector startup must surface
``listGroups`` failure (typically "account not registered") rather than
marking the account ready with empty contacts/groups.

In the new single-phone-per-container model (#301) any setup failure
propagates from :meth:`SignalConnector.setup` directly — there's no
multi-account "drop one, keep going" path because each container is
exactly one account.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector

PHONE = "+15551111111"


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
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    c = SignalConnector(cfg)
    # Stub the secrets fetch so setup() can read the phone without an
    # actual API round-trip.
    c._secrets_cache = {"phone": PHONE}
    return c


async def test_setup_propagates_list_groups_failure(
    connector: SignalConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If signal-cli rejects the boot probe (e.g. unregistered account),
    setup raises so the operator sees the container fail to start
    instead of green-but-dead."""
    daemon = _stub_daemon()
    daemon.discover_bot_uuids.return_value = {PHONE: "bot-uuid"}
    daemon.list_contacts.return_value = {}
    daemon.list_groups.side_effect = RuntimeError("Specified account does not exist")
    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )

    with pytest.raises(RuntimeError, match="Specified account does not exist"):
        await connector.setup()


async def test_setup_succeeds_when_account_healthy(
    connector: SignalConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _stub_daemon()
    daemon.discover_bot_uuids.return_value = {PHONE: "bot-uuid"}
    daemon.list_contacts.return_value = {"bot-uuid": "Good Bot"}
    daemon.list_groups.return_value = []
    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )

    await connector.setup()

    assert connector._bot_uuid == "bot-uuid"
    assert connector._phone == PHONE
    assert connector._contact_names == {"bot-uuid": "Good Bot"}
    assert connector._groups == []


async def test_setup_refuses_without_phone_secret(tmp_path: Path) -> None:
    """A connection without a ``phone`` secret can't run the connector;
    fail loud at setup() rather than handing signal-cli an empty string."""
    cfg = Settings(
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    c = SignalConnector(cfg)
    # Empty secrets — connector should refuse.
    c._client = AsyncMock()
    c._client.get_secrets = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="requires a 'phone' secret"):
        await c.setup()


async def test_setup_refuses_when_phone_secret_empty_string(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly-empty phone is still missing for our purposes."""
    connector._secrets_cache = {"phone": ""}
    daemon = _stub_daemon()
    monkeypatch.setattr(
        "aios_signal.connector.SignalDaemon",
        MagicMock(return_value=daemon),
    )
    with pytest.raises(RuntimeError, match="requires a 'phone' secret"):
        await connector.setup()
