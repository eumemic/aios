"""Regression for #223 §3: signal connector per-connection startup must
surface ``listGroups`` failure (typically "account not registered") and
missing-phone failures rather than marking the connection ready with
empty contacts/groups.

In the multi-connection model (#328 PR 6), per-account init moved from
``setup()`` (container-wide) to ``serve_connection(connection_id,
secrets)`` (one task per connection).  These tests drive that path
directly with a stubbed daemon — no SDK round-trip required.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector

PHONE = "+15551111111"
CONNECTION_ID = "conn_setup_test"


def _stub_daemon() -> MagicMock:
    daemon = MagicMock()
    daemon.verify_phone = AsyncMock()
    daemon.list_contacts = AsyncMock()
    daemon.list_groups = AsyncMock()
    return daemon


@pytest.fixture
def connector(tmp_path: Path) -> SignalConnector:
    cfg = Settings(
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    return SignalConnector(cfg)


async def _serve_until_idle(
    connector: SignalConnector, secrets: dict[str, str]
) -> None:
    """Run ``serve_connection`` up to the point where it starts draining
    the inbound queue, then cancel.  Lets us assert on the bot_uuid /
    contacts / groups registered in state without leaving a task hanging.
    """
    task = asyncio.create_task(connector.serve_connection(CONNECTION_ID, secrets))
    try:
        # Poll for the state to appear (means verify_phone + list_groups +
        # list_contacts have run and the queue-drain loop has started).
        for _ in range(50):
            if CONNECTION_ID in connector._conn_state:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_serve_connection_propagates_list_groups_failure(
    connector: SignalConnector,
) -> None:
    """If signal-cli rejects the boot probe (e.g. unregistered account),
    ``serve_connection`` raises so the SDK runner's TaskGroup tears the
    container down — operator sees the failure instead of green-but-dead."""
    daemon = _stub_daemon()
    daemon.verify_phone.return_value = "bot-uuid"
    daemon.list_contacts.return_value = {}
    daemon.list_groups.side_effect = RuntimeError("Specified account does not exist")
    connector._daemon = daemon

    with pytest.raises(RuntimeError, match="Specified account does not exist"):
        await connector.serve_connection(CONNECTION_ID, {"phone": PHONE})


async def test_serve_connection_registers_state_when_account_healthy(
    connector: SignalConnector,
) -> None:
    daemon = _stub_daemon()
    daemon.verify_phone.return_value = "bot-uuid"
    daemon.list_contacts.return_value = {"bot-uuid": "Good Bot"}
    daemon.list_groups.return_value = []
    connector._daemon = daemon

    await _serve_until_idle(connector, {"phone": PHONE})

    # Cancellation pops state on the way out; assert verify_phone +
    # list_contacts + list_groups all ran with the right account.
    daemon.verify_phone.assert_awaited_once_with(PHONE)
    daemon.list_contacts.assert_awaited_once_with(account=PHONE)
    daemon.list_groups.assert_awaited_once_with(account=PHONE)


async def test_serve_connection_refuses_without_phone_secret(
    connector: SignalConnector,
) -> None:
    """A connection without a ``phone`` secret can't run; fail loud at
    serve_connection rather than handing signal-cli an empty string."""
    with pytest.raises(RuntimeError, match="requires a 'phone' entry"):
        await connector.serve_connection(CONNECTION_ID, {})


async def test_serve_connection_refuses_when_phone_secret_empty_string(
    connector: SignalConnector,
) -> None:
    """An explicitly-empty phone is still missing for our purposes."""
    connector._daemon = _stub_daemon()
    with pytest.raises(RuntimeError, match="requires a 'phone' entry"):
        await connector.serve_connection(CONNECTION_ID, {"phone": ""})
