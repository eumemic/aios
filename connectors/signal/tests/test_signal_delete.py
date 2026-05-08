"""Unit coverage for ``signal_delete``."""

from __future__ import annotations

from aios_signal.connector import SignalConnector
from tests.conftest import ALICE_UUID, GROUP_CHAT_ID, GROUP_RAW_ID, PHONE


async def test_signal_delete_dm_uses_recipient(connector: SignalConnector) -> None:
    result = await connector.signal_delete(target_timestamp_ms=1700000000000, chat_id=ALICE_UUID)
    assert result == {"status": "ok"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "remoteDelete"
    assert params == {
        "account": PHONE,
        "targetTimestamp": 1700000000000,
        "recipient": [ALICE_UUID],
    }


async def test_signal_delete_group_uses_groupid(connector: SignalConnector) -> None:
    await connector.signal_delete(target_timestamp_ms=999, chat_id=GROUP_CHAT_ID)
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "remoteDelete"
    assert params == {
        "account": PHONE,
        "targetTimestamp": 999,
        "groupId": GROUP_RAW_ID,
    }
