"""Unit coverage for ``signal_delete``."""

from __future__ import annotations

import pytest

from aios_signal.connector import SignalConnector
from tests.conftest import (
    ALICE_UUID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    decode_tool_result,
    descriptor,
    stub_focal,
)


async def test_signal_delete_dm_uses_recipient(connector: SignalConnector) -> None:
    stub_focal(connector, f"bot-uuid/{ALICE_UUID}")
    result = decode_tool_result(
        await connector._invoke_tool(
            descriptor(connector, "signal_delete"),
            {"target_timestamp_ms": 1700000000000},
        )
    )
    assert result == {"status": "ok"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "remoteDelete"
    assert params == {
        "account": "+15550001",
        "targetTimestamp": 1700000000000,
        "recipient": [ALICE_UUID],
    }


async def test_signal_delete_group_uses_groupid(connector: SignalConnector) -> None:
    stub_focal(connector, f"bot-uuid/{GROUP_CHAT_ID}")
    await connector._invoke_tool(
        descriptor(connector, "signal_delete"), {"target_timestamp_ms": 999}
    )
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "remoteDelete"
    assert params == {
        "account": "+15550001",
        "targetTimestamp": 999,
        "groupId": GROUP_RAW_ID,
    }


async def test_signal_delete_unknown_account_raises(connector: SignalConnector) -> None:
    stub_focal(connector, f"nope/{ALICE_UUID}")
    with pytest.raises(ValueError, match="unknown account"):
        await connector._invoke_tool(
            descriptor(connector, "signal_delete"), {"target_timestamp_ms": 1}
        )
