"""Unit coverage for ``signal_create_group`` and ``signal_rename_group``."""

from __future__ import annotations

import pytest

from aios_signal.connector import SignalConnector
from tests.conftest import (
    ALICE_UUID,
    BOB_UUID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    decode_tool_result,
    descriptor,
    stub_focal,
)

# ── signal_create_group ──────────────────────────────────────────────


async def test_create_group_sends_no_groupid(connector: SignalConnector) -> None:
    connector._daemon.rpc.call.return_value = {"groupId": "newgroup_id"}  # type: ignore[union-attr]
    stub_focal(connector, f"bot-uuid/{ALICE_UUID}")  # focal can be a DM
    result = decode_tool_result(
        await connector._invoke_tool(
            descriptor(connector, "signal_create_group"),
            {"name": "Tea Party", "member_uuids": [ALICE_UUID, BOB_UUID]},
        )
    )
    assert result == {"group_id": "newgroup_id"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "updateGroup"
    assert params == {
        "account": "+15550001",
        "name": "Tea Party",
        "members": [ALICE_UUID, BOB_UUID],
    }
    assert "groupId" not in params


async def test_create_group_raises_when_no_groupid_in_result(
    connector: SignalConnector,
) -> None:
    # signal-cli's contract is to return {"groupId": ...} on a successful
    # create.  Anything else is a contract break — fail loud rather than
    # silently swallowing.
    stub_focal(connector, f"bot-uuid/{ALICE_UUID}")
    with pytest.raises(RuntimeError, match="did not return a groupId"):
        await connector._invoke_tool(
            descriptor(connector, "signal_create_group"),
            {"name": "x", "member_uuids": []},
        )


# ── signal_rename_group ──────────────────────────────────────────────


async def test_rename_group_with_group_focal(connector: SignalConnector) -> None:
    stub_focal(connector, f"bot-uuid/{GROUP_CHAT_ID}")
    result = decode_tool_result(
        await connector._invoke_tool(
            descriptor(connector, "signal_rename_group"), {"name": "Renamed"}
        )
    )
    assert result == {"status": "ok"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "updateGroup"
    assert params == {
        "account": "+15550001",
        "groupId": GROUP_RAW_ID,
        "name": "Renamed",
    }


async def test_rename_group_from_dm_focal_raises(connector: SignalConnector) -> None:
    stub_focal(connector, f"bot-uuid/{ALICE_UUID}")
    with pytest.raises(ValueError, match="DM, not a group"):
        await connector._invoke_tool(descriptor(connector, "signal_rename_group"), {"name": "X"})


async def test_rename_group_unknown_account_raises(connector: SignalConnector) -> None:
    stub_focal(connector, f"nope/{GROUP_CHAT_ID}")
    with pytest.raises(ValueError, match="unknown account"):
        await connector._invoke_tool(descriptor(connector, "signal_rename_group"), {"name": "X"})
