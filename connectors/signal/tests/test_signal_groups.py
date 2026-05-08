"""Unit coverage for ``signal_create_group`` and ``signal_rename_group``."""

from __future__ import annotations

import pytest

from aios_signal.connector import SignalConnector
from tests.conftest import (
    ALICE_UUID,
    BOB_UUID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    PHONE,
)

# ── signal_create_group ──────────────────────────────────────────────


async def test_create_group_sends_no_groupid(connector: SignalConnector) -> None:
    connector._daemon.rpc.call.return_value = {"groupId": "newgroup_id"}  # type: ignore[union-attr]
    result = await connector.signal_create_group(
        name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID], chat_id=ALICE_UUID
    )
    assert result == {"group_id": "newgroup_id"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "updateGroup"
    assert params == {
        "account": PHONE,
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
    with pytest.raises(RuntimeError, match="did not return a groupId"):
        await connector.signal_create_group(name="x", member_uuids=[], chat_id=ALICE_UUID)


# ── signal_rename_group ──────────────────────────────────────────────


async def test_rename_group_with_group_focal(connector: SignalConnector) -> None:
    result = await connector.signal_rename_group(name="Renamed", chat_id=GROUP_CHAT_ID)
    assert result == {"status": "ok"}
    method, params = connector._daemon.rpc.call.call_args.args  # type: ignore[union-attr]
    assert method == "updateGroup"
    assert params == {
        "account": PHONE,
        "groupId": GROUP_RAW_ID,
        "name": "Renamed",
    }


async def test_rename_group_from_dm_focal_raises(connector: SignalConnector) -> None:
    with pytest.raises(ValueError, match="DM, not a group"):
        await connector.signal_rename_group(name="X", chat_id=ALICE_UUID)


# ── _maybe_refresh_roster ──────────────────────────────────────────────


async def test_maybe_refresh_roster_calls_list_groups_on_update(
    connector: SignalConnector,
) -> None:
    from aios_signal.daemon import GroupInfo

    fresh = [GroupInfo(id="abc", name="QA", member_uuids=[ALICE_UUID, BOB_UUID])]
    connector._daemon.list_groups.return_value = fresh  # type: ignore[union-attr]
    envelope = {"dataMessage": {"groupInfo": {"groupId": "abc==", "type": "UPDATE"}}}
    await connector._maybe_refresh_roster(envelope)
    connector._daemon.list_groups.assert_awaited_once_with(account=PHONE)  # type: ignore[union-attr]
    assert connector._groups == fresh


async def test_maybe_refresh_roster_no_op_for_regular_message(
    connector: SignalConnector,
) -> None:
    envelope = {
        "dataMessage": {
            "message": "hi",
            "groupInfo": {"groupId": "abc==", "type": "DELIVER"},
        }
    }
    await connector._maybe_refresh_roster(envelope)
    connector._daemon.list_groups.assert_not_awaited()  # type: ignore[union-attr]
    assert connector._groups == []
