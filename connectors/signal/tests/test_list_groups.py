"""Unit tests for ``SignalDaemon.list_groups`` (issue #57)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aios_signal.daemon import GroupInfo, SignalDaemon


def _make_daemon() -> SignalDaemon:
    """Build a daemon with minimum scaffolding; tests drive the RPC directly."""
    from pathlib import Path

    return SignalDaemon(
        phone="+15550001111",
        config_dir=Path("/tmp/aios-signal-test"),
        cli_bin="signal-cli",
        host="127.0.0.1",
        port=9100,
    )


class TestListGroups:
    async def test_maps_rpc_response_into_group_infos(self) -> None:
        daemon = _make_daemon()
        daemon.rpc.call = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "id": "abc+/xyz=",
                    "name": "Group One",
                    "members": [
                        {"number": "+15551000001", "uuid": "11111111-0001-0000-0000-000000000000"},
                        {"number": "+15551000002", "uuid": "22222222-0002-0000-0000-000000000000"},
                    ],
                },
                {
                    "id": "def==",
                    "name": "Group Two",
                    "members": [
                        {"uuid": "33333333-0003-0000-0000-000000000000"},
                    ],
                },
            ]
        )

        groups = await daemon.list_groups()

        assert len(groups) == 2
        assert groups[0] == GroupInfo(
            id="abc-_xyz=",  # URL-safe-base64 re-encoded
            name="Group One",
            member_uuids=[
                "11111111-0001-0000-0000-000000000000",
                "22222222-0002-0000-0000-000000000000",
            ],
        )
        assert groups[1] == GroupInfo(
            id="def==",
            name="Group Two",
            member_uuids=["33333333-0003-0000-0000-000000000000"],
        )

    async def test_empty_list_when_rpc_fails(self) -> None:
        daemon = _make_daemon()
        daemon.rpc.call = AsyncMock(side_effect=RuntimeError("rpc broken"))  # type: ignore[method-assign]
        assert await daemon.list_groups() == []

    async def test_empty_list_when_rpc_returns_non_list(self) -> None:
        daemon = _make_daemon()
        daemon.rpc.call = AsyncMock(return_value={"unexpected": "shape"})  # type: ignore[method-assign]
        assert await daemon.list_groups() == []

    async def test_skips_malformed_entries(self) -> None:
        daemon = _make_daemon()
        daemon.rpc.call = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"id": "valid=", "name": "Ok", "members": [{"uuid": "aaaa"}]},
                "not a dict",
                {"no_id_field": True},
                {"id": "", "name": "empty id"},
                {"id": "skip-no-members", "name": "no members"},
            ]
        )
        groups = await daemon.list_groups()
        # Only the valid entry survives.
        assert len(groups) == 1
        assert groups[0].id == "valid="
        assert groups[0].name == "Ok"
        assert groups[0].member_uuids == ["aaaa"]


class TestGroupInfoIsFrozen:
    def test_fields_and_equality(self) -> None:
        a = GroupInfo(id="x", name="A", member_uuids=["u1"])
        b = GroupInfo(id="x", name="A", member_uuids=["u1"])
        assert a == b
        with pytest.raises((AttributeError, TypeError)):
            a.id = "y"  # type: ignore[misc]
