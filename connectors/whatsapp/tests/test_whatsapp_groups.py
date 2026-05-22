"""Tests for the WhatsApp group tools."""

from __future__ import annotations

import pytest

from aios_whatsapp.connector import WhatsappConnector

from .conftest import CONNECTION_ID, GROUP_JID


async def test_whatsapp_list_groups_dispatches_listGroups(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "groups": [
            {
                "jid": GROUP_JID,
                "name": "Test Group",
                "topic": "",
                "participants": [{"jid": "x@s.whatsapp.net", "is_admin": True}],
            }
        ]
    }
    result = await connector.whatsapp_list_groups(connection_id=CONNECTION_ID)
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with("listGroups", {})  # type: ignore[attr-defined]
    assert result["groups"][0]["jid"] == GROUP_JID
    assert result["groups"][0]["participants"][0]["is_admin"] is True


async def test_whatsapp_create_group_converts_phones_to_jids(
    connector: WhatsappConnector,
) -> None:
    # Model passes +E.164 phones; connector normalizes to JIDs the
    # daemon can hand to whatsmeow's CreateGroup.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "jid": "12345-67890@g.us",
        "name": "New Group",
        "participants": [],
    }
    await connector.whatsapp_create_group(
        name="New Group",
        participants=["+15551234567", "18007654321", "+1 555 999 0000"],
        connection_id=CONNECTION_ID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert sent["name"] == "New Group"
    assert sent["participants"] == [
        "15551234567@s.whatsapp.net",
        "18007654321@s.whatsapp.net",
        "15559990000@s.whatsapp.net",
    ]


async def test_whatsapp_create_group_strips_parens_and_dots(
    connector: WhatsappConnector,
) -> None:
    # Pre-fix: _phone_to_jid only stripped +/space/dash, so a
    # ``(555) 123-4567`` style phone leaked parens into the JID
    # local part and the daemon's ParseJID rejected with an opaque
    # error.  Post-fix: all non-digit characters are stripped so
    # common formatter variants all resolve to the same JID.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "jid": "g@g.us",
        "name": "x",
        "participants": [],
    }
    await connector.whatsapp_create_group(
        name="x",
        participants=["+1 (555) 123-4567", "+1.555.999.0000"],
        connection_id=CONNECTION_ID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert sent["participants"] == [
        "15551234567@s.whatsapp.net",
        "15559990000@s.whatsapp.net",
    ]


async def test_whatsapp_create_group_rejects_digit_free_phone(
    connector: WhatsappConnector,
) -> None:
    # Defensive: a participant string with no digits at all (e.g.
    # the model passed a display name) should fail at the Python
    # boundary with a clear ValueError rather than producing a
    # malformed JID the daemon rejects opaquely.
    with pytest.raises(ValueError, match="no digits"):
        await connector.whatsapp_create_group(
            name="x",
            participants=["Alice"],
            connection_id=CONNECTION_ID,
        )


async def test_whatsapp_create_group_passes_jid_through_unchanged(
    connector: WhatsappConnector,
) -> None:
    # If the model already has a JID (e.g., from list_groups output)
    # it can pass that verbatim; the connector recognizes the ``@``
    # and skips the phone-normalize.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "jid": "g@g.us",
        "name": "x",
        "participants": [],
    }
    await connector.whatsapp_create_group(
        name="x",
        participants=["existing@s.whatsapp.net"],
        connection_id=CONNECTION_ID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert sent["participants"] == ["existing@s.whatsapp.net"]


async def test_whatsapp_rename_group_forwards_renameGroup(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {"status": "ok"}  # type: ignore[attr-defined]
    await connector.whatsapp_rename_group(
        chat_id=GROUP_JID,
        name="Renamed",
        connection_id=CONNECTION_ID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "renameGroup",
        {"jid": GROUP_JID, "name": "Renamed"},
    )


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("whatsapp_list_groups", {}),
        ("whatsapp_create_group", {"name": "x", "participants": ["+15550000000"]}),
        ("whatsapp_rename_group", {"chat_id": GROUP_JID, "name": "x"}),
    ],
)
async def test_group_tools_raise_on_non_dict_result(
    connector: WhatsappConnector,
    method: str,
    kwargs: dict[str, object],
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = "not-a-dict"  # type: ignore[attr-defined]
    tool_fn = getattr(connector, method)
    with pytest.raises(RuntimeError, match="returned non-dict"):
        await tool_fn(**kwargs, connection_id=CONNECTION_ID)
