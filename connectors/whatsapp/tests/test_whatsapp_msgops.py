"""Tests for the ``whatsapp_react`` / ``whatsapp_edit_message`` /
``whatsapp_delete_message`` tool methods."""

from __future__ import annotations

import pytest

from aios_whatsapp.connector import WhatsappConnector

from .conftest import CONNECTION_ID

MSG_ID = "3EB0E03B46303C22D750E2"


async def test_whatsapp_react_dispatches_sendReaction(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "REACT-1",
        "timestamp_ms": 1700000300000,
    }
    result = await connector.whatsapp_react(
        message_id=MSG_ID,
        reaction="👍",
        connection_id=CONNECTION_ID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendReaction",
        {"message_id": MSG_ID, "reaction": "👍"},
    )
    assert result == {"message_id": "REACT-1", "timestamp_ms": 1700000300000}


async def test_whatsapp_react_empty_reaction_clears_prior(connector: WhatsappConnector) -> None:
    # Empty string is the documented "remove reaction" affordance —
    # the daemon forwards it verbatim to whatsmeow's BuildReaction.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "REACT-2",
        "timestamp_ms": 1700000301000,
    }
    await connector.whatsapp_react(
        message_id=MSG_ID,
        reaction="",
        connection_id=CONNECTION_ID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "sendReaction",
        {"message_id": MSG_ID, "reaction": ""},
    )


async def test_whatsapp_edit_message_dispatches_editMessage(connector: WhatsappConnector) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "EDIT-1",
        "timestamp_ms": 1700000400000,
    }
    result = await connector.whatsapp_edit_message(
        message_id=MSG_ID,
        text="corrected",
        connection_id=CONNECTION_ID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "editMessage",
        {"message_id": MSG_ID, "text": "corrected"},
    )
    assert result == {"message_id": "EDIT-1", "timestamp_ms": 1700000400000}


async def test_whatsapp_edit_message_encodes_mentions(
    connector: WhatsappConnector,
) -> None:
    # Pre-fix: edit silently stripped any @<E.164> mentions because
    # the edit path didn't run encode_mentions and the daemon's
    # editMessage RPC had no mentioned_jids param.  Post-fix: the
    # edit carries mentions through to whatsmeow's BuildEdit, which
    # builds an ExtendedTextMessage with ContextInfo.MentionedJID.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "EDIT-2",
        "timestamp_ms": 1700000401000,
    }
    await connector.whatsapp_edit_message(
        message_id=MSG_ID,
        text="hey @+15551234567 corrected",
        connection_id=CONNECTION_ID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert sent["mentioned_jids"] == ["15551234567@s.whatsapp.net"]


async def test_whatsapp_edit_message_omits_mentions_when_none(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "EDIT-3",
        "timestamp_ms": 1700000402000,
    }
    await connector.whatsapp_edit_message(
        message_id=MSG_ID,
        text="no tags here",
        connection_id=CONNECTION_ID,
    )
    sent = connector.state[CONNECTION_ID].daemon.rpc.call.await_args.args[1]  # type: ignore[attr-defined]
    assert "mentioned_jids" not in sent


async def test_whatsapp_delete_message_dispatches_deleteMessage(
    connector: WhatsappConnector,
) -> None:
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = {  # type: ignore[attr-defined]
        "message_id": "REV-1",
        "timestamp_ms": 1700000500000,
    }
    result = await connector.whatsapp_delete_message(
        message_id=MSG_ID,
        connection_id=CONNECTION_ID,
    )
    connector.state[CONNECTION_ID].daemon.rpc.call.assert_awaited_once_with(  # type: ignore[attr-defined]
        "deleteMessage",
        {"message_id": MSG_ID},
    )
    assert result == {"message_id": "REV-1", "timestamp_ms": 1700000500000}


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("whatsapp_react", {"message_id": MSG_ID, "reaction": "👍"}),
        ("whatsapp_edit_message", {"message_id": MSG_ID, "text": "x"}),
        ("whatsapp_delete_message", {"message_id": MSG_ID}),
    ],
)
async def test_msgops_raise_on_non_dict_result(
    connector: WhatsappConnector,
    method: str,
    kwargs: dict[str, str],
) -> None:
    # Each tool wraps RuntimeError around a malformed daemon response —
    # surfaces a clear failure path to the model instead of returning
    # an unparseable string into the tool_result.
    connector.state[CONNECTION_ID].daemon.rpc.call.return_value = "not-a-dict"  # type: ignore[attr-defined]
    tool_fn = getattr(connector, method)
    with pytest.raises(RuntimeError, match="returned non-dict"):
        await tool_fn(**kwargs, connection_id=CONNECTION_ID)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("whatsapp_react", {"message_id": MSG_ID, "reaction": "👍"}),
        ("whatsapp_edit_message", {"message_id": MSG_ID, "text": "x"}),
        ("whatsapp_delete_message", {"message_id": MSG_ID}),
    ],
)
async def test_msgops_raise_on_unknown_connection(
    connector: WhatsappConnector,
    method: str,
    kwargs: dict[str, str],
) -> None:
    tool_fn = getattr(connector, method)
    with pytest.raises(KeyError):
        await tool_fn(**kwargs, connection_id="conn_unknown")
