"""Unit coverage for ``signal_send``.

Drives the ``_build_send_params`` helper directly (no signal-cli) plus
exercises the high-level ``signal_send`` method by calling it as a
plain async method.  The SDK's focal-channel injection and SandboxPath
resolution are SDK concerns covered in
``packages/aios-connector-http/tests``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios_signal.connector import SignalConnector, _build_send_params
from aios_signal.daemon import GroupInfo
from aios_signal.parse import MENTION_PLACEHOLDER
from tests.conftest import (
    ALICE_UUID,
    BOB_UUID,
    CONNECTION_ID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    PHONE,
)

# ── _build_send_params: attachments ─────────────────────────────────


def test_build_params_no_attachments() -> None:
    params = _build_send_params(PHONE, "alice", "hello", attachments=[])
    assert "attachments" not in params
    assert params["message"] == "hello"


def test_build_params_with_attachments() -> None:
    params = _build_send_params(
        PHONE,
        "alice",
        "look",
        attachments=[Path("/host/a.jpg"), Path("/host/b.jpg")],
    )
    assert params["attachments"] == ["/host/a.jpg", "/host/b.jpg"]


# ── signal_send: direct method calls with already-resolved paths ─────


async def test_signal_send_dm_result_carries_channel_and_chat_type(
    connector: SignalConnector,
) -> None:
    result = await connector.signal_send(
        text="hello there", chat_id=ALICE_UUID, connection_id=CONNECTION_ID
    )
    # The result stamps the resolved focal channel + chat_type so
    # external observers don't have to reconstruct focal heuristically.
    # The connector fixture's bot_uuid is the literal "bot-uuid".
    assert result == {
        "status": "ok",
        "channel": f"signal/bot-uuid/{ALICE_UUID}",
        "chat_type": "dm",
    }
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "hello there"
    assert "attachments" not in sent_params


async def test_signal_send_group_result_carries_channel_and_chat_type(
    connector: SignalConnector,
) -> None:
    connector.state[CONNECTION_ID].groups = [
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    result = await connector.signal_send(
        text="hello group", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
    )
    assert result["channel"] == f"signal/bot-uuid/{GROUP_CHAT_ID}"
    assert result["chat_type"] == "group"


async def test_signal_send_sent_at_ms_branch_carries_channel(
    connector: SignalConnector,
) -> None:
    # When signal-cli returns a timestamp inline (DM path), the result
    # still stamps channel + chat_type alongside ``sent_at_ms``.
    connector._daemon.rpc.call.return_value = {"timestamp": 1700000000000}  # type: ignore[union-attr]
    result = await connector.signal_send(text="hi", chat_id=ALICE_UUID, connection_id=CONNECTION_ID)
    assert result == {
        "sent_at_ms": 1700000000000,
        "channel": f"signal/bot-uuid/{ALICE_UUID}",
        "chat_type": "dm",
    }


async def test_signal_send_with_resolved_attachments(
    connector: SignalConnector, tmp_path: Path
) -> None:
    photo = tmp_path / "cat.jpg"
    photo.write_bytes(b"x")
    await connector.signal_send(
        text="look",
        attachments=[photo],
        chat_id=ALICE_UUID,
        connection_id=CONNECTION_ID,
    )
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "look"
    assert sent_params["attachments"] == [str(photo)]


# ── dispatch_call exercises the SDK SandboxPath resolution end-to-end ─


async def test_signal_send_dispatch_resolves_sandbox_path(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "acc-1" / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")
    connector._client = AsyncMock()

    # The dispatch_call → _post_tool_result path hits the generated SDK
    # op which doesn't tolerate an AsyncMock client; override with a
    # no-op so the test focuses on the SandboxPath resolution, not the
    # result POST.
    async def _noop_result(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(connector, "_post_tool_result", _noop_result)

    await connector.dispatch_call(
        {
            "connection_id": CONNECTION_ID,
            "tool_call_id": "c1",
            "session_id": "sess-1",
            "name": "signal_send",
            "arguments": json.dumps({"text": "look", "attachments": ["/workspace/cat.jpg"]}),
            "focal_channel": f"signal/bot-uuid/{ALICE_UUID}",
            "workspace_path": str(ws),
        }
    )

    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["attachments"] == [str(ws / "cat.jpg")]


async def test_signal_send_dispatch_traversal_returns_error_result(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path that escapes the bind-mount root surfaces as an error
    result; the tool body never runs and signal-cli is never called."""
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "acc-1" / "sess-1").resolve()
    ws.mkdir(parents=True)
    connector._client = AsyncMock()

    captured: list[dict[str, Any]] = []

    async def _capture_result(
        client: Any,
        *,
        connection_id: str,
        session_id: str,
        tool_call_id: str,
        content: Any,
        is_error: bool = False,
    ) -> None:
        del client
        captured.append(
            {
                "connection_id": connection_id,
                "session_id": session_id,
                "tool_call_id": tool_call_id,
                "content": content,
                "is_error": is_error,
            }
        )

    monkeypatch.setattr(connector, "_post_tool_result", _capture_result)

    await connector.dispatch_call(
        {
            "connection_id": CONNECTION_ID,
            "tool_call_id": "c2",
            "session_id": "sess-1",
            "name": "signal_send",
            "arguments": json.dumps({"text": "look", "attachments": ["/workspace/../escape.jpg"]}),
            "focal_channel": f"signal/bot-uuid/{ALICE_UUID}",
            "workspace_path": str(ws),
        }
    )

    connector._daemon.rpc.call.assert_not_awaited()  # type: ignore[union-attr]
    assert len(captured) == 1
    assert captured[0]["is_error"] is True


# ── quote / reply ─────────────────────────────────────────────────────


def test_build_params_quote_both_set_passes_through() -> None:
    params = _build_send_params(
        PHONE,
        ALICE_UUID,
        "replying",
        attachments=[],
        quote_timestamp_ms=1700000000000,
        quote_author_uuid=ALICE_UUID,
    )
    assert params["quoteTimestamp"] == 1700000000000
    assert params["quoteAuthor"] == ALICE_UUID


def test_build_params_quote_partial_raises() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        _build_send_params(PHONE, ALICE_UUID, "x", attachments=[], quote_timestamp_ms=123)
    with pytest.raises(ValueError, match="must be set together"):
        _build_send_params(PHONE, ALICE_UUID, "x", attachments=[], quote_author_uuid=ALICE_UUID)


# ── edit ──────────────────────────────────────────────────────────────


def test_build_params_edit_timestamp_threaded() -> None:
    params = _build_send_params(
        PHONE,
        ALICE_UUID,
        "edited",
        attachments=[],
        edit_timestamp_ms=1700000005000,
    )
    assert params["editTimestamp"] == 1700000005000


# ── outbound mentions ─────────────────────────────────────────────────


def test_build_params_mentions_resolved_in_group() -> None:
    params = _build_send_params(
        PHONE,
        GROUP_CHAT_ID,
        f"hi @{ALICE_UUID[:8]}",
        attachments=[],
        member_uuids=[ALICE_UUID, BOB_UUID],
    )
    assert params["message"] == f"hi {MENTION_PLACEHOLDER}"
    assert params["mentions"] == [f"3:1:{ALICE_UUID}"]
    assert params["groupId"] == GROUP_RAW_ID


def test_build_params_no_member_uuids_no_mentions_added() -> None:
    # DMs (and groups before the roster cache populates) get empty
    # member_uuids; ``@<hex>`` syntax should pass through verbatim.
    params = _build_send_params(
        PHONE,
        ALICE_UUID,
        f"hi @{ALICE_UUID[:8]}",
        attachments=[],
    )
    assert params["message"] == f"hi @{ALICE_UUID[:8]}"
    assert "mentions" not in params


def test_build_params_mention_after_markdown_uses_stripped_offset() -> None:
    # Markdown delimiters before a mention shift the placeholder leftward
    # in ``stripped`` text; the mention offset must reflect ``stripped``,
    # not the pre-strip ``encoded`` form.  Without offset rebasing, Signal
    # would highlight the wrong character.
    params = _build_send_params(
        PHONE,
        GROUP_CHAT_ID,
        f"**bold** @{ALICE_UUID[:8]} check",
        attachments=[],
        member_uuids=[ALICE_UUID, BOB_UUID],
    )
    assert params["message"] == f"bold {MENTION_PLACEHOLDER} check"
    # Placeholder is at UTF-16 offset 5 in stripped text, not 9 in encoded.
    assert params["mentions"] == [f"5:1:{ALICE_UUID}"]


def test_build_params_pre_existing_placeholder_does_not_crash() -> None:
    # Forwarded inbound text could carry a stray U+FFFC (e.g. attachment-
    # inline marker).  encode_mentions strips it so build_mention_strings
    # doesn't see an orphan placeholder it can't pair with a UUID.
    params = _build_send_params(
        PHONE,
        GROUP_CHAT_ID,
        f"{MENTION_PLACEHOLDER}stray @{ALICE_UUID[:8]}",
        attachments=[],
        member_uuids=[ALICE_UUID, BOB_UUID],
    )
    assert params["message"] == f"stray {MENTION_PLACEHOLDER}"
    assert params["mentions"] == [f"6:1:{ALICE_UUID}"]


# ── group focal: signal_send picks up member_uuids from connector state ─


async def test_signal_send_in_group_encodes_mentions(
    connector: SignalConnector,
) -> None:
    # Seed the connector's group cache so signal_send knows the focal
    # group's members at send time.
    connector.state[CONNECTION_ID].groups = [
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    await connector.signal_send(
        text=f"hey @{ALICE_UUID[:8]}, ready?",
        chat_id=GROUP_CHAT_ID,
        connection_id=CONNECTION_ID,
    )
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == f"hey {MENTION_PLACEHOLDER}, ready?"
    assert sent_params["mentions"] == [f"4:1:{ALICE_UUID}"]
    assert sent_params["groupId"] == GROUP_RAW_ID


async def test_signal_send_refreshes_group_cache_on_miss(
    connector: SignalConnector,
) -> None:
    # signal-cli sometimes returns an empty listGroups at boot before the
    # account state has finished loading; the boot-time cache stays
    # empty for the lifetime of the connector and outbound mentions
    # silently degrade.  signal_send must refresh on miss so the second
    # listGroups (now populated) populates the cache and the mention
    # encodes correctly.
    connector.state[CONNECTION_ID].groups = []
    connector._daemon.list_groups.return_value = [  # type: ignore[union-attr]
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    await connector.signal_send(
        text=f"hey @{ALICE_UUID[:8]}", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
    )
    connector._daemon.list_groups.assert_awaited_once_with(account=PHONE)  # type: ignore[union-attr]
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["mentions"] == [f"4:1:{ALICE_UUID}"]


async def test_signal_send_skips_refresh_on_cache_hit(
    connector: SignalConnector,
) -> None:
    connector.state[CONNECTION_ID].groups = [
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    await connector.signal_send(
        text="no mentions", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
    )
    connector._daemon.list_groups.assert_not_awaited()  # type: ignore[union-attr]


async def test_signal_send_dm_skips_refresh(connector: SignalConnector) -> None:
    # DMs never need a group roster — refreshing on every DM send would
    # be a wasted RPC.
    connector.state[CONNECTION_ID].groups = []
    await connector.signal_send(text="hey", chat_id=ALICE_UUID, connection_id=CONNECTION_ID)
    connector._daemon.list_groups.assert_not_awaited()  # type: ignore[union-attr]
