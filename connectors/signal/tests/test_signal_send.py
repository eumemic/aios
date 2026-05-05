"""Unit coverage for ``signal_send``.

Drives the ``_build_send_params`` helper directly (no signal-cli) plus
exercises the high-level ``signal_send`` method via the SDK's
``_invoke_tool`` dispatch wrapper — that's the layer where
``SandboxPath`` resolution happens, so connector-side tests must go
through it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios_signal.connector import SignalConnector, _build_send_params
from aios_signal.daemon import GroupInfo
from aios_signal.parse import MENTION_PLACEHOLDER
from tests.conftest import (
    ALICE_UUID,
    BOB_UUID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    decode_tool_result,
    descriptor,
    stub_focal,
)


def _stub_session_id(connector: SignalConnector, value: str | None) -> None:
    connector.current_session_id = lambda: value  # type: ignore[method-assign]


# ── _build_send_params: attachments ─────────────────────────────────


def test_build_params_no_attachments() -> None:
    params = _build_send_params("+15550001", "alice", "hello", attachments=[])
    assert "attachments" not in params
    assert params["message"] == "hello"


def test_build_params_with_attachments() -> None:
    params = _build_send_params(
        "+15550001",
        "alice",
        "look",
        attachments=[Path("/host/a.jpg"), Path("/host/b.jpg")],
    )
    assert params["attachments"] == ["/host/a.jpg", "/host/b.jpg"]


# ── descriptor metadata ─────────────────────────────────────────────


def test_signal_send_descriptor_records_sandbox_param(
    connector: SignalConnector,
) -> None:
    d = descriptor(connector, "signal_send")
    assert d.sandbox_params == {"attachments": "list"}


def test_signal_send_schema_publishes_string_array_with_description(
    connector: SignalConnector,
) -> None:
    d = descriptor(connector, "signal_send")
    attachments_schema = d.input_schema["properties"]["attachments"]
    assert attachments_schema["type"] == "array"
    assert attachments_schema["items"]["type"] == "string"
    assert "/workspace/" in attachments_schema["items"]["description"]


# ── signal_send via _invoke_tool (exercises SandboxPath resolution) ─


async def test_signal_send_text_only_no_session_id_required(
    connector: SignalConnector,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, None)
    result = decode_tool_result(
        await connector._invoke_tool(descriptor(connector, "signal_send"), {"text": "hello there"})
    )
    assert result == {"status": "ok"}
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "hello there"
    assert "attachments" not in sent_params


async def test_signal_send_with_attachments_resolves_to_host_paths(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    await connector._invoke_tool(
        descriptor(connector, "signal_send"),
        {"text": "look", "attachments": ["/workspace/cat.jpg"]},
    )

    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "look"
    assert sent_params["attachments"] == [str(ws / "cat.jpg")]


async def test_signal_send_attachments_without_session_id_raises(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await connector._invoke_tool(
            descriptor(connector, "signal_send"),
            {"text": "look", "attachments": ["/workspace/cat.jpg"]},
        )


async def test_signal_send_attachment_traversal_rejected(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(
            descriptor(connector, "signal_send"),
            {"text": "look", "attachments": ["/workspace/../escape.jpg"]},
        )


async def test_signal_send_attachment_disallowed_root_rejected(
    connector: SignalConnector,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(
            descriptor(connector, "signal_send"),
            {"text": "boom", "attachments": ["/etc/passwd"]},
        )


async def test_signal_send_attachment_missing_file_raises_clear_error(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    with pytest.raises(ValueError, match="does not exist"):
        await connector._invoke_tool(
            descriptor(connector, "signal_send"),
            {"text": "look", "attachments": ["/workspace/typo.jpg"]},
        )


async def test_signal_send_unknown_account_raises(
    connector: SignalConnector,
) -> None:
    stub_focal(connector, "nope/alice")
    with pytest.raises(ValueError, match="unknown account"):
        await connector._invoke_tool(descriptor(connector, "signal_send"), {"text": "hi"})


# ── quote / reply ─────────────────────────────────────────────────────


def test_build_params_quote_both_set_passes_through() -> None:
    params = _build_send_params(
        "+15550001",
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
        _build_send_params("+15550001", ALICE_UUID, "x", attachments=[], quote_timestamp_ms=123)
    with pytest.raises(ValueError, match="must be set together"):
        _build_send_params(
            "+15550001", ALICE_UUID, "x", attachments=[], quote_author_uuid=ALICE_UUID
        )


# ── edit ──────────────────────────────────────────────────────────────


def test_build_params_edit_timestamp_threaded() -> None:
    params = _build_send_params(
        "+15550001",
        ALICE_UUID,
        "edited",
        attachments=[],
        edit_timestamp_ms=1700000005000,
    )
    assert params["editTimestamp"] == 1700000005000


# ── outbound mentions ─────────────────────────────────────────────────


def test_build_params_mentions_resolved_in_group() -> None:
    params = _build_send_params(
        "+15550001",
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
        "+15550001",
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
        "+15550001",
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
        "+15550001",
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
    connector._groups_by_account["+15550001"] = [
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    stub_focal(connector, f"bot-uuid/{GROUP_CHAT_ID}")
    await connector._invoke_tool(
        descriptor(connector, "signal_send"), {"text": f"hey @{ALICE_UUID[:8]}, ready?"}
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
    connector._groups_by_account["+15550001"] = []
    connector._daemon.list_groups.return_value = [  # type: ignore[union-attr]
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    stub_focal(connector, f"bot-uuid/{GROUP_CHAT_ID}")
    await connector._invoke_tool(
        descriptor(connector, "signal_send"), {"text": f"hey @{ALICE_UUID[:8]}"}
    )
    connector._daemon.list_groups.assert_awaited_once_with(account="+15550001")  # type: ignore[union-attr]
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["mentions"] == [f"4:1:{ALICE_UUID}"]


async def test_signal_send_skips_refresh_on_cache_hit(
    connector: SignalConnector,
) -> None:
    connector._groups_by_account["+15550001"] = [
        GroupInfo(id=GROUP_CHAT_ID, name="Tea Party", member_uuids=[ALICE_UUID, BOB_UUID])
    ]
    stub_focal(connector, f"bot-uuid/{GROUP_CHAT_ID}")
    await connector._invoke_tool(descriptor(connector, "signal_send"), {"text": "no mentions"})
    connector._daemon.list_groups.assert_not_awaited()  # type: ignore[union-attr]


async def test_signal_send_dm_skips_refresh(connector: SignalConnector) -> None:
    # DMs never need a group roster — refreshing on every DM send would
    # be a wasted RPC.
    connector._groups_by_account["+15550001"] = []
    stub_focal(connector, f"bot-uuid/{ALICE_UUID}")
    await connector._invoke_tool(descriptor(connector, "signal_send"), {"text": "hey"})
    connector._daemon.list_groups.assert_not_awaited()  # type: ignore[union-attr]
