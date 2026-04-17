"""Unit tests for harness/channels.py.

Pure-function coverage.  The connection-lookup helper (async, hits the
DB) is covered in tests/e2e.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aios.harness.channels import (
    apply_monologue_prefix,
    augment_with_channels,
    build_channels_system_block,
    connection_server_name,
)
from aios.models.channel_bindings import ChannelBinding
from aios.models.connections import Connection


def _binding(address: str, session_id: str = "sess_x") -> ChannelBinding:
    now = datetime(2026, 4, 16)
    return ChannelBinding(
        id=f"cbnd_{hash(address) & 0xFFFF:04x}",
        address=address,
        session_id=session_id,
        created_at=now,
        updated_at=now,
    )


def _connection(cid: str, connector: str = "signal", account: str = "acct") -> Connection:
    now = datetime(2026, 4, 16)
    return Connection(
        id=cid,
        connector=connector,
        account=account,
        mcp_url="https://example.com",
        vault_id="vlt_x",
        metadata={},
        created_at=now,
        updated_at=now,
    )


# ── connection_server_name ─────────────────────────────────────────────────


class TestConnectionServerName:
    """The connection id already starts with the reserved ``conn_``
    prefix (via ``ids.CONNECTION``), so it doubles as the server name
    directly — no stutter, still unambiguous by construction.
    """

    def test_uses_id_directly(self) -> None:
        c = _connection("conn_01HQR2K7VXBZ9MNPL3WYCT8F")
        assert connection_server_name(c) == "conn_01HQR2K7VXBZ9MNPL3WYCT8F"

    def test_distinct_connections_produce_distinct_names(self) -> None:
        c1 = _connection("conn_aaa", connector="signal", account="abc_def")
        c2 = _connection("conn_bbb", connector="signal_abc", account="def")
        assert connection_server_name(c1) != connection_server_name(c2)

    def test_is_stable_for_same_connection(self) -> None:
        c = _connection("conn_stable")
        assert connection_server_name(c) == connection_server_name(c)


# ── build_channels_system_block / augment_with_channels ────────────────────


class TestBuildChannelsSystemBlock:
    def test_empty_bindings_returns_empty_string(self) -> None:
        assert build_channels_system_block([]) == ""

    def test_single_binding_lists_address(self) -> None:
        block = build_channels_system_block([_binding("signal/alice/chat-1")])
        assert "signal/alice/chat-1" in block
        assert "INTERNAL_MONOLOGUE" in block
        assert "bound to the following channels" in block.lower()

    def test_multiple_bindings_all_listed(self) -> None:
        block = build_channels_system_block(
            [
                _binding("signal/alice/chat-1"),
                _binding("slack/ws/C123/t"),
            ]
        )
        assert "signal/alice/chat-1" in block
        assert "slack/ws/C123/t" in block


class TestAugmentWithChannels:
    def test_no_bindings_returns_base_unchanged(self) -> None:
        assert augment_with_channels("you are a helpful agent", []) == "you are a helpful agent"

    def test_appends_block_after_base(self) -> None:
        result = augment_with_channels("you are helpful", [_binding("signal/a/1")])
        assert result.startswith("you are helpful")
        assert "signal/a/1" in result
        # Separator between base and appended block (blank line).
        assert "\n\n" in result

    def test_empty_base_yields_block_only(self) -> None:
        result = augment_with_channels("", [_binding("signal/a/1")])
        assert "signal/a/1" in result
        assert not result.startswith("\n")


# ── apply_monologue_prefix ─────────────────────────────────────────────────


class TestApplyMonologuePrefix:
    def test_string_content_prefixed(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": "thinking out loud"}
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE: thinking out loud"

    def test_already_prefixed_string_unchanged(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": "INTERNAL_MONOLOGUE: hi"}
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE: hi"

    def test_empty_string_left_alone(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": ""}
        out = apply_monologue_prefix(msg)
        assert out["content"] == ""

    def test_none_content_left_alone(self) -> None:
        """A tool-only assistant turn (content=None) has nothing to prefix."""
        msg: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
        out = apply_monologue_prefix(msg)
        assert out.get("content") is None

    def test_missing_content_left_alone(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "tool_calls": []}
        out = apply_monologue_prefix(msg)
        assert "content" not in out

    def test_list_content_all_text_blocks_prefixed(self) -> None:
        """Multi-block content must prefix EVERY text block — not just the
        first — so providers that interleave text with tool_use don't leave
        later text segments unmarked.
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "id": "x", "name": "y", "input": {}},
                {"type": "text", "text": "second"},
            ],
        }
        out = apply_monologue_prefix(msg)
        blocks = out["content"]
        assert blocks[0] == {"type": "text", "text": "INTERNAL_MONOLOGUE: first"}
        assert blocks[1] == {"type": "tool_use", "id": "x", "name": "y", "input": {}}
        assert blocks[2] == {"type": "text", "text": "INTERNAL_MONOLOGUE: second"}

    def test_list_content_tool_use_only_left_alone(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "y", "input": {}}],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"] == [{"type": "tool_use", "id": "x", "name": "y", "input": {}}]

    def test_list_content_mixed_already_prefixed(self) -> None:
        """Idempotent on a per-block basis."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "INTERNAL_MONOLOGUE: first"},
                {"type": "text", "text": "second"},
            ],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"][0]["text"] == "INTERNAL_MONOLOGUE: first"
        assert out["content"][1]["text"] == "INTERNAL_MONOLOGUE: second"

    def test_returns_new_dict_preserving_other_fields(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [{"id": "x"}],
            "reacting_to": 42,
        }
        out = apply_monologue_prefix(msg)
        assert out["role"] == "assistant"
        assert out["tool_calls"] == [{"id": "x"}]
        assert out["reacting_to"] == 42
        # Input must not be mutated in place.
        assert msg["content"] == "hi"
