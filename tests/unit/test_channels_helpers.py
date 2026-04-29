"""Unit tests for harness/channels.py.

Pure-function coverage.  The connection-lookup helper (async, hits the
DB) is covered in tests/e2e.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.channels import (
    apply_monologue_prefix,
    augment_with_connector_instructions,
    augment_with_focal_paradigm,
    build_channels_tail_block,
    build_connector_instructions_block,
    build_focal_paradigm_block,
    connection_server_name,
)
from aios.models.channel_bindings import ChannelBinding, NotificationMode
from aios.models.connections import Connection
from aios.models.events import Event


def _binding(
    address: str,
    session_id: str = "sess_x",
    *,
    notification_mode: NotificationMode = "focal_candidate",
) -> ChannelBinding:
    now = datetime(2026, 4, 16)
    # Reconstruct the (connection_id, path) storage form from the display
    # address the tests supply.  The connection_id is a stable derivation
    # of the first two segments so tests get consistent IDs.
    parts = address.split("/", 2)
    connector, account, path = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
    return ChannelBinding(
        id=f"cbnd_{hash(address) & 0xFFFF:04x}",
        connection_id=f"conn_{hash((connector, account)) & 0xFFFF:04x}",
        path=path,
        address=address,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        notification_mode=notification_mode,
    )


def _user_event(
    seq: int,
    *,
    orig: str | None = None,
    focal_at: str | None = None,
    content: str = "hello",
) -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        session_id="sess_x",
        seq=seq,
        kind="message",
        data={"role": "user", "content": content},
        cumulative_tokens=None,
        created_at=datetime(2026, 4, 17, tzinfo=UTC),
        orig_channel=orig,
        focal_channel_at_arrival=focal_at,
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
    """Per-connection instruction aliases use the connection id directly."""

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


# ── build_focal_paradigm_block / augment_with_focal_paradigm ───────────────


class TestBuildFocalParadigmBlock:
    """Slice 7: paradigm prose introducing the focal-channel model.

    Cache-stable: text does not vary with per-channel state (unread
    counts, previews) — that lives in the ephemeral tail block.
    """

    def test_empty_bindings_returns_empty_string(self) -> None:
        assert build_focal_paradigm_block([]) == ""

    def test_mentions_focal_and_switch_channel(self) -> None:
        block = build_focal_paradigm_block([_binding("signal/alice/chat-1")])
        assert "focal" in block.lower()
        assert "switch_channel" in block

    def test_describes_tail_symbols(self) -> None:
        block = build_focal_paradigm_block([_binding("signal/alice/chat-1")])
        assert "▸" in block
        assert "○" in block
        assert "◌" in block

    def test_contains_monologue_reminder(self) -> None:
        block = build_focal_paradigm_block([_binding("signal/alice/chat-1")])
        assert "INTERNAL_MONOLOGUE" in block

    def test_no_per_channel_data_leakage(self) -> None:
        """The block must not name any specific bound channel — that's
        the tail block's job.  Paradigm prose stays cache-stable.
        """
        block = build_focal_paradigm_block(
            [
                _binding("signal/alice/chat-1"),
                _binding("slack/workspace/channel/thread"),
            ]
        )
        assert "signal/alice/chat-1" not in block
        assert "slack/workspace/channel/thread" not in block

    def test_describes_phone_down_state(self) -> None:
        block = build_focal_paradigm_block([_binding("signal/alice/chat-1")])
        assert "phone down" in block.lower() or "target=null" in block


# ── build_connector_instructions_block / augment_with_connector_instructions ──


class TestBuildConnectorInstructionsBlock:
    def test_empty_dict_returns_empty(self) -> None:
        assert build_connector_instructions_block({}, []) == ""

    def test_no_matching_connection_returns_empty(self) -> None:
        """Instructions keyed under a server name that no connection
        matches must NOT be rendered — we only describe the connectors
        the session is actually bound to.
        """
        c = _connection("conn_aaa", connector="signal", account="alice")
        block = build_connector_instructions_block({"conn_unknown": "stray prose"}, [c])
        assert block == ""

    def test_single_connection_renders_heading_and_body(self) -> None:
        c = _connection("conn_aaa", connector="signal", account="alice")
        block = build_connector_instructions_block({connection_server_name(c): "be brief"}, [c])
        assert "## Connector: signal/alice" in block
        assert "be brief" in block

    def test_multiple_connections_rendered_in_input_order(self) -> None:
        """Ordering is caller-controlled — important for prompt-cache
        stability across steps.  Test by passing two connections and
        asserting the output order matches.
        """
        c1 = _connection("conn_aaa", connector="signal", account="alice")
        c2 = _connection("conn_bbb", connector="signal", account="bob")
        instructions = {
            connection_server_name(c1): "alice prose",
            connection_server_name(c2): "bob prose",
        }
        block = build_connector_instructions_block(instructions, [c1, c2])
        assert block.index("alice prose") < block.index("bob prose")
        # Reverse the connections list — output order flips.
        block_rev = build_connector_instructions_block(instructions, [c2, c1])
        assert block_rev.index("bob prose") < block_rev.index("alice prose")

    def test_connection_without_instructions_skipped(self) -> None:
        c1 = _connection("conn_aaa", connector="signal", account="alice")
        c2 = _connection("conn_bbb", connector="signal", account="bob")
        block = build_connector_instructions_block(
            {connection_server_name(c2): "bob prose"}, [c1, c2]
        )
        assert "alice" not in block
        assert "bob prose" in block


class TestAugmentWithConnectorInstructions:
    def test_no_instructions_returns_base_unchanged(self) -> None:
        c = _connection("conn_aaa")
        assert augment_with_connector_instructions("base", {}, [c]) == "base"

    def test_appends_block_after_base(self) -> None:
        c = _connection("conn_aaa", connector="signal", account="alice")
        out = augment_with_connector_instructions(
            "base system", {connection_server_name(c): "prose"}, [c]
        )
        assert out.startswith("base system")
        assert "## Connector: signal/alice" in out
        assert "\n\n" in out

    def test_empty_base_yields_block_only(self) -> None:
        c = _connection("conn_aaa", connector="signal", account="alice")
        out = augment_with_connector_instructions("", {connection_server_name(c): "prose"}, [c])
        assert out.startswith("## Connector:")
        assert not out.startswith("\n")


class TestAugmentWithFocalParadigm:
    def test_no_bindings_returns_base_unchanged(self) -> None:
        assert (
            augment_with_focal_paradigm("you are a helpful agent", []) == "you are a helpful agent"
        )

    def test_appends_block_after_base(self) -> None:
        result = augment_with_focal_paradigm("you are helpful", [_binding("signal/a/1")])
        assert result.startswith("you are helpful")
        assert "switch_channel" in result
        # Separator between base and appended block (blank line).
        assert "\n\n" in result

    def test_empty_base_yields_block_only(self) -> None:
        result = augment_with_focal_paradigm("", [_binding("signal/a/1")])
        assert "switch_channel" in result
        assert not result.startswith("\n")


# ── build_channels_tail_block ──────────────────────────────────────────────


class TestBuildChannelsTailBlock:
    """Slice 7: the ephemeral per-step listing of bound channels.

    Pure data block — no prose explaining the paradigm (that's the job
    of :func:`build_focal_paradigm_block`, which is cache-stable and
    lives in the system prompt).
    """

    _ALICE = "signal/bot/alice"
    _FAMILY = "signal/bot/family"
    _ANNOUNCEMENTS = "signal/bot/announcements"

    def test_no_bindings_returns_none(self) -> None:
        assert build_channels_tail_block([], [], focal_channel=None) is None

    def test_focal_line_marked_with_triangle_no_unread_count(self) -> None:
        block = build_channels_tail_block([_binding(self._ALICE)], [], focal_channel=self._ALICE)
        assert block is not None
        content = block["content"]
        assert "▸ channel_id=signal/bot/alice (focal)" in content
        # Focal line must not advertise an unread count — you ARE in it.
        # Check that no digit appears on the focal line.
        focal_line = next(ln for ln in content.splitlines() if "▸" in ln)
        assert not any(ch.isdigit() for ch in focal_line), focal_line

    def test_non_focal_channel_shows_unread_count(self) -> None:
        events = [
            _user_event(1, orig=self._FAMILY, focal_at=self._ALICE, content="hi from mom"),
            _user_event(2, orig=self._FAMILY, focal_at=self._ALICE, content="and again"),
        ]
        block = build_channels_tail_block(
            [_binding(self._ALICE), _binding(self._FAMILY)],
            events,
            focal_channel=self._ALICE,
        )
        assert block is not None
        content = block["content"]
        assert "○ channel_id=signal/bot/family — 2 unread" in content

    def test_non_focal_preview_truncated(self) -> None:
        long = "x" * 200
        events = [_user_event(1, orig=self._FAMILY, focal_at=self._ALICE, content=long)]
        block = build_channels_tail_block(
            [_binding(self._ALICE), _binding(self._FAMILY)],
            events,
            focal_channel=self._ALICE,
        )
        assert block is not None
        content = block["content"]
        assert "x" * 60 + "…" in content
        assert "x" * 61 not in content

    def test_muted_channel_shows_count_only_no_preview(self) -> None:
        events = [
            _user_event(
                1,
                orig=self._ANNOUNCEMENTS,
                focal_at=self._ALICE,
                content="system noise",
            ),
        ]
        block = build_channels_tail_block(
            [
                _binding(self._ALICE),
                _binding(self._ANNOUNCEMENTS, notification_mode="silent"),
            ],
            events,
            focal_channel=self._ALICE,
        )
        assert block is not None
        content = block["content"]
        assert f"◌ channel_id={self._ANNOUNCEMENTS} (muted) — 1 unread" in content
        # No preview (no quoted content).
        assert "system noise" not in content

    def test_phone_down_shows_no_focal_marker(self) -> None:
        block = build_channels_tail_block(
            [_binding(self._ALICE), _binding(self._FAMILY)],
            [],
            focal_channel=None,
        )
        assert block is not None
        content = block["content"]
        assert "▸" not in content
        assert "(focal)" not in content

    def test_tail_message_shape_is_user_role(self) -> None:
        block = build_channels_tail_block([_binding(self._ALICE)], [], focal_channel=self._ALICE)
        assert block is not None
        assert block["role"] == "user"
        assert isinstance(block["content"], str)
        assert set(block.keys()) <= {"role", "content"}

    def test_zero_unread_non_focal_still_listed(self) -> None:
        block = build_channels_tail_block(
            [_binding(self._ALICE), _binding(self._FAMILY)],
            [],
            focal_channel=self._ALICE,
        )
        assert block is not None
        content = block["content"]
        assert f"○ channel_id={self._FAMILY} — 0 unread" in content


# ── apply_monologue_prefix ─────────────────────────────────────────────────


class TestApplyMonologuePrefix:
    def test_string_content_prefixed(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": "thinking out loud"}
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: thinking out loud"

    def test_already_prefixed_string_unchanged(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: hi",
        }
        out = apply_monologue_prefix(msg)
        assert out["content"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: hi"

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

    def test_list_content_only_first_text_block_prefixed(self) -> None:
        """Multi-block content must prefix ONLY the first text block.  The
        assistant message is one logical turn; stamping every text segment
        produced double/triple prefixes on providers (e.g. Gemma) that emit
        a reasoning block before the actual response.
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
        assert blocks[0] == {"type": "text", "text": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"}
        assert blocks[1] == {"type": "tool_use", "id": "x", "name": "y", "input": {}}
        assert blocks[2] == {"type": "text", "text": "second"}

    def test_list_content_tool_use_only_left_alone(self) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": "y", "input": {}}],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"] == [{"type": "tool_use", "id": "x", "name": "y", "input": {}}]

    def test_list_content_first_block_already_prefixed_is_idempotent(self) -> None:
        """When the first text block already carries the prefix, the call
        is a no-op — no double-prefix, and subsequent blocks still stay
        un-stamped (see :func:`apply_monologue_prefix`).
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"},
                {"type": "text", "text": "second"},
            ],
        }
        out = apply_monologue_prefix(msg)
        assert out["content"][0]["text"] == "INTERNAL_MONOLOGUE_NOT_SEEN_BY_USER: first"
        assert out["content"][1]["text"] == "second"

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
