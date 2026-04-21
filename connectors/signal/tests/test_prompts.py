"""Unit tests for the instructions builder (issues #55, #57)."""

from __future__ import annotations

from aios_signal.daemon import GroupInfo
from aios_signal.prompts import SIGNAL_SERVER_INSTRUCTIONS, build_instructions


class TestBuildInstructions:
    def test_includes_bot_uuid(self) -> None:
        result = build_instructions(
            bot_uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", phone="+15551234567"
        )
        assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" in result

    def test_includes_phone(self) -> None:
        result = build_instructions(
            bot_uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", phone="+15551234567"
        )
        assert "+15551234567" in result

    def test_appends_tool_instructions_body(self) -> None:
        result = build_instructions(bot_uuid="u", phone="+1")
        # Every line of the static body must appear verbatim after the
        # identity block — we're prepending, not rewriting.
        assert SIGNAL_SERVER_INSTRUCTIONS in result

    def test_identity_block_comes_first(self) -> None:
        result = build_instructions(bot_uuid="u", phone="+1")
        identity_idx = result.find("## Your identity")
        body_idx = result.find("## chat_id")
        assert identity_idx >= 0
        assert body_idx > identity_idx

    def test_is_string(self) -> None:
        result = build_instructions(bot_uuid="u", phone="+1")
        assert isinstance(result, str)


class TestGroupRoster:
    def test_no_groups_no_roster_section(self) -> None:
        result = build_instructions(bot_uuid="u", phone="+1")
        assert "## Your Signal groups" not in result

    def test_empty_groups_list_no_roster_section(self) -> None:
        result = build_instructions(bot_uuid="u", phone="+1", groups=[])
        assert "## Your Signal groups" not in result

    def test_renders_group_name_and_id(self) -> None:
        result = build_instructions(
            bot_uuid="bot-uuid",
            phone="+1",
            groups=[
                GroupInfo(id="grp-abc=", name="Weekend Plans", member_uuids=["bot-uuid", "u1"]),
            ],
            contact_names={"u1": "Alice"},
        )
        assert "## Your Signal groups" in result
        assert "grp-abc=" in result
        assert "Weekend Plans" in result

    def test_flags_bot_as_YOU_in_member_list(self) -> None:
        result = build_instructions(
            bot_uuid="bot-uuid",
            phone="+1",
            groups=[GroupInfo(id="g", name="G", member_uuids=["bot-uuid", "u1"])],
            contact_names={"u1": "Alice"},
        )
        assert "(YOU)" in result
        # Alice rendered by display name, not uuid
        assert "Alice" in result

    def test_missing_name_shows_unknown_tag(self) -> None:
        result = build_instructions(
            bot_uuid="bot-uuid",
            phone="+1",
            groups=[GroupInfo(id="g", name="G", member_uuids=["u-anon"])],
            contact_names={},
        )
        assert "(name unknown)" in result

    def test_roster_block_comes_before_static_body(self) -> None:
        result = build_instructions(
            bot_uuid="u",
            phone="+1",
            groups=[GroupInfo(id="g", name="G", member_uuids=["u"])],
        )
        roster_idx = result.find("## Your Signal groups")
        body_idx = result.find("## chat_id")
        assert roster_idx >= 0 and body_idx >= 0
        assert roster_idx < body_idx

    def test_multiple_groups_listed(self) -> None:
        result = build_instructions(
            bot_uuid="b",
            phone="+1",
            groups=[
                GroupInfo(id="g1", name="First", member_uuids=["b", "x"]),
                GroupInfo(id="g2", name="Second", member_uuids=["b", "y"]),
            ],
            contact_names={"x": "Xavier", "y": "Yolanda"},
        )
        assert "First" in result and "Second" in result
        assert "Xavier" in result and "Yolanda" in result

    def test_unnamed_group_renders_as_unnamed(self) -> None:
        result = build_instructions(
            bot_uuid="b",
            phone="+1",
            groups=[GroupInfo(id="g", name="", member_uuids=["b", "x"])],
            contact_names={"x": "X"},
        )
        assert "(unnamed)" in result
