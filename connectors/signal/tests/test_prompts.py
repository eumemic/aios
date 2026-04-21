"""Unit tests for the instructions builder (issue #55)."""

from __future__ import annotations

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
