"""Unit tests for the Telegram instructions builder (issue #55).

Mirrors the Signal connector's ``test_prompts.py`` — the agent otherwise
has no canonical source of truth for who it is on Telegram (numeric
``bot_id``, ``@username``, and display ``first_name`` all come from
``Bot.get_me()`` and should be surfaced in the MCP init instructions).
"""

from __future__ import annotations

from aios_telegram.prompts import TELEGRAM_SERVER_INSTRUCTIONS, build_instructions


class TestBuildInstructions:
    def test_includes_bot_id(self) -> None:
        result = build_instructions(bot_id=123456789, username="mybot_bot", first_name="My Bot")
        assert "123456789" in result

    def test_includes_username(self) -> None:
        result = build_instructions(bot_id=123456789, username="mybot_bot", first_name="My Bot")
        assert "mybot_bot" in result

    def test_includes_first_name(self) -> None:
        result = build_instructions(bot_id=123456789, username="mybot_bot", first_name="My Bot")
        assert "My Bot" in result

    def test_appends_tool_instructions_body(self) -> None:
        """The static affordance body must appear verbatim after the
        identity block — we're prepending, not rewriting."""
        result = build_instructions(bot_id=1, username="u", first_name="n")
        assert TELEGRAM_SERVER_INSTRUCTIONS in result

    def test_identity_block_comes_first(self) -> None:
        result = build_instructions(bot_id=1, username="u", first_name="n")
        identity_idx = result.find("## Your identity")
        body_idx = result.find("## chat_id")
        assert identity_idx >= 0
        assert body_idx > identity_idx

    def test_is_string(self) -> None:
        result = build_instructions(bot_id=1, username="u", first_name="n")
        assert isinstance(result, str)

    def test_username_rendered_with_at_prefix(self) -> None:
        """``@username`` is how Telegram users actually refer to bots, so
        render it that way rather than as a bare token — makes the agent's
        mental model match what it sees in chats."""
        result = build_instructions(bot_id=1, username="mybot_bot", first_name="My Bot")
        assert "@mybot_bot" in result

    def test_omits_username_line_when_none(self) -> None:
        """Not every bot has a username set (rare, but possible via
        BotFather before the bot is finalized).  Skip the bullet entirely
        rather than render a blank entry — the agent's parse of the
        identity block stays consistent."""
        result = build_instructions(bot_id=1, first_name="My Bot", username=None)
        assert "username" not in result

    def test_omits_username_line_when_empty_string(self) -> None:
        """Empty string equivalent to None — don't render a blank entry."""
        result = build_instructions(bot_id=1, first_name="My Bot", username="")
        assert "username" not in result
