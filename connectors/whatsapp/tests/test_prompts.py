"""Tests for the WhatsApp prompts.py identity + roster prelude."""

from __future__ import annotations

from aios_whatsapp.prompts import (
    WHATSAPP_SERVER_INSTRUCTIONS,
    GroupRosterEntry,
    build_instructions,
)


def test_build_instructions_emits_identity_and_static_body() -> None:
    result = build_instructions(
        bot_jid="15555550000@s.whatsapp.net",
        phone="+15555550000",
    )
    assert "**jid**: `15555550000@s.whatsapp.net`" in result
    assert "**phone**: `+15555550000`" in result
    # Static body always concatenates at the end.
    assert WHATSAPP_SERVER_INSTRUCTIONS in result


def test_build_instructions_includes_push_name_when_set() -> None:
    result = build_instructions(
        bot_jid="x@s.whatsapp.net",
        phone="+1",
        push_name="Ultron",
    )
    assert "**push_name**: `Ultron`" in result


def test_build_instructions_omits_push_name_when_blank() -> None:
    # An empty push_name shouldn't render a blank bullet that
    # consumes a line in the system prompt for no signal.
    result = build_instructions(bot_jid="x@s.whatsapp.net", phone="+1", push_name="")
    assert "push_name" not in result


def test_build_instructions_renders_group_roster() -> None:
    bot = "15555550000@s.whatsapp.net"
    groups = [
        GroupRosterEntry(
            jid="g1@g.us",
            name="Engineering",
            member_jids=[bot, "15551234567@s.whatsapp.net", "18007654321@s.whatsapp.net"],
        ),
        GroupRosterEntry(
            jid="g2@g.us",
            name="",
            member_jids=[bot, "15551234567@s.whatsapp.net"],
        ),
    ]
    result = build_instructions(bot_jid=bot, phone="+15555550000", groups=groups)
    assert "## Your WhatsApp groups" in result
    assert "`g1@g.us` — Engineering" in result
    # Unnamed group falls through to a placeholder so the section
    # isn't visually broken by an empty name.
    assert "`g2@g.us` — (unnamed)" in result
    # Self-tag distinguishes the bot from peers in the roster.
    assert "(YOU)" in result


def test_build_instructions_omits_group_section_when_no_groups() -> None:
    result = build_instructions(bot_jid="x@s.whatsapp.net", phone="+1")
    assert "## Your WhatsApp groups" not in result


def test_build_instructions_matches_bot_with_device_suffix_jid() -> None:
    # Pre-fix: (YOU) tag used literal string equality, so a roster
    # entry showing the bot's JID with a device-suffix variant
    # silently dropped the self-tag.  Post-fix: comparison is on the
    # identity-bearing local part (strip device suffix + host).
    bot = "15555550000@s.whatsapp.net"
    groups = [
        GroupRosterEntry(
            jid="g1@g.us",
            name="Engineering",
            member_jids=["15555550000:7@s.whatsapp.net", "15551234567@s.whatsapp.net"],
        ),
    ]
    result = build_instructions(bot_jid=bot, phone="+15555550000", groups=groups)
    assert "(YOU)" in result


def test_static_body_mentions_only_existing_tools() -> None:
    # Defensive: the static body must reference only the WhatsApp
    # tools this connector actually publishes, so the model isn't
    # told to call something that doesn't exist.
    for expected in (
        "whatsapp_send",
        "whatsapp_react",
        "whatsapp_edit_message",
        "whatsapp_delete_message",
        "whatsapp_list_groups",
        "whatsapp_create_group",
        "whatsapp_rename_group",
    ):
        assert expected in WHATSAPP_SERVER_INSTRUCTIONS, f"missing {expected}"
    # Sanity check: don't accidentally mention a tool from another
    # connector (e.g., signal_send).
    assert "signal_" not in WHATSAPP_SERVER_INSTRUCTIONS
