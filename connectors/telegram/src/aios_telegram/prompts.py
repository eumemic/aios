"""Per-connector affordance prose surfaced to the agent via MCP.

Returned in the ``InitializeResult.instructions`` field of the MCP
``initialize`` response; the aios harness concatenates it into the
session's system prompt under a ``## Connector: telegram/<account>``
heading.

Covers the tools this server exposes — ``telegram_send``,
``telegram_typing``, ``telegram_edit_message``, ``telegram_delete_message``,
``telegram_react``.  Telling the model about tools that don't exist
would be worse than silence.

``build_instructions`` prepends an identity block (the bot's own numeric
``bot_id``, ``@username``, and display ``first_name``) so the agent knows
who it is on Telegram without having to learn that from inbound traffic
(issue #55).
"""

from __future__ import annotations


def build_instructions(
    *,
    bot_id: int,
    first_name: str,
    username: str | None = None,
) -> str:
    """Compose the MCP ``initialize`` instructions with identity prelude.

    All three identity fields come from ``Bot.get_me()`` at connector
    startup.  ``bot_id`` and ``first_name`` are guaranteed by Telegram's
    schema; ``username`` is optional (a bot can exist without one during
    BotFather setup) and its bullet is omitted when absent or empty.
    ``bot_id`` doubles as the ``<account>`` segment of this connector's
    channel addresses.
    """
    lines = [
        "## Your identity on this Telegram bot account",
        "",
        f"- **bot_id**: `{bot_id}` — this is YOU.  In group chats, any "
        f"inbound whose header shows this id is your own prior message, "
        f"not another participant.  It also appears as the ``<account>`` "
        f"segment of every ``telegram/<account>/<chat_id>`` address.",
    ]
    if username:
        lines.append(
            f"- **username**: `@{username}` — what users type to mention "
            f"you in chats and how people refer to your bot."
        )
    if first_name:
        lines.append(f"- **first_name**: `{first_name}` — your display name on Telegram.")
    identity = "\n".join(lines) + "\n"
    return identity + "\n" + TELEGRAM_SERVER_INSTRUCTIONS


TELEGRAM_SERVER_INSTRUCTIONS = """\
## chat_id

Each Telegram channel address is path-shaped: ``telegram/<bot_id>/<chat_id>``.
The ``chat_id`` segment is a signed integer:

- Positive for direct messages (the counterparty's user id).
- Negative for groups (``-123456789``) and supergroups (``-1001234567890``).

Pass it verbatim from the channel path — do not decode or modify it.

## Sending messages — `telegram_send`

**Your text responses are NOT sent automatically.** Bare assistant text
is internal monologue; nobody on Telegram sees it. To deliver a message
you MUST call:

    telegram_send(text="your message here")

The chat_id is taken implicitly from your focal channel — aios injects
it on each call. Set focal with the built-in ``switch_channel`` tool.

If you don't call this tool, no one will see your response.

### Formatting — opt in with `parse_mode="html"`

`telegram_send` defaults to plain text: any markdown you write appears
as literal characters.  When you genuinely want emphasis or structure,
pass `parse_mode="html"` and write Markdown — the connector converts
it to Telegram's HTML parse mode.  Supported: ``**bold**``, ``*italic*``,
``__bold__``, ``_italic_``, ``~~strike~~``, ``||spoiler||``, ``` `code` ```,
fenced code blocks, ``[label](url)`` links, and ``> blockquote`` lines.

If you don't need formatting, leave ``parse_mode`` at its default —
plain prose is usually clearer anyway.

### Avoid splitting one thought into multiple messages

Before calling `telegram_send`, ask: "am I about to do some trivial work
and then send again?" If yes, finish the work first and send one
combined message. Back-to-back messages without substantial work in
between feel like a bot narrating its own steps.

Multiple `telegram_send` calls are fine when there is genuinely heavy
work between them (research, file processing, long tasks) and you are
giving progress updates. The test is whether the human would be left
wondering what is happening — if so, send an update.

### Message length

Telegram caps a single message at 4096 characters. Longer messages are
rejected by the API. Keep replies under that — concise is usually
better anyway. If you truly need more, split the content across multiple
``telegram_send`` calls on your own judgment.

## Showing you're working — `telegram_typing`

Before slow work (a long tool run, a heavy LLM hop) you can call
``telegram_typing()`` to show a "typing…" bubble in the chat.  Telegram
displays it for up to 5 seconds or until your next message arrives — no
need to call it on a timer for fast replies.  Use ``action="upload_photo"``
or ``"upload_document"`` etc. before sending media if the upload is
likely to take a moment.

## Editing and deleting — `telegram_edit_message` / `telegram_delete_message`

If you need to revise a message you sent (typo fix, streaming-style
update from "thinking…" to a full answer), use ``telegram_edit_message``
with the ``message_id`` returned by ``telegram_send``.  Same
``parse_mode`` knob applies.  Telegram only lets you edit your own
messages and only for 48 hours after sending.

``telegram_delete_message`` removes one of your messages outright.
In groups, deleting other people's messages requires admin "Delete
Messages" permission — the API will reject the call otherwise.

## Reacting — `telegram_react`

Pass an emoji glyph plus a ``message_id`` to react to that message.
Bots can set at most one reaction per message; pass ``emoji=None`` to
clear yours.  Telegram restricts which emojis bots can use to a curated
allowlist — unsupported emojis are rejected with a Bad Request you can
react to (so to speak) by retrying with a different emoji or skipping
the reaction.

## What inbound looks like

Edits arrive as a fresh inbound carrying ``metadata.edit_of_message_id``
equal to the message_id being edited — the body is the new text.  Treat
this like the user changing their mind, not as a brand-new message.

Reactions arrive with empty body content and ``metadata.reaction``
containing ``target_message_id``, ``old_emojis``, ``new_emojis``.  An
addition has empty ``old_emojis``; a removal has empty ``new_emojis``.
Anonymous-supergroup reactions and custom (premium) emoji reactions
are dropped at the connector boundary.

Stickers arrive as image (or video, for animated stickers) attachments
with the sticker's emoji surfaced as ``metadata.sticker_emoji`` so you
have a textual cue even when the sticker file isn't vision-readable.
"""
