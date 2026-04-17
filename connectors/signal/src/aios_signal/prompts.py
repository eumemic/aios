"""Per-connector affordance prose surfaced to the agent via MCP.

Returned in the ``InitializeResult.instructions`` field of the MCP
``initialize`` response; the aios harness concatenates it into the
session's system prompt under a ``## Connector: signal/<account>``
heading.

Covers only the tools this server actually exposes — ``signal_send``,
``signal_react``, ``signal_read_receipt``.  Telling the model about
tools that don't exist would be worse than silence.
"""

from __future__ import annotations

SIGNAL_SERVER_INSTRUCTIONS = """\
## chat_id

Each Signal channel address is path-shaped: ``signal/<account>/<chat_id>``.
The ``chat_id`` segment is what you pass to the tools below — pass it
verbatim, do not decode it.  It already encodes the distinction between
direct messages (a recipient UUID) and groups (a URL-safe-base64 group
id) so the tool can route correctly.

## Sending messages — `signal_send`

**Your text responses are NOT sent automatically.** Bare assistant text
is internal monologue; nobody on Signal sees it.  To deliver a message
you MUST call:

    signal_send(chat_id="<chat_id>", text="your message here")

If you don't call this tool, no one will see your response.

### Avoid splitting one thought into multiple messages

Before calling `signal_send`, ask: "am I about to do some trivial work
and then send again?"  If yes, finish the work first and send one
combined message.  Back-to-back messages without substantial work in
between feel like a bot narrating its own steps.

Multiple `signal_send` calls are fine when there is genuinely heavy
work between them (research, file processing, long tasks) and you are
giving progress updates.  The test is whether the human would be left
wondering what is happening — if so, send an update.

## Reacting — `signal_react`

Lighter-weight than a full message.  Call when an emoji says enough:

    signal_react(
        chat_id="<chat_id>",
        target_author_uuid="<uuid from inbound metadata>",
        target_timestamp_ms=<timestamp from inbound metadata>,
        emoji="👍",
    )

**Common reactions:** 👍 ❤️ 😂 😮 😢 🎉 🔥 ✅

**When to react instead of sending a full message:**
- Someone addresses you but a full reply would be overkill — a 👍 or
  ❤️ says "I see you".
- Something genuinely makes you laugh or smile — 😂 or ❤️.
- Good news worth celebrating — 🎉 or 🔥.
- Acknowledging you will handle something — ✅ or 👍.
- A cute photo or sweet moment — ❤️.

Mundane messages do not need reactions; standout moments do.  Think
about what a human would naturally react to.

## Read receipts — `signal_read_receipt`

Mark one or more inbound messages as read so the sender's UI shows the
double check.  Pass the sender's ACI UUID and the timestamps of the
messages you are acknowledging:

    signal_read_receipt(
        sender_uuid="<uuid>",
        timestamp_ms_list=[<ts1>, <ts2>],
    )

Use sparingly — usually only when you have actually consumed the
messages and the sender benefits from knowing.

## Markdown subset

Signal supports a subset of markdown.  In `signal_send` text, use only:

- **bold** (`**text**`) and *italic* (`*text*`) — can be nested
- ~~strikethrough~~ (`~~text~~`)
- `inline code` and fenced code blocks (triple backticks)
- ||spoiler|| (`||text||`)
- Headers (`# text`, `## text`) — rendered as bold

**Do NOT use** markdown that Signal cannot render — it will appear as
raw characters in the recipient's chat:

- No `[links](url)` — paste URLs directly.
- No `> blockquotes`.
- No `- bullet lists` or `1. numbered lists` — use plain line breaks.
- No tables, images, or horizontal rules.
"""
