"""Per-connector affordance prose surfaced to the agent via MCP.

Returned in the ``InitializeResult.instructions`` field of the MCP
``initialize`` response; the aios harness concatenates it into the
session's system prompt under a ``## Connector: telegram/<account>``
heading.

Covers only the tool this server exposes — ``telegram_send``. Telling the
model about tools that don't exist would be worse than silence.
"""

from __future__ import annotations

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

### Plain text only — no markdown

v1 sends messages without a ``parse_mode``. Any markdown you write
(``**bold**``, ``*italic*``, backticks, links) will appear as literal
characters in the recipient's chat, not as formatted text.

Write plain prose. If you need emphasis, use word choice, not syntax.

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
"""
