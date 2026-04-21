"""Per-connector affordance prose surfaced to the agent via MCP.

Returned in the ``InitializeResult.instructions`` field of the MCP
``initialize`` response; the aios harness concatenates it into the
session's system prompt under a ``## Connector: signal/<account>``
heading.

Covers only the tools this server actually exposes — ``signal_send``,
``signal_react``, ``signal_read_receipt``.  Telling the model about
tools that don't exist would be worse than silence.

The instructions are composed per-run: ``build_instructions`` prepends
an identity block (bot's own ``sender_uuid`` + phone number) and a
group-roster block (who's in each group the bot is a member of) to
the static body so the agent knows who it is and who it's talking to
without having to learn that from inbound traffic (issues #55, #57).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .daemon import GroupInfo


def build_instructions(
    *,
    bot_uuid: str,
    phone: str,
    profile_name: str | None = None,
    groups: list[GroupInfo] | None = None,
    contact_names: dict[str, str] | None = None,
) -> str:
    """Compose the MCP ``initialize`` instructions with identity + roster.

    The agent otherwise has no reliable source of truth for which
    ``sender_uuid`` is itself — models have been observed confabulating
    identities in group chats when this is absent.  The group-roster
    block makes every participant knowable without having to wait for
    them to speak; silent peers don't effectively disappear.

    ``profile_name``, when set, is the display name peers see for this
    account.  Rendered as a third identity bullet; omitted entirely if
    ``None`` or empty so an un-set-profile account doesn't get a blank
    line in the system prompt.
    """
    identity = (
        "## Your identity on this Signal account\n"
        "\n"
        f"- **sender_uuid**: `{bot_uuid}` — this is YOU.  In group "
        "messages, any inbound whose header shows this uuid is your own "
        "prior message, not another participant.\n"
        f"- **phone**: `{phone}` — this Signal account is identified to "
        "peers by this number.\n"
    )
    if profile_name:
        identity += f"- **profile_name**: `{profile_name}` — your display name on Signal.\n"
    roster = _render_group_roster(bot_uuid, groups or [], contact_names or {})
    return identity + roster + "\n" + SIGNAL_SERVER_INSTRUCTIONS


def _render_group_roster(
    bot_uuid: str,
    groups: list[GroupInfo],
    contact_names: dict[str, str],
) -> str:
    if not groups:
        return ""
    lines: list[str] = ["\n## Your Signal groups\n"]
    for g in groups:
        name = g.name or "(unnamed)"
        lines.append(f"\n- `{g.id}` — {name}")
        for uuid in g.member_uuids:
            if uuid == bot_uuid:
                tag = "(YOU)"
            else:
                display = contact_names.get(uuid)
                tag = display if display else "(name unknown)"
            lines.append(f"    - `{uuid}`: {tag}")
    lines.append("")
    return "\n".join(lines) + "\n"


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
