# aios-telegram

Telegram connector for [aios](../../README.md).  Per-account paradigm:
each Telegram bot token is a distinct platform identity (PTB's
`Application` is bound 1:1 to a token), so this connector runs as one
subprocess per bot.  Operators deploy multiple bots by listing
multiple instances under one connector type at the supervisor.

## Prerequisites

- Python ≥ 3.13
- A Telegram bot token (talk to [@BotFather](https://t.me/BotFather))
- A running aios worker that includes `telegram` (or
  `telegram:<instance>`) in its `connectors_enabled` list

## Operator walkthrough

### 1. Create the bot(s)

Message [@BotFather](https://t.me/BotFather) on Telegram: `/newbot`.
Copy the token.  Repeat for each bot you want to deploy.

### 2. Configure the worker's connector instance(s)

Single bot — default-instance shape:

```bash
AIOS_CONNECTORS_ENABLED=telegram
AIOS_TELEGRAM_BOT_TOKEN=123456:AA...
```

Multiple bots — explicit instances with scoped env:

```bash
AIOS_CONNECTORS_ENABLED=telegram:support,telegram:alerts
AIOS_TELEGRAM_SUPPORT_BOT_TOKEN=111:BB...   # → AIOS_TELEGRAM_BOT_TOKEN inside support
AIOS_TELEGRAM_ALERTS_BOT_TOKEN=222:CC...    # → AIOS_TELEGRAM_BOT_TOKEN inside alerts
```

The supervisor re-exports `AIOS_TELEGRAM_<INSTANCE>_*` as
`AIOS_TELEGRAM_*` inside each subprocess so the connector reads its
config under the standard prefix.  Each instance runs as its own OS
process — one bot's PTB `Application` crashing doesn't affect siblings.

### 3. Restart the worker

`aios worker` boots, spawns one telegram subprocess per instance, and
each runs `Bot.get_me()` to identify itself.  Verify with
`aios connectors list`.

### 4. Attach connections to sessions

For each bot the connector serves, attach the auto-created
`(connector=telegram, account=<bot_id>)` connection to a session:

```
aios connections list --connector=telegram
aios connections attach <conn_id> --session=<session_id>
```

#### Operator-curated per-chat bindings

To route a specific chat on a bot's account to a specific existing
session — the middle case between attach (whole bot → one session)
and configure-per-chat (each chat → fresh template-spawn) — pre-populate
a binding:

```
aios connections recent-chats <conn_id>           # find the chat_id
aios connections bind-chat <conn_id> --chat-id=<id> --session-id=<sess_id>
aios connections bound-chats <conn_id>            # inspect operator + supervisor rows
aios connections unbind-chat <conn_id> --chat-id=<id>
```

The binding is consulted before the connection's mode-default fallback,
so it works on top of any mode (single_session, per_chat, or even
detached). Inbound on bound chats routes to the bound session;
unbound chats fall back to the connection's default behaviour.

### 5. DM the bot — the agent replies

Inbound messages on each bot route to its connection's session.  The
agent's `telegram_send` tool takes only `chat_id` from the focal
channel meta; aios injects it automatically when the session has a
focal channel set via `switch_channel`.

## Configuration reference

| Env var | Default | Description |
|---|---|---|
| `AIOS_TELEGRAM_BOT_TOKEN` | required | Bot token from BotFather |

When deploying multiple instances, set
`AIOS_TELEGRAM_<INSTANCE_UPPER>_BOT_TOKEN` for each instance (e.g.
`AIOS_TELEGRAM_SUPPORT_BOT_TOKEN`).  Instance names match
`^[a-z][a-z0-9_]*$`.

## Migration from pre-cloister `~/.aios/connectors`

See [`../MIGRATION.md`](../MIGRATION.md#per-instance-cloister-238) — connector
state moved to `~/.aios/instances/<instance_id>/connectors/<name>/` in #238.

## Inbound surface

Updates the connector subscribes to (set explicitly via PTB's
`allowed_updates`):

- **`message`** — new messages.  Photos, voice notes, video, audio,
  documents, **stickers** (static `.webp`, video `.webm`, animated
  `.tgs`), **animations** (GIFs, MP4-encoded), and **video notes** all
  flow as attachments via the harness vision pipeline.  Captions become
  the message text.  Sticker emoji is exposed in
  `metadata.sticker_emoji` so models that can't see the sticker file
  still get a textual cue.
- **`edited_message`** — edits arrive as a fresh inbound with
  `metadata.edit_of_message_id` set to the message_id being edited.
  The body is the new (post-edit) text.
- **`message_reaction`** — emoji reactions on messages flow with empty
  body content and `metadata.reaction` containing
  `target_message_id` plus `old_emojis` / `new_emojis` as the delta.
  Anonymous-supergroup reactions (`actor_chat`) and custom (premium)
  reactions are dropped at the connector boundary.

Channel posts (no `from_user`) and bot-to-bot traffic are filtered out
in `parse.py`.

## Attachments

Outbound: pass an `attachments: list[str]` parameter to
`telegram_send` alongside `text`.  Type is inferred from extension
— `.jpg`/`.png` photo, `.mp4` video, `.ogg` voice, `.mp3` audio,
anything else document.  Single attachment uses
`send_photo`/`send_voice`/etc with caption; multiple attachments
use `send_media_group` (caption rides on the first item only, per
Telegram's API).  Paths must be under `/workspace/` or
`/mnt/attachments/`.

## Outbound tools

| Tool | What it does |
|---|---|
| `telegram_send` | Send a message (text + optional attachments).  Optional `parse_mode="html"` runs the body through a small Markdown→Telegram-HTML converter (bold/italic/strike/spoiler/code/links/blockquote). |
| `telegram_typing` | Show a chat-action bubble (`typing`, `upload_photo`, `record_voice`, …).  Useful before slow work. |
| `telegram_edit_message` | Replace the text of one of your earlier messages by `message_id` (48-hour window). |
| `telegram_delete_message` | Delete one of your messages by `message_id`.  In groups, deleting others' messages requires admin permission. |
| `telegram_react` | Set or clear the bot's reaction to a message.  Telegram restricts bot reactions to a curated emoji allowlist. |

All five take `chat_id` implicitly from the focal channel via
`focal_required`.

## Out of scope

- Inline mode and payment-related update types (`shipping_query`,
  `pre_checkout_query`).
- Inline keyboards / `callback_query` handling — would require a
  two-way tool callback path; not in this PR.
- Forum topics (`message_thread_id`) — every message is treated as
  top-level in the chat.
- Webhook mode — polling only.
- Message splitting — Telegram's 4096-char limit surfaces as a Bad
  Request the model handles by retrying shorter.
- User allowlist — connection attachments gate access server-side.
- Auto-reconnect on Telegram outage (the supervisor restarts the whole
  subprocess on PTB exit).

## Development

From `connectors/telegram/`:

```
uv run pytest -q           # unit tests, no network
uv run mypy src tests      # strict
uv run ruff check src tests
uv run ruff format --check src tests
```
