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

## Attachments

Inbound photos, voice notes, documents, video, and audio are
downloaded via `bot.get_file()` and surfaced as `image_url` content
parts (vision-capable minds) or text markers, via the harness's
vision pipeline.  Stickers and animations are dropped; captions
become the message text.

Outbound: pass an `attachments: list[str]` parameter to
`telegram_send` alongside `text`.  Type is inferred from extension
— `.jpg`/`.png` photo, `.mp4` video, `.ogg` voice, `.mp3` audio,
anything else document.  Single attachment uses
`send_photo`/`send_voice`/etc with caption; multiple attachments
use `send_media_group` (caption rides on the first item only, per
Telegram's API).  Paths must be under `/workspace/` or
`/mnt/attachments/`.

## Out of scope for v1

- Stickers and animations (dropped on inbound).
- Reactions — punt.
- Message editing / deletion.
- Typing indicators / chat actions.
- Forum topics (`message_thread_id`) — ignored; every message treated as
  top-level in the chat.
- Webhook mode — polling only.
- Markdown rendering — plain text only.  The agent's `**bold**` will
  render literally until a v2 adds MarkdownV2 escaping.
- Message splitting — Telegram's 4096-char limit surfaces as a Bad
  Request the model must handle by retrying shorter.
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
