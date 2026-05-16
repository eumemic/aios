# aios-telegram

Telegram connector for [aios](../../README.md).  Built on
``aios-connector-http``.  Each connector container is one bot — the
container's bearer token resolves to a single ``connection_id`` on
the management API, and that connection holds the bot's identity
(``connector="telegram"``, ``external_account_id=<bot_id>``) plus the encrypted
``bot_token`` secret the connector reads at startup.  Multi-bot
deployments run multiple containers, each with its own connector
token.

## Prerequisites

- A Telegram bot token (talk to [@BotFather](https://t.me/BotFather))
- A reachable aios api with operator credentials (``AIOS_API_KEY``)

## Operator walkthrough

### 1. Create the bot(s)

Message [@BotFather](https://t.me/BotFather) on Telegram: ``/newbot``.
Copy the token.  Repeat for each bot you want to deploy.

### 2. Provision the connection

```bash
aios connections create \
    --connector telegram \
    --external-account-id <bot_id> \
    --secret bot_token=<token>
# → returns connection_id
```

The bot id is the numeric prefix of your bot token (or whatever
``Bot.get_me().id`` would return).  The ``--secret bot_token=...``
flag stores the token encrypted at rest under ``AIOS_VAULT_KEY``;
operator reads after this never return its value.  Rotate later
with ``aios connections set-secrets <connection_id> --secret bot_token=<new>``.

### 3. Issue a connector token

```bash
aios runtime-tokens issue --connection-id <connection_id> --label <label>
# → prints the plaintext token ONCE
```

Capture the plaintext.  The server stores only its hash; if you lose
it you must issue a new one.

### 4. Run the connector container

```bash
docker run \
    -e AIOS_URL=https://api.aios.example/ \
    -e AIOS_RUNTIME_TOKEN=aios_runtime_... \
    -v /var/lib/aios/workspaces:/var/lib/aios/workspaces:ro \
    aios-telegram:latest
```

The container reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from env,
subscribes to ``GET /v1/connectors/connections`` to discover every
active connection of type ``"telegram"``, fetches each connection's
bot token via ``GET /v1/connectors/runtime/secrets``, and starts a
PTB long-polling loop per bot.  The workspace bind-mount is required for outbound
attachments — paths under ``/workspace/...`` resolve to host files
under ``$AIOS_WORKSPACE_ROOT/<session_id>/...``.

### 5. Bind the connection to a session (or template)

For each bot, choose a routing mode:

```bash
# single_session — every inbound on this bot lands in one session
aios connections attach <connection_id> --session-id <session_id>

# per_chat — each new chat partner spawns a fresh session via a template
aios connections configure-per-chat <connection_id> --template <template_id>
```

#### Operator-curated per-chat bindings

To route a specific chat on a bot's account to a specific existing
session — the middle case between attach (whole bot → one session)
and configure-per-chat (each chat → fresh template-spawn) — pre-populate
a binding:

```bash
aios connections recent-chats <connection_id>           # find the chat_id
aios connections bind-chat <connection_id> --chat-id <id> --session-id <sess_id>
aios connections bound-chats <connection_id>            # inspect rows
aios connections unbind-chat <connection_id> --chat-id <id>
```

The binding is consulted before the connection's mode-default
fallback, so it works on top of any mode (single_session, per_chat,
or detached). Inbound on bound chats routes to the bound session;
unbound chats fall back to the connection's default behaviour.

### 6. DM the bot — the agent replies

Inbound messages route to the bound session.  The agent's
``telegram_send`` tool takes ``chat_id`` from the focal channel
automatically when the session has a focal channel set via
``switch_channel``.

## Configuration reference

The connector container reads two env vars (both supplied by
``HttpConnector.__init__`` from process env):

| Env var | Default | Description |
|---|---|---|
| ``AIOS_URL`` | required | Base URL of the aios api |
| ``AIOS_RUNTIME_TOKEN`` | required | Bearer token from ``aios runtime-tokens issue`` |

The bot token lives on the connection's encrypted secrets and is not
read from env.

## Inbound surface

Updates the connector subscribes to (set explicitly via PTB's
``allowed_updates``):

- **``message``** — new messages.  Photos, voice notes, video, audio,
  documents, **stickers** (static ``.webp``, video ``.webm``, animated
  ``.tgs``), **animations** (GIFs, MP4-encoded), and **video notes** all
  flow as attachments via the harness vision pipeline.  Captions become
  the message text.  Sticker emoji is exposed in
  ``metadata.sticker_emoji`` so models that can't see the sticker file
  still get a textual cue.
- **``edited_message``** — edits arrive as a fresh inbound with
  ``metadata.edited == True``.  The ``message_id`` is the same as the
  original (Telegram preserves it across edits), and the body is the
  new (post-edit) text.
- **``message_reaction``** — emoji reactions on messages flow with empty
  body content and ``metadata.reaction`` containing
  ``target_message_id`` plus ``old_emojis`` / ``new_emojis`` as the delta.
  Anonymous-supergroup reactions (``actor_chat``) and custom (premium)
  reactions are dropped at the connector boundary.

Channel posts (no ``from_user``) and bot-to-bot traffic are filtered out
in ``parse.py``.

## Attachments

Outbound: pass an ``attachments: list[str]`` parameter to
``telegram_send`` alongside ``text``.  Type is inferred from extension
— ``.jpg``/``.png`` photo, ``.mp4`` video, ``.ogg`` voice, ``.mp3``
audio, anything else document.  Single attachment uses
``send_photo``/``send_voice``/etc with caption; multiple attachments
use ``send_media_group`` (caption rides on the first item only, per
Telegram's API).  Paths must be under ``/workspace/`` or
``/mnt/attachments/``.

## Outbound tools

| Tool | What it does |
|---|---|
| ``telegram_send`` | Send a message (text + optional attachments).  Optional ``parse_mode="html"`` runs the body through a small Markdown→Telegram-HTML converter (bold/italic/strike/spoiler/code/links/blockquote). |
| ``telegram_typing`` | Show a chat-action bubble (``typing``, ``upload_photo``, ``record_voice``, …).  Useful before slow work. |
| ``telegram_edit_message`` | Replace the text of one of your earlier messages by ``message_id`` (48-hour window). |
| ``telegram_delete_message`` | Delete one of your messages by ``message_id``.  In groups, deleting others' messages requires admin permission. |
| ``telegram_react`` | Set or clear the bot's reaction to a message.  Telegram restricts bot reactions to a curated emoji allowlist. |

All five take ``chat_id`` implicitly from the focal channel.

## Out of scope

- Inline mode and payment-related update types (``shipping_query``,
  ``pre_checkout_query``).
- Inline keyboards / ``callback_query`` handling — would require a
  two-way tool callback path.
- Forum topics (``message_thread_id``) — every message is treated as
  top-level in the chat.
- Webhook mode — polling only.
- Message splitting — Telegram's 4096-char limit surfaces as a Bad
  Request the model handles by retrying shorter.
- User allowlist — connection attachments gate access server-side.

## Development

From ``connectors/telegram/``:

```
uv run pytest -q           # unit tests, no network
uv run mypy .              # strict
uv run ruff check .
uv run ruff format --check .
```
