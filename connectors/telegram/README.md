# aios-telegram

Telegram connector for [aios](../../README.md). One long-running process per
Telegram bot serves a stateful MCP server. aios uses the same MCP server for
outbound tools (`telegram_send`) and inbound message subscriptions.

## Prerequisites

- Python >= 3.13
- A Telegram bot token (talk to [@BotFather](https://t.me/BotFather))
- A running aios API, worker, and inbound process.

## Install

Pip-installable standalone:

```
pip install ./connectors/telegram
```

Or as a uv workspace member (already wired from the repo root):

```
uv sync --all-packages --dev
```

## Operator walkthrough

### 1. Create the bot

Message [@BotFather](https://t.me/BotFather) on Telegram: `/newbot`. Copy
the token. Learn the bot's numeric id by running a throwaway script
(`curl https://api.telegram.org/bot<TOKEN>/getMe`). Use that numeric id as
the vault credential `account_id`.

### 2. Create an aios vault + credential for the MCP token

```
VLT=$(curl -X POST :8090/v1/vaults -d '{"display_name": "Telegram personal"}' | jq -r .id)

curl -X POST :8090/v1/vaults/$VLT/credentials -d '{
  "mcp_server_url": "http://localhost:9200/mcp",
  "account_id": "<bot-numeric-id-from-step-1>",
  "auth_type": "static_bearer",
  "token": "supersecret"
}'
```

### 3. Add the MCP server to your agent

Add the Telegram MCP server as a normal agent MCP server:

```
{
  "mcp_servers": [
    {"type": "url", "name": "telegram", "url": "http://localhost:9200/mcp"}
  ],
  "tools": [
    {
      "type": "mcp_toolset",
      "mcp_server_name": "telegram",
      "default_config": {
        "permission_policy": {"type": "always_allow"}
      }
    }
  ]
}
```

### 4. Create a session with the Telegram vault

Inbound subscriptions are session-scoped. Create or update a session for the
agent above with `vault_ids: ["$VLT"]`. The `aios inbound` process will
subscribe that session to the Telegram MCP server using the vault credential.

### 5. Start the connector

```
export AIOS_TELEGRAM_MCP_TOKEN=supersecret

python -m aios_telegram start --bot-token 123456:AA...
```

All settings can also be passed via env vars. Full list via
`python -m aios_telegram start --help`.

### 6. Start aios inbound

```
aios inbound
```

### 7. DM the bot — the agent replies

DM the bot from a Telegram client. Watch the aios event log: the inbound
message shows up with `metadata.channel` set to
`telegram/<bot-numeric-id>/<chat-id>`. The agent can focus that channel with
`switch_channel`; `telegram_send` uses the focused channel via MCP `_meta`.

## Configuration reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--bot-token` | `AIOS_TELEGRAM_BOT_TOKEN` | required | Bot token from BotFather |
| `--mcp-bind` | `AIOS_TELEGRAM_MCP_BIND` | `127.0.0.1:9200` | Host:port for MCP server |
| `--mcp-token` | `AIOS_TELEGRAM_MCP_TOKEN` | required | Token MCP clients must present |

## Architecture

`app.run()` wires two tasks under a single `asyncio.TaskGroup`:

1. **PTB `Application`** - python-telegram-bot's asyncio Application drives
   the update loop (long polling). Discovers the bot's numeric id via
   `getMe` on startup. Every incoming message is parsed into an
   `InboundMessage` and published to the connector-local MCP inbound broker.
2. **MCP server** - stateful FastMCP on uvicorn, bearer-auth-gated, exposing
   `telegram_send` and the hidden `aios_inbound_subscribe` hook used by
   `aios inbound`.

**Crash-is-fatal.** Any task failure propagates through the TaskGroup and
exits the process non-zero. There is no auto-reconnect. Run under systemd
(`Restart=on-failure`) or Docker (`restart: unless-stopped`).

## Out of scope for v1

- Inbound media (photos, voice, documents, stickers) — dropped silently.
- Outbound media — no `telegram_send_photo` / `_document`.
- Reactions — punt.
- Message editing / deletion.
- Typing indicators / chat actions.
- Forum topics (`message_thread_id`) — ignored; every message treated as
  top-level in the chat.
- Webhook mode — polling only.
- Markdown rendering — plain text only. The agent's `**bold**` will render
  literally until a v2 adds MarkdownV2 escaping.
- Message splitting — Telegram's 4096-char limit surfaces as a Bad Request
  the model must handle by retrying shorter.
- User allowlist - session and vault selection gate which agents subscribe.
- Auto-reconnect on Telegram outage.

## Troubleshooting

- **MCP calls returning 401**: the `--mcp-token` on the connector doesn't
  match the `token` stored in the aios vault's credential for this
  MCP server.
- **Inbound messages never appear in aios**: check that `aios inbound` is
  running, the session has the vault attached, and the vault credential has
  `account_id` set to the bot numeric id.
- **`Unauthorized` from Telegram on startup**: the bot token is wrong or
  revoked. Re-check with BotFather.

## Development

From `connectors/telegram/`:

```
uv run pytest -q           # unit tests, no network
uv run mypy src tests      # strict
uv run ruff check src tests
uv run ruff format --check src tests
```
