# aios-telegram

Telegram connector for [aios](../../README.md). One long-running process per
Telegram bot — ingests inbound messages into aios via
`POST /v1/connections/{id}/messages`, and serves an MCP server exposing a
`telegram_send` tool that the aios worker calls back into.

## Prerequisites

- Python ≥ 3.13
- A Telegram bot token (talk to [@BotFather](https://t.me/BotFather))
- A running aios instance with the connector/channel routing infra

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
(`curl https://api.telegram.org/bot<TOKEN>/getMe`) — you'll need it for
the routing rule.

### 2. Create an aios vault + credential for the MCP token

```
VLT=$(curl -X POST :8090/v1/vaults -d '{"display_name": "Telegram personal"}' | jq -r .id)

curl -X POST :8090/v1/vaults/$VLT/credentials -d '{
  "mcp_server_url": "http://localhost:9200/mcp",
  "auth_type": "static_bearer",
  "token": "supersecret"
}'
```

### 3. Add the MCP server to your agent

Add the Telegram MCP server as a normal agent MCP server, and mark its toolset
as focal-channel aware:

```
{
  "mcp_servers": [
    {"type": "url", "name": "telegram", "url": "http://localhost:9200/mcp"}
  ],
  "tools": [
    {
      "type": "mcp_toolset",
      "mcp_server_name": "telegram",
      "default_config": {"permission_policy": {"type": "always_allow"}},
      "channel_context": {"type": "focal"}
    }
  ]
}
```

### 4. Register the aios connection

The connection is the inbound channel account. `mcp_url` and `vault_id` are
still required by the current API for legacy compatibility; normal MCP
discovery comes from the agent config above.

```
CONN=$(curl -X POST :8090/v1/connections -d "{
  \"connector\": \"telegram\",
  \"account\": \"<bot-numeric-id-from-step-1>\",
  \"mcp_url\": \"http://localhost:9200/mcp\",
  \"vault_id\": \"$VLT\"
}" | jq -r .id)
```

### 5. Start the connector

```
export AIOS_URL=http://localhost:8090
export AIOS_API_KEY=...
export AIOS_CONNECTION_ID=$CONN
export AIOS_TELEGRAM_MCP_TOKEN=supersecret

python -m aios_telegram start --bot-token 123456:AA...
```

All settings can also be passed via env vars. Full list via
`python -m aios_telegram start --help`.

### 6. Add a routing rule

```
curl -X POST :8090/v1/connections/$CONN/routing-rules -d "{
  \"prefix\": \"\",
  \"target\": \"agent:<agent-id>\",
  \"session_params\": {\"environment_id\": \"<env-id>\", \"vault_ids\": [\"$VLT\"]}
}"
```

The empty prefix is the per-connection catch-all. The session vault binding is
what gives the agent-declared Telegram MCP server its bearer token.

### 7. DM the bot — the agent replies

DM the bot from a Telegram client. Watch the aios event log: the inbound
message shows up with `metadata.channel` set, a session is created on
first inbound, and the agent's `telegram_send` call delivers the reply
back to Telegram.

## Configuration reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--bot-token` | `AIOS_TELEGRAM_BOT_TOKEN` | required | Bot token from BotFather |
| `--aios-url` | `AIOS_URL` | required | Base URL of aios API |
| `--aios-api-key` | `AIOS_API_KEY` | required | Bearer token for aios API |
| `--aios-connection-id` | `AIOS_CONNECTION_ID` | required | ID of the pre-registered connection |
| `--mcp-bind` | `AIOS_TELEGRAM_MCP_BIND` | `127.0.0.1:9200` | Host:port for MCP server |
| `--mcp-token` | `AIOS_TELEGRAM_MCP_TOKEN` | required | Token MCP clients must present |

## Architecture

`app.run()` wires two tasks under a single `asyncio.TaskGroup`:

1. **PTB `Application`** — python-telegram-bot's asyncio Application drives
   the update loop (long polling). Discovers the bot's numeric id via
   `getMe` on startup. Every incoming message is parsed into an
   `InboundMessage` and posted to aios via
   `POST /v1/connections/{id}/messages`.
2. **MCP server** — FastMCP on uvicorn, bearer-auth-gated, exposing
   `telegram_send`.

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
- User allowlist — routing rules gate access server-side.
- Auto-reconnect on Telegram outage.

## Troubleshooting

- **MCP calls returning 401**: the `--mcp-token` on the connector doesn't
  match the `token` stored in the aios vault's credential for this
  connection.
- **Inbound messages never appear in aios**: check the connector logs for
  `ingest.client_error` (malformed payload — likely a connector bug) or
  `ingest.retries_exhausted` (aios was unreachable long enough to exhaust
  the 1/2/4/8s backoff).
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
