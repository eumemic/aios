# aios-signal

Signal connector for [aios](../../README.md). One long-running process per
Signal account wraps `signal-cli` in daemon mode and serves a stateful MCP
server. aios uses the same MCP server for outbound tools (`signal_send`,
`signal_react`) and inbound message subscriptions.

## Prerequisites

- Python ≥ 3.13
- [signal-cli](https://github.com/AsamK/signal-cli) ≥ 0.13.x (provides the JSON-RPC `listAccounts`, `send`, `sendReaction`, `sendReceipt` methods, and `--receive-mode=on-connection`)
- A JRE available to signal-cli
- A running aios API, worker, and inbound process.

## Install

Pip-installable standalone:

```
pip install ./connectors/signal
```

Or as a uv workspace member (already wired from the repo root):

```
uv sync --all-packages --dev
```

## Operator walkthrough

### 1. Register the Signal account

```
signal-cli -a +15551234567 register --captcha signalcaptcha://...
signal-cli -a +15551234567 verify 123456
```

Grab the bot's ACI UUID — you'll use it as the vault credential `account_id`:

```
signal-cli -a +15551234567 listAccounts
```

### 2. Create an aios vault + credential for the MCP token

```
VLT=$(curl -X POST :8090/v1/vaults -d '{"display_name": "Signal personal"}' | jq -r .id)

curl -X POST :8090/v1/vaults/$VLT/credentials -d '{
  "mcp_server_url": "http://localhost:9100/mcp",
  "account_id": "<bot-aci-uuid-from-step-1>",
  "auth_type": "static_bearer",
  "token": "supersecret"
}'
```

### 3. Add the MCP server to your agent

Add the Signal MCP server as a normal agent MCP server:

```
{
  "mcp_servers": [
    {"type": "url", "name": "signal", "url": "http://localhost:9100/mcp"}
  ],
  "tools": [
    {
      "type": "mcp_toolset",
      "mcp_server_name": "signal",
      "default_config": {
        "permission_policy": {"type": "always_allow"}
      }
    }
  ]
}
```

### 4. Create a session with the Signal vault

Inbound subscriptions are session-scoped. Create or update a session for the
agent above with `vault_ids: ["$VLT"]`. The `aios inbound` process will
subscribe that session to the Signal MCP server using the vault credential.

### 5. Start the connector

```
export AIOS_SIGNAL_MCP_TOKEN=supersecret

python -m aios_signal start \
  --phone +15551234567 \
  --config-dir ~/.config/signal-cli
```

All settings can also be passed via env vars. Full list via `python -m aios_signal start --help`.

### 6. Start aios inbound

```
aios inbound
```

### 7. DM your bot

DM the bot's phone number from another Signal client. Watch the aios event
log: the inbound message shows up with `metadata.channel` set to
`signal/<bot-aci-uuid>/<chat-id>`. The agent can focus that channel with
`switch_channel`; `signal_send` and `signal_react` use the focused channel via
MCP `_meta`.

## Configuration reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--phone` | `AIOS_SIGNAL_PHONE` | required | E.164 phone number |
| `--config-dir` | `AIOS_SIGNAL_CONFIG_DIR` | required | signal-cli config directory |
| `--signal-cli-bin` | `AIOS_SIGNAL_CLI_BIN` | `signal-cli` | Path to signal-cli binary |
| `--daemon-port` | `AIOS_SIGNAL_DAEMON_PORT` | `7583` | TCP port for signal-cli daemon |
| `--mcp-bind` | `AIOS_SIGNAL_MCP_BIND` | `127.0.0.1:9100` | Host:port for MCP server |
| `--mcp-token` | `AIOS_SIGNAL_MCP_TOKEN` | required | Token MCP clients must present |

## Architecture

`app.run()` wires three tasks under a single `asyncio.TaskGroup`:

1. **`SignalDaemon`** — subprocess lifecycle for `signal-cli daemon`. Drains
   stdout/stderr, polls TCP readiness via `version`, discovers the bot's
   own ACI UUID from signal-cli's account file.
2. **`InboundPump`** — drains the daemon's persistent listener connection,
   parses each envelope into an `InboundMessage`, and publishes it to the
   connector-local MCP inbound broker.
3. **MCP server** — stateful FastMCP on uvicorn, bearer-auth-gated, exposing
   `signal_send`, `signal_react`, and the hidden `aios_inbound_subscribe`
   hook used by `aios inbound`.

**Crash-is-fatal.** Any task failure propagates through the TaskGroup and
exits the process non-zero. There is no auto-reconnect. Run under systemd
(`Restart=on-failure`) or Docker (`restart: unless-stopped`).

## Out of scope for v1

- Attachments (inbound messages with attachments are posted text-only with `[attachment: <name> (<mime>)]` markers; `signal_send` has no attachment parameter).
- Voice / Say / SoundEffect / Listen tools.
- Group create / rename / add-members.
- Typing indicators.
- Message editing / deletion.
- Automated registration.
- Auto-reconnect on signal-cli crash.

## Troubleshooting

- **`signal.bot_uuid.not_found`**: `signal-cli listAccounts` returned no
  entry matching `--phone`. Re-run registration (step 1).
- **`signal.daemon.crashed`**: signal-cli exited unexpectedly. Check the
  logs for the `signal.daemon.stderr` events immediately preceding. Common
  causes: stale session state, JRE absent, port already in use.
- **MCP calls returning 401**: the `--mcp-token` on the connector does not
  match the `token` stored in the aios vault credential for this MCP server.
- **Inbound messages never appear in aios**: check that `aios inbound` is
  running, the session has the vault attached, and the vault credential has
  `account_id` set to the bot ACI UUID.

## Development

From `connectors/signal/`:

```
uv run pytest -q           # unit + integration, ~1.5s, no Docker / no signal-cli
uv run mypy src tests      # strict
uv run ruff check src tests
uv run ruff format --check src tests
```
