# aios-signal

Signal connector for [aios](../../README.md). One long-running process per
Signal account — wraps `signal-cli` in daemon mode, ingests inbound messages
into aios via `POST /v1/connections/{id}/messages`, and serves an MCP server
exposing `signal_send`, `signal_react`, `signal_read_receipt` tools that the
aios worker calls back into.

## Prerequisites

- Python ≥ 3.13
- [signal-cli](https://github.com/AsamK/signal-cli) ≥ 0.13.x (provides the JSON-RPC `listAccounts`, `send`, `sendReaction`, `sendReceipt` methods, and `--receive-mode=on-connection`)
- A JRE available to signal-cli
- A running aios instance you can point at (Phase 1 routing infra required)

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

Grab the bot's ACI UUID — you'll need it for the routing rule:

```
signal-cli -a +15551234567 listAccounts
```

### 2. Create an aios vault + credential for the MCP token

```
VLT=$(curl -X POST :8090/v1/vaults -d '{"display_name": "Signal personal"}' | jq -r .id)

curl -X POST :8090/v1/vaults/$VLT/credentials -d '{
  "mcp_server_url": "http://localhost:9100/mcp",
  "auth_type": "static_bearer",
  "token": "supersecret"
}'
```

### 3. Add the MCP server to your agent

Add the Signal MCP server as a normal agent MCP server, and mark its toolset
as focal-channel aware:

```
{
  "mcp_servers": [
    {"type": "url", "name": "signal", "url": "http://localhost:9100/mcp"}
  ],
  "tools": [
    {
      "type": "mcp_toolset",
      "mcp_server_name": "signal",
      "channel_context": {"type": "focal"}
    }
  ]
}
```

### 4. Register the aios connection

The connection is the inbound channel account. Normal MCP discovery comes from
the agent config above; the vault is supplied through the routing rule's
session params.

```
CONN=$(curl -X POST :8090/v1/connections -d "{
  \"connector\": \"signal\",
  \"account\": \"<bot-aci-uuid-from-step-1>\"
}" | jq -r .id)
```

### 5. Start the connector

```
export AIOS_URL=http://localhost:8090
export AIOS_API_KEY=...
export AIOS_CONNECTION_ID=$CONN
export AIOS_SIGNAL_MCP_TOKEN=supersecret

python -m aios_signal start \
  --phone +15551234567 \
  --config-dir ~/.config/signal-cli
```

All settings can also be passed via env vars. Full list via `python -m aios_signal start --help`.

### 6. Add a routing rule

```
curl -X POST :8090/v1/connections/$CONN/routing-rules -d "{
  \"prefix\": \"\",
  \"target\": \"agent:<agent-id>\",
  \"session_params\": {\"environment_id\": \"<env-id>\", \"vault_ids\": [\"$VLT\"]}
}"
```

The empty prefix is the per-connection catch-all. The session vault binding is
what gives the agent-declared Signal MCP server its bearer token.

### 7. DM your bot — the agent replies

DM the bot's phone number from another Signal client. Watch the aios event
log: the inbound message shows up with `metadata.channel` set, a session is
created on first inbound, and the agent's `signal_send` call delivers the
reply back to Signal.

## Configuration reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--phone` | `AIOS_SIGNAL_PHONE` | required | E.164 phone number |
| `--config-dir` | `AIOS_SIGNAL_CONFIG_DIR` | required | signal-cli config directory |
| `--signal-cli-bin` | `AIOS_SIGNAL_CLI_BIN` | `signal-cli` | Path to signal-cli binary |
| `--daemon-port` | `AIOS_SIGNAL_DAEMON_PORT` | `7583` | TCP port for signal-cli daemon |
| `--aios-url` | `AIOS_URL` | required | Base URL of aios API |
| `--aios-api-key` | `AIOS_API_KEY` | required | Bearer token for aios API |
| `--aios-connection-id` | `AIOS_CONNECTION_ID` | required | ID of the pre-registered connection |
| `--mcp-bind` | `AIOS_SIGNAL_MCP_BIND` | `127.0.0.1:9100` | Host:port for MCP server |
| `--mcp-token` | `AIOS_SIGNAL_MCP_TOKEN` | required | Token MCP clients must present |

## Architecture

`app.run()` wires three tasks under a single `asyncio.TaskGroup`:

1. **`SignalDaemon`** — subprocess lifecycle for `signal-cli daemon`. Drains
   stdout/stderr, polls TCP readiness via `listAccounts`, discovers the bot's
   own ACI UUID by matching `--phone` against `listAccounts` output.
2. **`InboundPump`** — drains the daemon's persistent listener connection,
   parses each envelope into an `InboundMessage`, and POSTs to aios with
   the metadata envelope specified in #33.
3. **MCP server** — FastMCP on uvicorn, bearer-auth-gated, exposing
   `signal_send`, `signal_react`, `signal_read_receipt`.

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
- **MCP calls returning 401**: the `--mcp-token` on the connector doesn't
  match the `token` stored in the aios vault's credential for this
  connection. They must match exactly.
- **Inbound messages never appear in aios**: check the connector logs for
  `ingest.client_error` (malformed payload — likely a connector bug) or
  `ingest.retries_exhausted` (aios was unreachable long enough to exhaust
  the 1/2/4/8s backoff).

## Development

From `connectors/signal/`:

```
uv run pytest -q           # unit + integration, ~1.5s, no Docker / no signal-cli
uv run mypy src tests      # strict
uv run ruff check src tests
uv run ruff format --check src tests
```
