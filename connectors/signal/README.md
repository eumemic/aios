# aios-signal

Signal connector for [aios](../../README.md). Wraps `signal-cli` in
multi-account daemon mode and surfaces it as an MCP stdio server that
the aios worker spawns and supervises.  One `signal-cli daemon`
process serves N registered phones natively (no `-a` flag); every RPC
call routes via the `account` field in its params.

## Prerequisites

- Python ≥ 3.13
- [signal-cli](https://github.com/AsamK/signal-cli) ≥ 0.13.x (the
  multi-account JSON-RPC daemon mode is documented in
  `signal-cli-jsonrpc.5.adoc` upstream)
- A JRE available to signal-cli
- A running aios worker that includes `signal` (or `signal:<instance>`)
  in its `connectors_enabled` list

## Operator walkthrough

### 1. Register the Signal account(s)

```
signal-cli --config ~/.aios/connectors/signal -a +15551234567 register --captcha signalcaptcha://...
signal-cli --config ~/.aios/connectors/signal -a +15551234567 verify 123456
```

Repeat for each phone number you want this connector instance to serve.
All registered accounts share one `--config` directory.

### 2. Configure the worker's connector instance

In the worker's environment:

```bash
AIOS_CONNECTORS_ENABLED=signal                   # default-instance shape
AIOS_SIGNAL_PHONES=+15551234567,+15559876543     # CSV — both phones served by one daemon
AIOS_SIGNAL_CONFIG_DIR=~/.aios/connectors/signal
```

For multi-instance deployments (e.g., separate config dirs for
unrelated bots), use the `<connector>:<instance>` syntax and scope env
vars by instance:

```bash
AIOS_CONNECTORS_ENABLED=signal:personal,signal:work
AIOS_SIGNAL_PERSONAL_PHONES=+15551234567
AIOS_SIGNAL_PERSONAL_CONFIG_DIR=~/.aios/connectors/signal-personal
AIOS_SIGNAL_WORK_PHONES=+15559876543,+15558887777
AIOS_SIGNAL_WORK_CONFIG_DIR=~/.aios/connectors/signal-work
```

The supervisor re-exports `AIOS_SIGNAL_<INSTANCE>_*` as
`AIOS_SIGNAL_*` inside each subprocess so the connector reads its
config under the standard prefix.

### 3. Restart the worker

`aios worker` boots, spawns the signal subprocess, runs `signal-cli
daemon`, and reports the discovered accounts on
`aios connectors accounts signal`.

### 4. Attach connections to sessions

For each phone the connector serves, attach the auto-created
`(connector=signal, account=<bot_uuid>)` connection to a session:

```
aios connections list --connector=signal
aios connections attach <conn_id> --session=<session_id>
```

Or configure per-chat session spawning via a session template
(`aios connections configure-per-chat <conn_id> --template=<tpl_id>`).

### 5. DM the bot — the agent replies

Inbound messages on each phone route to its connection's session.
The agent's `signal_send` and `signal_react` tools take `account` +
`chat_id` from the focal channel meta; aios injects them automatically
when the session has a focal channel set via `switch_channel`.

## Configuration reference

| Env var | Default | Description |
|---|---|---|
| `AIOS_SIGNAL_PHONES` | required | CSV of E.164 phone numbers; all must be registered in `AIOS_SIGNAL_CONFIG_DIR` |
| `AIOS_SIGNAL_CONFIG_DIR` | required | signal-cli config directory; account database lives here |
| `AIOS_SIGNAL_CLI_BIN` | `signal-cli` | Path to signal-cli binary |
| `AIOS_SIGNAL_DAEMON_HOST` | `127.0.0.1` | Internal TCP host for signal-cli daemon |
| `AIOS_SIGNAL_DAEMON_PORT` | `7583` | Internal TCP port for signal-cli daemon |

## Migration from single-phone PR3

Pre-this-PR signal connectors used `AIOS_SIGNAL_PHONE` (singular).
That env var is removed cleanly — set `AIOS_SIGNAL_PHONES=+1...` (a
one-element CSV) for an equivalent single-phone deployment.  Multi-
phone setups become `AIOS_SIGNAL_PHONES=+1...,+2...`.

## Attachments

Inbound photos, voice notes, and documents surface to the model as
`image_url` content parts (vision-capable minds) or text markers,
via the harness's vision pipeline.  Each file is staged under
`<workspace_root>/_attachments/<session>/signal/...` and made
readable inside the sandbox at `/mnt/attachments/signal/...`.

Outbound: pass an `attachments: list[str]` parameter to
`signal_send` alongside `text`.  Each path must be under
`/workspace/` or `/mnt/attachments/` (the latter is read-only, so to
forward an inbound file `cp` it into `/workspace/` first).

## Out of scope for v1

- Voice / Say / SoundEffect / Listen tools.
- Group create / rename / add-members.
- Typing indicators.
- Message editing / deletion.
- Automated registration.

## Development

From `connectors/signal/`:

```
uv run pytest -q           # unit, ~1.5s, no Docker / no signal-cli
uv run mypy src tests      # strict
uv run ruff check src tests
uv run ruff format --check src tests
```
