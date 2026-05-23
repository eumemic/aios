# aios-whatsapp

WhatsApp connector for aios. Pairs a WhatsApp account via the unofficial
Multi-Device protocol (backed by [`go.mau.fi/whatsmeow`](https://github.com/tulir/whatsmeow))
and surfaces inbound messages into an aios session as channel-tagged inbounds
the model can respond to with a small toolset.

Mirrors `connectors/signal`'s shape: a Python `HttpConnector` subclass spawns a
Go daemon (`whatsapp-daemon`) as a subprocess and talks to it over
line-delimited JSON-RPC on a loopback TCP port.

## Tool surface

The connector publishes these model-facing tools (auto-discovered by the aios
runtime via the connector's `tools_schema`):

| Tool | Purpose |
|---|---|
| `whatsapp_send(text, attachments?)` | Send a message; text supports CommonMark inline emphasis which is auto-converted to WhatsApp's inline syntax. |
| `whatsapp_react(message_id, reaction)` | React with an emoji; empty `reaction` clears a prior reaction. |
| `whatsapp_edit_message(message_id, text)` | Rewrite your own outbound (within WhatsApp's ~15-minute edit window). |
| `whatsapp_delete_message(message_id)` | Delete-for-everyone your own outbound. |
| `whatsapp_list_groups()` | List every group the bot is a member of. |
| `whatsapp_create_group(name, participants)` | Create a new group; `participants` are +E.164 phones. |
| `whatsapp_rename_group(chat_id, name)` | Rename a group the bot is an admin in. |

Read receipts fire automatically when the bot sends to a chat — all prior peer
messages on that chat are MarkRead'd in one batch (Signal-style implicit
receipts-after-emit). No tool surface for this; happens inside `whatsapp_send`.

## Pairing

The connector publishes three operator-side management endpoints which the
`aios whatsapp` CLI wraps:

```bash
# 1. Begin a QR pair session — prints a wa.me linked-device URL.
aios whatsapp start-pairing +15551234567

# 2. Scan the QR from WhatsApp → Settings → Linked Devices → Link a Device,
#    then block until the handshake completes.
aios whatsapp confirm-pairing +15551234567

# 3. (Later) unlink server-side and clear local store.
aios whatsapp unpair +15551234567
```

The CLI hits the equivalent `POST /v1/connectors/whatsapp/{start,confirm,unpair}-pairing`
endpoints directly if you prefer scripting against the API. The daemon swaps
in a fresh whatsmeow.Client behind an `atomic.Pointer` after `unpair` so the
next `start-pairing` works in the same daemon process without a restart.

## Data directory

The daemon writes per-phone state under `<data_dir>/<phone>/`:

```
<data_dir>/+15551234567/
├── store.db        # whatsmeow sqlstore (device identity, sessions, keys)
├── messages.db     # daemon-side message-key index (powers react/edit/delete)
└── media/          # decrypted inbound media (mode 0600)
```

`<data_dir>` defaults to `/var/lib/aios/whatsapp-data` inside the container
(env: `AIOS_WHATSAPP_DATA_DIR`). Bind-mount a host volume there to persist
state across container rebuilds; the entrypoint chowns the dir to uid 1000 so
the api container can read the inbound media files when staging them into the
session event log.

## Running

### Docker (production-shaped)

```bash
docker build -f connectors/whatsapp/Dockerfile -t aios-whatsapp .

docker run --rm \
  -v aios-whatsapp-data:/var/lib/aios/whatsapp-data \
  -e AIOS_URL=http://aios-api:8080 \
  -e AIOS_RUNTIME_TOKEN=<token from POST /v1/runtime-tokens> \
  aios-whatsapp
```

Multi-arch: the Dockerfile uses `ARG TARGETARCH` so `docker buildx build
--platform linux/amd64,linux/arm64` produces a multi-arch image. The Go
daemon builds natively for both architectures (whatsmeow is pure Go, no CGo
deps).

### Local development

The connector needs Go 1.25+ to build the daemon (whatsmeow requires it).
For local dev where the host has Go 1.18 or older:

```bash
# Cross-compile the daemon via Docker.
docker run --rm \
  -v "$PWD/connectors/whatsapp/daemon:/src" \
  -v aios-whatsapp-gocache:/go \
  -w /src \
  -e GOPATH=/go \
  -e GOCACHE=/go/build-cache \
  -e GOOS=darwin -e GOARCH=arm64 -e CGO_ENABLED=0 \
  golang:1.25 go build -o /go/bin/whatsapp-daemon ./cmd/whatsapp-daemon

# Run the Python connector with the cross-built daemon.
AIOS_URL=http://127.0.0.1:8080 \
AIOS_RUNTIME_TOKEN=<token> \
AIOS_WHATSAPP_DAEMON_BIN=/Users/tom/.go-cache/bin/whatsapp-daemon \
AIOS_WHATSAPP_DATA_DIR=/tmp/aios-whatsapp-dev \
uv run python -m aios_whatsapp
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AIOS_URL` | _required_ | aios api base URL (e.g. `http://127.0.0.1:8080`). |
| `AIOS_RUNTIME_TOKEN` | _required_ | Connector runtime token (mint via `POST /v1/runtime-tokens`). |
| `AIOS_WHATSAPP_DAEMON_BIN` | `/usr/local/bin/whatsapp-daemon` | Path to the compiled Go daemon binary. |
| `AIOS_WHATSAPP_DATA_DIR` | `/var/lib/aios/whatsapp-data` | Per-phone state dir; one subdir per paired phone. |
| `AIOS_WHATSAPP_DAEMON_HOST` | `127.0.0.1` | Loopback address the daemon listens on. |

## Architecture

```
                    HTTP (SSE + multipart)
   ┌────────────┐   ◄──────────────────►   ┌──────────────────────┐
   │ aios api/  │                          │ aios_whatsapp (this) │
   │ worker     │                          │                      │
   └────────────┘                          └──────────┬───────────┘
                                                      │  spawns
                                                      ▼
                                          ┌────────────────────────┐
                                          │ whatsapp-daemon (Go)   │
                                          │  - whatsmeow client    │
                                          │  - sqlstore (store.db) │
                                          │  - messages.db (PR 5)  │
                                          │  - media/ (PR 5)       │
                                          └─────────┬──────────────┘
                                                    │  TLS
                                                    ▼
                                              WhatsApp servers
```

The Python connector and Go daemon communicate via line-delimited JSON-RPC
2.0 over a loopback TCP port (ephemeral, allocated per connection). The
daemon owns the WhatsApp protocol state (whatsmeow client, sqlstore,
message-key index); the Python side owns aios integration (session routing,
attachment marshalling, tool dispatch).
