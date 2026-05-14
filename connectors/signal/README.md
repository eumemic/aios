# aios-signal

Signal connector for [aios](../../README.md).  Built on
``aios-connector-http``.  Wraps ``signal-cli`` in single-account
daemon mode: each connector container runs its own ``signal-cli
daemon`` serving exactly one registered phone, and the container's
bearer token resolves to that one ``connection_id`` on the management
API.  Multi-phone deployments use multiple containers.

## Prerequisites

- A registered Signal phone (the connector's Dockerfile installs
  ``signal-cli`` itself; registration is a one-time operator action
  against an existing config dir on the host)
- A reachable aios api with operator credentials (``AIOS_API_KEY``)

## Operator walkthrough

### 1. Register the Signal account on the host

```bash
signal-cli --config /var/lib/aios/signal -a +15551234567 register --captcha signalcaptcha://...
signal-cli --config /var/lib/aios/signal -a +15551234567 verify 123456
```

The config directory holds the account's encryption keys and is the
state volume the connector container mounts.  One container = one
phone = one config directory.

### 2. Provision the connection

```bash
aios connections create \
    --connector signal \
    --account <bot_uuid> \
    --secret phone=+15551234567
# → returns connection_id
```

The bot UUID is the account's ACI UUID (visible via
``signal-cli listAccounts``).  ``--secret phone=...`` stores the
phone number encrypted at rest under ``AIOS_VAULT_KEY``; rotate
later with ``aios connections set-secrets <connection_id> --secret phone=<new>``.

### 3. Issue a connector token

```bash
aios runtime-tokens issue --connection-id <connection_id> --label <label>
# → prints the plaintext token ONCE
```

### 4. Run the connector container

```bash
docker run \
    -e AIOS_URL=https://api.aios.example/ \
    -e AIOS_RUNTIME_TOKEN=aios_runtime_... \
    -e AIOS_SIGNAL_CONFIG_DIR=/var/lib/aios/signal \
    -v /var/lib/aios/signal:/var/lib/aios/signal \
    -v /var/lib/aios/workspaces:/var/lib/aios/workspaces:ro \
    aios-signal:latest
```

The container reads ``AIOS_URL`` and ``AIOS_RUNTIME_TOKEN`` from
env, fetches its phone via ``GET /v1/connectors/runtime/secrets``, spawns
``signal-cli daemon`` against the bind-mounted config dir, and
starts the inbound pump.  The workspace bind-mount is required for
outbound attachments — paths under ``/workspace/...`` resolve to host
files under ``$AIOS_WORKSPACE_ROOT/<session_id>/...``.

### 5. Bind the connection to a session (or template)

```bash
# single_session — every inbound on this phone lands in one session
aios connections attach <connection_id> --session-id <session_id>

# per_chat — each new chat partner spawns a fresh session via a template
aios connections configure-per-chat <connection_id> --template <template_id>
```

#### Operator-curated per-chat bindings

```bash
aios connections recent-chats <connection_id>           # find the chat_id
aios connections bind-chat <connection_id> --chat-id <id> --session-id <sess_id>
aios connections bound-chats <connection_id>
aios connections unbind-chat <connection_id> --chat-id <id>
```

### 6. Message the phone — the agent replies

Inbound messages route to the bound session.  The agent's
``signal_send`` and ``signal_react`` tools take ``chat_id`` from the
focal channel automatically when the session has a focal channel set
via ``switch_channel``.

## Configuration reference

The phone is on connection secrets, not env.  The connector reads
two SDK env vars + four signal-cli deployment-shape vars:

| Env var | Default | Description |
|---|---|---|
| ``AIOS_URL`` | required | Base URL of the aios api |
| ``AIOS_RUNTIME_TOKEN`` | required | Bearer token from ``aios runtime-tokens issue`` |
| ``AIOS_SIGNAL_CONFIG_DIR`` | required | signal-cli config directory; account database lives here |
| ``AIOS_SIGNAL_CLI_BIN`` | ``signal-cli`` | Path to signal-cli binary |
| ``AIOS_SIGNAL_DAEMON_HOST`` | ``127.0.0.1`` | Internal TCP host for signal-cli daemon |
| ``AIOS_SIGNAL_DAEMON_PORT`` | ``7583`` | Internal TCP port for signal-cli daemon |

## Attachments

Inbound photos, voice notes, and documents surface to the model as
``image_url`` content parts (vision-capable minds) or text markers,
via the harness's vision pipeline.  Each file is staged under
``<workspace_root>/_attachments/<session>/signal/...`` and made
readable inside the sandbox at ``/mnt/attachments/signal/...``.

Outbound: pass an ``attachments: list[str]`` parameter to
``signal_send`` alongside ``text``.  Each path must be under
``/workspace/`` or ``/mnt/attachments/`` (the latter is read-only, so
to forward an inbound file ``cp`` it into ``/workspace/`` first).

## Out of scope

- Voice / Say / SoundEffect / Listen tools.
- Group add/remove members.
- Typing indicators.
- Automated registration.

## Development

From ``connectors/signal/``:

```
uv run pytest -q           # unit, ~1.5s, no Docker / no signal-cli
uv run mypy .              # strict
uv run ruff check .
uv run ruff format --check .
```
