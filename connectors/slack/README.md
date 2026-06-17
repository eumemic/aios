# aios-slack

Slack connector for [aios](../../README.md).  Built on
``aios-connector-http``.  One container serves N connections of type
``slack``; each connection is one Slack workspace install
(``connector="slack"``, ``external_account_id=<team_id>``) holding the
encrypted ``bot_token`` (``xoxb-…``) + ``app_token`` (``xapp-…``) the
connector reads at ``serve_connection`` spawn.

> **Status — MVP slice 1/4 (the connection layer).** This slice stands
> up the package, the Socket-Mode transport, and the
> ``serve_connection`` lifecycle.  The socket listener acks each
> envelope and pushes the *raw* event onto a per-connection queue; it
> does **not** yet parse, gate, or emit inbound events (slice B), and
> ships no outbound ``slack_send`` / ``slack_react`` tools yet (a later
> slice).  See `docs/design/slack-connector.md` §3.1–§3.3.

## Transport: Socket Mode

Socket Mode is the v0 transport.  The only inbound channel is the
authenticated WebSocket the sidecar dials with the ``xapp-`` app token,
so there is no public ingress, no signing secret, and no
``url_verification`` handshake.

The async ``AsyncSocketModeClient`` does **not** auto-acknowledge.  The
socket listener therefore **acks first** — it
``send_socket_mode_response(SocketModeResponse(envelope_id=…))`` at the
very top, before any parsing or enqueueing — so the ~3-second ack window
is never missed under load.  Missing it makes Slack redeliver (up to 3×)
then throttle/close the socket, a silently-dead bot.

## Prerequisites

- A Slack app created from the manifest (with ``socket_mode_enabled:
  true``) and installed to your workspace, yielding a bot token
  (``xoxb-…``) and an app-level token (``xapp-…``).
- A reachable aios api with operator credentials (``AIOS_API_KEY``).

## Operator walkthrough

### 1. Provision the connection

```bash
aios connections create \
    --connector slack \
    --external-account-id <team_id> \
    --secret bot_token=xoxb-... \
    --secret app_token=xapp-...
# → returns connection_id
```

``<team_id>`` is the Slack workspace id (``T…``) — the durable install
identity.  At ``serve_connection`` the connector calls ``auth.test`` and
**refuses to serve** (fail-closed, INV-5) if the reported ``team_id``
does not match this ``external_account_id``, so a wrong-token paste is a
loud refusal rather than a silent split-brain.  The ``--secret`` flags
store the tokens encrypted at rest under ``AIOS_VAULT_KEY``.

### 2. Issue a connector token and run the container

Issue a ``slack`` connector-type runtime token, then run the container
with ``AIOS_URL`` + ``AIOS_RUNTIME_TOKEN`` in its environment:

```bash
docker run --rm \
    -e AIOS_URL=https://your-aios \
    -e AIOS_RUNTIME_TOKEN=aios_runtime_... \
    aios-slack
```
