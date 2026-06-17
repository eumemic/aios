# aios-slack

Slack connector for [aios](../../README.md).  Built on
``aios-connector-http``.  One container serves N connections of type
``slack``; each connection is one Slack workspace install
(``connector="slack"``, ``external_account_id=<team_id>``) holding the
encrypted ``bot_token`` (``xoxb-…``) + ``app_token`` (``xapp-…``) the
connector reads at ``serve_connection`` spawn.

> **Status — MVP slices 1–3/4.** Slice 1 stood up the package, the
> Socket-Mode transport, and the ``serve_connection`` lifecycle. Slice 2
> added inbound normalization + the four connector-side gates
> (self/bot-loop, cross-app/team, subtype, mention-gate). Slice 3 adds
> the outbound reply layer: the ``slack_send`` and ``slack_react``
> ``@tool``\ s, plus the markdown→``mrkdwn`` pipeline and hard clamps in
> `format.py`. A live DM-round-trip smoke lands in slice D. See
> `docs/design/slack-connector.md` §3.1–§3.6.

## Outbound tools (slice 3, §3.5)

The model is heard on Slack only through two tools; ``connection_id`` and
``chat_id`` are server-authoritative (injected by the SDK from the call's
focal channel — the model cannot pick a workspace or conversation):

- **``slack_send(text, thread_ts=None)`` → ``{ts, channel}``** —
  ``chat.postMessage(channel=chat_id, text=…, mrkdwn=True)``. ``text`` is
  written in Markdown and rendered to Slack ``mrkdwn`` then clamped to the
  per-message ceiling before the call. ``thread_ts`` (read off the inbound
  metadata header) threads the reply; default ``None`` posts top-level.
  ``channel`` = ``self.focal_channel(team_id, chat_id)``.
- **``slack_react(message_ts, emoji)`` → ``{status}``** —
  ``reactions.add(channel=chat_id, timestamp=message_ts, name=emoji)``
  (colon-stripped + normalized) when ``emoji`` is set. Mirrors
  ``telegram_react``.

> **Known v0 property — ``slack_send`` is at-least-once** (design §4). A
> ``tool-result`` POST failure *after* a successful ``chat.postMessage``
> re-dispatches the call on reconnect and posts a duplicate. The
> ``SqliteAnsweredSpool`` does not close this window; a deterministic
> idempotency-key fix is a separate SDK follow-up (benefits telegram too).

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
