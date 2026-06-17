# AIOS Slack Connector — Final Design

## 1. Summary

We are building a **`connector='slack'` sidecar** on the existing `aios-connector-http` SDK, mirroring the telegram connector skeleton (`connectors/slack/src/aios_slack/{connector,parse,format,prompts}.py` + `__main__.py` + `Dockerfile` + `pyproject.toml`). It uses **Socket Mode only** for v0, and writes **only** Slack glue — AIOS's existing runtime contract, three-tier resolver, dedup ledger, per-account-subkey secret store, and capability lattice provide everything else with **zero new core endpoints**.

The single most important design insight, in two clauses:

- **Outbound is model tool calls.** The model never "talks to Slack." It emits `type="custom"` tool calls (`slack_send`, `slack_react`) that the harness does not execute; they stream to the connector over `GET /runtime/calls`, the connector calls the Slack Web API, and the result returns via `POST /runtime/tool-results`. The connector's outbound surface is the published `tools_schema` row, and nothing more.
- **The connector is a credential-isolated sidecar.** It is a separate process holding one connector-type runtime bearer plus per-connection Slack bot/app tokens fetched at spawn. It speaks HTTP+SSE to the API. The AIOS master key, the all-accounts pool, and every other tenant's data are structurally unreachable from it.

Two genuine "where AIOS can't" lines are drawn honestly and deferred: **native word-by-word streaming** (no harness primitive pushes in-flight model tokens out to a connector) and **true form modals** (`view_submission` has no first-class inbound shape).

A correction the adversarial review forced into the spine: the security model is **mostly** existing-invariant foreclosure, but it rests on **two honest lines of connector discipline** — the bot-loop filter and the mention-gate. These are pure functions the connector chooses to run; no core invariant backstops them. We name them as discipline rather than dress them as foreclosure.

---

## 2. Background: the AIOS connector contract

A reader who has never seen the code needs four facts about the seam.

**A connector is a sidecar, not in-tree.** It is a standalone process/container that authenticates with **one per-connector-TYPE bearer token** (`aios_runtime_<32-byte-base64url>`), hosts N connections of a single connector type, and speaks HTTP+SSE to the AIOS API. The API process never runs tools or calls models; the worker process does. The connector is a third party to both.

**The runtime token resolves to a tenant scope.** `require_runtime_auth` (`src/aios/api/deps.py`) resolves the bearer to `(token_id, connector, account_id, connection_ids)`. The matched row's `account_id` becomes the authenticated tenant for every downstream query. `connection_ids` is the #350 allowlist: `None` = unscoped (sees every connection of its type), a list (including `[]`) restricts. Two scope checks gate per-connection ops: `_check_runtime_scope` (connector-type match → 403) and `_check_runtime_connection_scope` (allowlist membership → 403).

**The five runtime routes that carry the whole loop** (all `RuntimeAuthDep`):

| Route | Direction | Purpose |
|---|---|---|
| `GET /v1/connectors/connections` (SSE) | discovery | backfills `added` for every active connection of the type, then tails `connections_<connector>` NOTIFY |
| `POST /v1/connectors/runtime/inbound` (multipart, 201) | transport→model | idempotent on `event_id`; calls `handle_inbound` (dedup, resolve, stage, wake) |
| `GET /v1/connectors/runtime/calls` (SSE) | model→transport | streams pending custom tool calls `{session_id, tool_call_id, name, arguments, focal_channel, connection_id, workspace_path}` |
| `POST /v1/connectors/runtime/tool-results` (201) | transport→model | returns a custom tool result; appends `tool_result`, `defer_wake` |
| `GET /v1/connectors/runtime/secrets?connection_id=…` | — | the only decrypt path for a connection's bot/app tokens |

Plus `POST /v1/connectors/runtime/lifecycle` (append a `kind=lifecycle` event to bound sessions) and `PUT /v1/connectors/{connector}/tools_schema` (**root-only**, publish the type's tool catalog).

**Routing is server-side data, not connector logic.** A connection is a `(connector, external_account_id)` pair, account-scoped, neither segment containing `/`. The three-tier resolver (`src/aios_connectors/resolver.py`) maps an inbound `chat_id` to a session: Tier 1 `chat_sessions` ledger → Tier 2 `routing_rules` prefix demux → Tier 3 `bindings.mode` fallback (`single_session` → `binding.session_id`; `per_chat` → spawn from `binding.session_template_id`; no active binding → `DETACHED` → HTTP 422). The focal-channel address scheme is `{connector}/{external_account_id}/{chat_id}`. Dedup is at-most-once-append via `try_record_inbound_ack` (PK `(account_id, connector, external_account_id, event_id)`) in the same transaction as `append_event`. **422 (`DETACHED`/`ARCHIVED_TEMPLATE`) is terminal — the connector acks the platform event and stops; only 5xx is retryable** (the #523/#526/#541 retry-storm contract).

A Slack connector reuses all of this verbatim. It writes only: the `connector='slack'` attr, `serve_connection` (Socket Mode client + inbound mapping → `emit_inbound`), the `@tool` outbound methods, the parse/gate functions, and a Dockerfile/pyproject mirroring telegram.

---

## 3. Design

### 3.1 Package structure (mirrors telegram)

`connectors/slack/` src-layout package, `src/aios_slack/`, hatchling, `python>=3.13`. Dependencies: `aios-connector-http` + `structlog` + **`slack-sdk`** (not `slack-bolt` — we need only `AsyncWebClient` + `AsyncSocketModeClient`; Bolt's HTTP receiver/ack machinery is dead weight for socket-only). Ruff `line-length=100`, select `(E,W,F,I,B,UP,SIM,RUF,TID,ASYNC)`, mypy strict, pytest `asyncio_mode=auto`. Dockerfile is a near-copy of telegram's: `COPY` aios-sdk + aios-connector-http + slack, `uv pip install --system --no-sources` by path, `CMD python -m aios_slack`.

Files:
- `connector.py` — `SlackConnector(HttpConnector)` lifecycle + `@tool` methods.
- `parse.py` — Slack event → frozen `InboundMessage` dataclass; the four connector-side gates; `chat_id`/`event_id` derivation.
- `format.py` — markdown → Slack `mrkdwn` IR pipeline + hard clamps.
- `prompts.py` — `build_instructions` identity prelude + `SLACK_SERVER_INSTRUCTIONS`.
- `__main__.py` — `asyncio.run(SlackConnector().run_until_stopped())`, with `SqliteAnsweredSpool` wired in 3 lines.

### 3.2 Transport: Socket Mode

Socket Mode is the v0 transport, chosen deliberately over the Events API:

- **No public ingress, no signing secret, survives behind NAT.** The only inbound channel is the authenticated WebSocket the sidecar dialed; the `xapp-` app token is the auth boundary. This eliminates the entire HMAC-signature / replay / `url_verification`-challenge attack surface that an HTTP receiver carries.
- **The 3-second ack deadline is NOT eliminated — it is the connector's responsibility.** *(Red-team, correctness, sev 82.)* The async `AsyncSocketModeClient` does **not** auto-acknowledge. The listener **must** `await socket_client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))` **immediately at the top of the listener, before any parsing or enqueueing** — ack-then-process. If no ack reaches Slack within ~3s, Slack redelivers (up to 3×) and, under sustained non-ack, throttles/closes the socket — a silently-dead bot indistinguishable from a healthy idle one. The earlier "Socket Mode eliminates the 3s-ack surface" framing was false and is corrected: Socket Mode removes only the signature/replay surface.

The deferred HTTP/Events-API arm is **gated** on shipping: constant-time HMAC-SHA256 over `v0:{ts}:{rawbody}` with a ~5-min replay window, `url_verification` challenge echo, and ack-then-process. It is specified as the bar that arm must pass, not built speculatively.

**Inbound at-most-once loss is a real Socket Mode limitation, named not papered over.** *(Red-team, operability, sev 68.)* Socket Mode does **not** buffer-and-redeliver events that fire while the socket is down (unlike the Events API's HTTP retry, and unlike Telegram long-polling's offset cursor which replays missed updates — the analog this skeleton otherwise mirrors). On a WebSocket drop (Slack rotates the socket periodically by design; plus network blips), any message posted in the gap is gone, with no cursor to replay from. The dedup ledger defends against *duplicate* delivery, not *missed* delivery — it is the wrong defense for this failure. v0 stance:

1. **Document the loss window** as a third genuine "where AIOS can't" line and emit a `lifecycle`/structured-log signal on every disconnect so the gap is observable.
2. **On every socket (re)connect, run a bounded `conversations.history` backfill** for channels with recent bot activity (Slack's own recommended reconnect pattern), watermarked by last-seen `ts` and keyed through the existing `event_id` dedup ledger so re-fetched-but-already-seen events are no-ops. This is the inbound analog of the missing-`thread_ts` backfill (§3.4) and reuses the same idempotency machinery. **Specified for v0, not deferred.**

### 3.3 `serve_connection` lifecycle

`serve_connection(connection_id, secrets)`:
1. Build `AsyncWebClient(token=secrets['bot_token'])`. **Append `AsyncRateLimitErrorRetryHandler(max_retry_count>=2)` to `client.retry_handlers`** *(red-team, correctness, sev 48)* — the bare client retries only `ConnectionError`, never HTTP 429. `chat.postMessage` is ~1 msg/s/channel and `reactions.add` is Tier 3; without the handler a 429 surfaces as an opaque `is_error` tool result, the model retries Retry-After-blind, and the throttle compounds across wasted background turns. The handler honors `Retry-After` transparently inside the tool call. Single-send stays at-most-once-safe (429 = not sent).
2. Build `AsyncSocketModeClient(app_token=secrets['app_token'], web_client=...)`.
3. `auth.test` → cache `bot_user_id` + `team_id`. **Fail-closed identity gate: if `auth.test`'s `team_id != connection.external_account_id`, refuse to serve** *(red-team, operability, sev 38)* — `emit_lifecycle('slack.install.identity_mismatch')`, log `slack_install_identity_mismatch`, and let the task die rather than act under a mislabeled identity. The `team_id` is already in hand; this costs nothing and converts a wrong-token paste (most likely and least detectable exactly at manual install) from a silent split-brain into a loud refusal. This is the WhatsApp `confirmPairing` fail-closed gate, adopted in v0 rather than deferred to the OAuth arm.
4. Register a socket-event listener that **acks first** (§3.2), then parses and pushes onto a per-connection `inbound_queue`.
5. Race `socket_client.connect()` + a `_drain_queue → emit_inbound` task in a `TaskGroup`. The backfill (§3.2) and `thread_ts` resolution (§3.4) run **off the ack path, in the drain task**, never in the listener.
6. `finally`: close the socket client (`teardown` is a no-op, exactly like telegram).

`_SlackConnectionState` dataclass: `{web_client, socket_client, bot_user_id, team_id, inbound_queue}`.

**Dead-worker recovery is a named obligation, not "the SDK handles it."** *(Red-team, operability, sev 76.)* The SDK's `_isolated_serve_connection` isolates a *failure* (siblings survive) but provides **no recovery path**: its `finally` pops `self.state[connection_id]` but not `self._connections[connection_id]`, and `_on_connection_added` short-circuits when the id is already in `_connections`. So after a non-recoverable auth error the connection is a permanent zombie — every discovery-backfill `added` is ignored as "already running," a later secret correction is never picked up (secrets are read only at spawn), and the workspace stays silently dead while process-level health shows green. v0 resolution: **on a non-recoverable auth error (`invalid_auth|account_inactive|token_revoked`), `serve_connection` does NOT die — it `emit_lifecycle('slack.token.revoked')` and BLOCKS in a bounded-backoff loop that re-fetches `/runtime/secrets` and retries the build.** A corrected token is then picked up without a container restart and without a `removed`/`added` cycle. (This is also the cleanest path for routine token rotation; see §8 for the residual multi-workspace blast-radius note.) The SDK-level fix — have `_isolated_serve_connection` also pop `_connections` so a backfill re-spawns the worker, paired with backoff — is filed as a separate SDK issue; the Slack connector does not ship on the broken assumption.

### 3.4 `chat_id` / `focal_channel` scheme: threads share the channel session

`focal_channel = slack/{external_account_id}/{chat_id}` via `self.focal_channel(team_id, chat_id)`.

- **`external_account_id` = the Slack `team_id` (`T0…`)** — the durable workspace install identity (survives bot reinstall, which can mint a new `bot_user_id`), slash-free, the unit of credential isolation + tenant scoping, and the `## Connector: slack/<team_id>` MCP heading. `bot_user_id` is cached for self-filtering + mention detection but is **not** the address segment.

**Thread mapping — the central fork, resolved toward threads-SHARE-the-channel-session.** *(Red-team: architecture-fit sev 62, product-scope sev 58, plus the operator-route/double-encoding cluster.)* The v0 design **does not** fork a session per thread. Instead:

- `chat_id` = the bare Slack conversation id (`C…`/`G…`/`D…`) for **both** top-level and thread messages. A top-level channel message keys on the bare channel id (the openclaw #10686 load-bearing rule: a busy channel is **one** persistent session).
- `thread_ts` rides `connector_metadata` under a **non-reserved** key, and is the model-supplied `slack_send(thread_ts=…)` param read off the inbound metadata header — exactly the Telegram `reply_to_message_id` shape, the only verified-in-tree threading precedent. The model threads its outbound replies; context accumulates in one per-channel session.

This is the **single source of thread truth** and dissolves an entire cluster of confirmed findings at once:

- It eliminates the `channel:thread_ts` **double-encoding** (the SDK-injected `chat_id` suffix *and* a model `thread_ts` param with no specified precedence — a flag-not-kind ambiguity). With bare `chat_id`, `thread_ts` is the lone authority; `slack_send`/`slack_react` pass `chat_id` directly to the Slack API exactly as telegram does, with **no `:`-splitting logic** to get wrong.
- It eliminates the **per-thread session/workspace/ledger sprawl** — session count is bounded by *channels the bot is in* (operator-controlled), not *threads ever opened* (user-controlled, unbounded, with no `chat_sessions` GC and no per-account session cap). The `archive_when_idle`-vs-multi-turn-thread dilemma (self-archive after one turn → 422 on follow-up, OR no archive → permanent per-thread sprawl) does not arise.
- It removes the connector-side `chat_id`→`(channel, thread_ts)` decomposition the API would otherwise require, and the `slack_react` target-`ts`-vs-thread-`ts` ambiguity.

The opposite encoding (`chat_id = channel:thread_ts`, threads-as-sessions) is **verified parse-safe** at the core/SQL boundary (`split_part(channel,'/',3)` and the SDK's `focal_channel.split('/',2)` both treat the `:`-suffixed segment as opaque) and is **strictly more capable**, but it is **not free**: it costs a full session per thread, forks the connector's outbound contract, and depends on `chat_sessions` GC + per-template idle-TTL that AIOS does not have. It is therefore reserved as an **explicit per-connection opt-in** (a `routing_rules` prefix or a binding knob) for when a bounded use case and the supporting GC exist. **This is an open product fork flagged for the human (§8).**

`event_id` (connector-owned, deterministic, survives Socket-Mode redelivery + restart):
- messages: `slack-{channel}-{ts}`
- edits (`message_changed`): `slack-{channel}-{ts}-e{edit_ts}` (so an edit does not dedup-collide with the original — the telegram message-id-reuse trap)

These ride the existing `connector_inbound_acks` PK unchanged, giving durable at-most-once-append in the same txn as `append_event`, neutralizing both Slack's at-least-once redelivery and the `message`/`app_mention` double-fire (app_mention is treated **only** as a `wasMentioned=true` hint, never a second emitted delivery — both events share the same `ts`, so even if both reached `emit_inbound` with the canonical id, the ledger collapses them).

**Missing-`thread_ts` backfill, hardened.** *(Red-team, correctness, sev 42/32.)* Slack sometimes delivers a thread reply with `parent_user_id` set but `thread_ts` missing. The backfill via `conversations.history` (latest=oldest=ts, limit 1, 60s cache + in-flight dedup) is confined to **exactly that rare shape**, runs **in the drain task off the ack path** (never blocking the 3s ack), uses the 429-retry-handler client, and on unrecoverable failure (429/timeout) **falls back to channel-scoped routing with a logged signal — never errors and never blocks** (a drop or a logged channel-scope fallback is observable; a silent mis-route is not). Because threads share the channel session in v0, this backfill is now only a metadata-enrichment step, not a routing-critical one — the failure mode is benign by construction.

### 3.5 Outbound `@tool` vocabulary

Two tools in the v0 floor; rich Google-style docstrings; the SDK derives the JSON schema.

**`slack_send(text, thread_ts: str|None=None, *, connection_id, chat_id) -> {ts, channel}`** — MVP, mandatory. `chat.postMessage` with `mrkdwn`. Bare assistant text is invisible to Slack (same contract as `telegram_send`); without this the bot cannot reply at all. `thread_ts` (read by the model off the inbound metadata header) threads the reply; default `None` posts top-level. `channel` return = `self.focal_channel(team_id, chat_id)` so observers read the send target off the `tool_result`. Text-only in v0; no `blocks`/`attachments` param (deferred). `connection_id` is server-authoritative (stamped on the calls-SSE payload from the session→connection binding, stripped from the model-facing schema) — the model cannot select a workspace.

**`slack_react(message_ts, emoji: str|None, *, connection_id, chat_id) -> {status}`** — MVP. `reactions.add` when `emoji` set (colon-stripped, normalized), `reactions.remove` when `None`. Exactly one Web API call, the cheapest acknowledgement affordance ("react :eyes: to show I'm working"). With bare `chat_id`, this calls `reactions.add(channel=chat_id, timestamp=message_ts)` directly — no decomposition. Mirrors `telegram_react`.

Each tool justified: `slack_send` is the only way the model is heard; `slack_react` is near-zero-surface acknowledgement. Both are the floor; everything else (edit/delete/upload/blocks/streaming) is deferred (§7).

### 3.6 Inbound normalization + mention-gating + bot-loop prevention

`parse.py` produces a frozen `InboundMessage(slots=True, frozen=True)`: `{chat_kind(im|mpim|channel|group), channel_id, sender_id, sender_name, message_ts, thread_ts|None, text, edited, edit_ts|None, mentions}`, and the four connector-side gates as pure functions, run **before** `emit_inbound`:

**(1) Self / bot-loop filter — handling `message_changed` correctly.** *(Red-team, two findings, sev 70 + 68.)* For a plain message, drop `event.user == bot_user_id` (self, the primary loop guard) and drop `event.bot_id`-bearing (`allowBots=false` default). **For `message_changed`, the author identity is NESTED under `event.message` — there is no top-level `user`/`bot_id`.** A top-level read fails open for every edit. The gate therefore, for `subtype == message_changed`, reads `event.message.user` / `event.message.bot_id` / `event.message.edited.user` and drops on `event.message.user == bot_user_id` OR `event.message.bot_id` present, exactly as for a plain message. The cleaner correct-by-construction form — **mirror openclaw and route `message_changed` to a non-emitting system-event path rather than into `emit_inbound` at all** — is the chosen v0 implementation; either way a bot-authored edit is dropped before it can wake the session. A contract test feeds a synthetic bot-authored `message_changed` and asserts it is dropped. The `event_id` for an edit is built from `event.message.ts` + `event.message.edited.ts`, not the top-level event `ts`.

**(2) Cross-app / cross-team filter.** Drop events whose `api_app_id` or `team_id` mismatch the connection's own app/workspace (cheap, since one runtime token serves all the tenant's Slack connections; protects a shared process from a mis-delivered event).

**(3) Subtype filter.** Drop subtypes other than the plain user message (and the `message_changed` system-event diversion above).

**(4) Mention-gate — mandatory, with thread-continuation and the right mpim default.** *(Red-team, three findings: sev 82, 48, and a framing correction.)* AIOS has **no** inbound mention gate; this is one of the two honest lines of connector discipline in the spine (the other is bot-loop prevention). It is a pure decision function over the typed `chat_kind` discriminant — encoded as a kind, not booleans, so illegal combinations are unrepresentable. The policy:

- `im` (1:1 DM, single human) → **always respond**.
- `mpim` / `channel` / `group` (any room with >1 human) → **require an explicit `<@bot_user_id>` mention** — **with implicit-mention bypasses** (below). *(mpim defaults to the channel rule, not the DM rule: an mpim is a small group of humans who added the bot; always-responding makes it spam every human-to-human message. The "it was added so it should respond freely" intuition is wrong for any room with >1 human.)*

**Implicit-mention bypass (the thread-continuation exception) is in v0, not deferred** *(sev 82)*: a human who @-mentions the bot to start a channel thread does **not** re-mention it on every follow-up. Without a bypass the bot answers exactly the first message then ghosts. The robust signal is **`bot_thread_participant`: a channel thread reply bypasses the mention requirement when its thread already has bot activity** — implemented as "the inbound's `thread_ts` matches a thread the bot has posted in," which the connector knows from its own sent-`ts` set per thread (or via `parent_user_id == bot_user_id` for the direct reply-to-bot case). This ties cleanly to the channel-session model: the gate consults per-thread bot participation, not a per-thread session. If gated out, the message is recorded into per-channel pending-history (for later context) and dropped — fail-quiet.

**Sender attribution.** The connector supplies `sender.display_name` (the human Slack display name) and `sender.id` (the opaque `U…` id). It **sanitizes `display_name` in parse.py before `emit_inbound` — strip newlines, length-cap** — so it cannot carry multi-line injected content into the rendered `from=` clause. (This is cheap hygiene, not a foreclosure; see §4 for the precise trust framing.) Slack-specific extras (`team_id`, `thread_ts`, `app_id`, `chat_kind`, `message_ts`) ride `connector_metadata` under **non-reserved** keys only.

### 3.7 Install flow and secrets

**No OAuth in v0. Manual manifest-paste install**, mirroring openclaw's deliberate choice — AIOS never holds a `client_id`/`client_secret`, never receives refresh tokens, runs no public redirect endpoint.

The connector ships a printed Slack **app manifest** with `socket_mode_enabled: true` and a **least-privilege, feature-gated** bot-scope set. v0 scopes, each justified by a shipped behavior:
- `chat:write` (slack_send), `reactions:write` (slack_react), `app_mentions:read` (mention-gate), `users:read` (sender display-name resolution).
- `channels:history`, `groups:history`, `im:history`, `mpim:history` — **gated on the `conversations.history` backfill** (reconnect catch-up §3.2 + missing-`thread_ts` §3.4). These back real v0 behaviors and are therefore in.
- **`reactions:read` is REMOVED from v0** *(red-team, product-scope, sev 48)*: v0 ingests no inbound reaction events (there is no inbound-reaction handler and no use case), and ingesting them ungated would wake the model on every `:thumbsup:` in any joined channel — the exact spam the mention-gate exists to prevent (reactions carry no text to mention in). The `slack-react-…` inbound `event_id` arm and the inbound-reaction path are removed until an inbound-reaction feature is actually designed. `reactions:write` stays (outbound `slack_react`). `files:*` / `pins:* `/ `commands` are dropped until those tools land.

**Secrets stored, and where.** The operator creates the app from the manifest, installs it to their workspace via Slack's UI, and stores the resulting long-lived static tokens as **encrypted connection secrets** (Slack has no daemon/`store.db`, so the connection-secret store is the only credential home):

```
aios connections create --connector slack --external-account-id <team_id> \
  --secret bot_token=xoxb-... --secret app_token=xapp-...
```

These are encrypted at rest under `AIOS_VAULT_KEY`, keyed to the owning `account_id` via `derive_account_subkey` (another tenant's derived key cannot decrypt even if the row leaks). The **only** decrypt path is the runtime-scoped `GET /v1/connectors/runtime/secrets`, reachable only by a connector container holding a `slack` runtime token for that account. Archive scrubs `secrets_ciphertext` + `secrets_nonce` to NULL. Secrets are cached at `serve_connection` spawn; rotation is picked up by the dead-worker re-fetch loop (§3.3) without a full restart.

### 3.8 Session-binding defaults

The connector stores **no** routing state. The operator binds a mode after connection-create:
- `single_session` (whole workspace → one session, the personal-assistant shape): `POST .../attach`.
- `per_chat` (each conversation → its own session from a template, the assistant-in-many-conversations shape): `POST .../configure-per-chat --template <session_template>`.

**v0 walkthrough leads with `single_session` for a single personal workspace** (simplest, bounded, no sprawl considerations). The `per_chat` shape is documented for the multi-conversation case; `routing_rules` prefix demux (e.g. `D` DMs vs `C` channels) is available as a free knob. Default binding mode is an explicit product fork flagged for the human (§8).

---

## 4. Security model (invariants)

The spine: **most threats are foreclosed by an existing invariant the connector structurally cannot bypass; exactly two are connector discipline.** Stated as invariants:

**INV-1 (Tenant isolation, structural).** The runtime bearer resolves to one `account_id` that scopes every connection / secret / pending-call / inbound-ack query; `list_pending_calls_for_connector` and the discovery tail drop cross-tenant rows. `_check_runtime_scope` locks the bearer to `connector='slack'`. A `slack` token for tenant A can never see tenant B's data nor touch a telegram connection. Tenant isolation does **not** depend on connector behavior.

**INV-2 (Connection allowlist, structural).** Issuing the token with `connection_ids` restricted scopes a container to a subset of workspaces; per-connection routes (inbound/secrets/tool-results/lifecycle) enforce it; type-wide routes (tools_schema PUT, management SSE) deliberately don't; SSE streams filter silently.

**INV-3 (`_RESERVED_METADATA_KEYS` boundary, structural).** `{channel, sender_name, attachments, platform_timestamp}` are **stripped from `connector_metadata` before merge**; the trusted server path then writes them. This forecloses the **metadata channel** for two specific forgeries: (a) the connector cannot plant a `metadata.attachments` record with `in_sandbox_path='/workspace/…'` that the vision renderer would `read_bytes` + inline as a base64 `image_url` part — the **/workspace exfil chain**; attachments (deferred) may therefore be delivered **only** as inline multipart `InboundAttachment` bytes confined by staging to read-only `/mnt/attachments`, never a sandbox path. (b) the connector cannot set `metadata.sender_name` to a value *different* from the `display_name` it observed — the server owns that slot.

  **Precise framing of sender attribution** *(red-team, security, sev 20 — note-as-residual, not a foreclosure).* INV-3 forecloses the *metadata channel*, not the *content* of `display_name`. The server writes `metadata.sender_name = sender.display_name` verbatim, and the harness renders it as `from=<sender_name>`. A Slack display name is attacker-controlled and reassignable. So `from=` is **descriptive context for the model, not an authenticated identity** — it is governed by AIOS's untrusted-inbound posture (the same as message body, `chat_name`, quoted text), and the connector's parse-step newline-strip + length-cap (§3.6) is hygiene that keeps it single-line. **Authorization never keys on `display_name`** — it keys on the opaque `U…`/connection/tenant layer. This is identical to the already-shipped telegram/signal/whatsapp behavior; we do not claim INV-3 authenticates the sender's name.

**INV-4 (Capability / no ambient authority).** The `slack` tools enter a session's model prelude **only** via `list_connection_tools_for_session`, which JOINs a bound `slack` connection to the **root-published** `connectors.tools_schema` row — so the tools are visible only to sessions explicitly bound to a `slack` connection, scoped to the owning `account_id`. The connector itself holds one connector-type runtime bearer (authority = act as `connector='slack'` for one `account_id`, optionally narrowed by INV-2) plus per-connection bot/app tokens fetched at spawn — both attenuations of the operator's grant, neither ambient. **The connector introduces no ambient authority.**

  **Correction on the #794 clamp claim** *(red-team, security, sev 42 — fold-into-design).* A **`per_chat`-spawned Slack session does NOT receive the #794 attenuation-lattice meet.** The clamp (`attenuation_service.clamp(surface_of(agent), surface_of(run))`) is wired **only** into the workflow spawn path (`create_child_session`, reached via `workflows/step.py`). The connector resolver's `_spawn_per_chat_session` calls `create_session`, which takes no `surface` argument and applies no clamp; the spawned session reads its surface live from the bound template's agent (`parent_run_id is None`), inheriting it **verbatim**. The earlier claim that "a Slack-bound child's authority can never exceed its launcher's snapshotted surface" is **false on this path** and is struck. The actual, honest security boundary for a `per_chat` Slack session is **(a) which template the operator binds** and **(b) INV-1/INV-2 tenant + connection scoping** — *not* the attenuation lattice. The outcome is safe (the session gets exactly the operator-configured surface), but the mechanism is operator-binding + scoping, not the clamp. If meet-bounded authority for connector-spawned sessions is later wanted, that is a core change (thread a clamp through `create_session` for the resolver) and must be filed as its own issue.

**INV-5 (Install identity, fail-closed).** `serve_connection` refuses to serve if `auth.test`'s `team_id != connection.external_account_id` (§3.3) — closing the wrong-token-paste split-brain.

**INV-6 (Signature verification — Socket Mode).** In Socket Mode the `xapp-` token IS the inbound auth boundary; there is no HMAC surface. The deferred HTTP arm is gated on constant-time HMAC-SHA256 over `v0:{ts}:{rawbody}` + ~5-min replay window + `url_verification` echo before it may ship.

**The two honest lines of connector discipline** (no core backstop — a bug here is not caught by any invariant):
- **Bot-loop prevention** (INV-disc-A): drop `user == bot_user_id`, drop `bot_id`-bearing (`allowBots=false`), `message_changed` nested-identity read (§3.6), cross-app/team drop.
- **Mention-gating** (INV-disc-B): §3.6. This is the prime candidate to **promote into core** (a typed inbound `ChatType` + a `resolveInboundMentionDecision` pure function) the moment a **second** channel connector needs the same gate — written now in the openclaw facts×policy shape so promotion is a lift-not-rewrite. For one connector it stays connector-side per "unify on the second instance."

**Retry/drop posture (outbound + inbound).** Inbound: AIOS 422 (`DETACHED`/`ARCHIVED_TEMPLATE`) is terminal (ack the Slack event, stop); 5xx is retryable. Outbound: `slack_send` is **at-least-once** — a `tool-result` POST that fails *after* a successful `chat.postMessage` re-dispatches the call on reconnect and posts a second Slack message *(red-team, correctness, sev 52)*. The `SqliteAnsweredSpool` covers restart-replay of **completed** calls (persisted only after a successful dispatch); it does **not** cover the send-succeeded/result-POST-failed window. v0 documents `slack_send` as at-least-once (rare duplicate under AIOS-side outage); a clean idempotency-key fix (deterministic `client_msg_id` derived from `tool_call_id`, or persist sent-`ts` before the result POST) is a worthwhile SDK-level follow-up that benefits telegram too — filed separately, not forked into v0.

**always_ask over Slack** (deferred with Block Kit): render the pending tool as Approve/Deny blocks, persist `{channel, ts}`, on the inbound button decision flip the **same** message in place via `chat.update` and POST `/sessions/:id/tool-confirmations`; approver authorized **strictly by opaque Slack user-id membership**, never display name.

---

## 5. AIOS-side changes

**Core code: NONE.** No change to the runtime contract, the three-tier resolver, the inbound pipeline, the secret store, the management-call plane, or the SSE streams. A new connector type `slack` is pure **data**: `insert_connection` upserts the `connectors` catalog row on first connection-create, and the SDK PUTs the `slack` `tools_schema` at startup. No migration, no route, no service code.

**One hard operational precondition (root-only `tools_schema`).** *(Red-team, product-scope, sev 28 — fold-into-design.)* `connectors.tools_schema` is **one global per-type row shared across all tenants**, so publication is **root-only** (`services/connectors.py` rejects `parent_account_id is not None`) — a child publishing it is a cross-tenant prompt-injection vector. This is an operational step, not a code change, but it must be a **first-class ordered step** in the walkthrough: **"Step 0 (root operator, once per constellation): publish the `slack` `tools_schema`"** (run the connector once under a root runtime token, or have the root operator do the first PUT). The SDK's startup PUT **fails hard on a non-root 403 by design** (no silent skip — that would mask a token-account misconfiguration, violating AIOS's fail-hard stance); the crash IS the signal. We surface the prerequisite loudly in the walkthrough so the most likely first-run confusion (wrong-account token) is pre-empted, rather than burying it as an aside.

**No management endpoints in v0.** The Signal/WhatsApp operator-RPC trio (typed Pydantic bodies + a `_slack_management_call` helper + `@router.post` per method) is purely-additive and required **only** if/when a hosted 3-legged OAuth install arm is added. `submit_call` is already connector-agnostic; a future Slack OAuth flow reuses it verbatim and adds only the thin operator-facing veneer (the `captcha_required`-as-200 / `confirm-pairing`-long-poll shapes are the template, and the fail-closed `team_id` gate is adopted there too). **If that arm is ever built, the recurrence of the near-identical veneer across signal/whatsapp/slack is the signal to generalize the operator-management route into a small declarative table (method → params model → response model → error mapping) — not a third hand-written trio.** *(Red-team, architecture-fit, sev 12 — noted as a future generalization, not pre-committed and not built now; two instances exist, the third is hypothetical.)*

**No mention-gate or `ChatType` added to core in v0** — connector-side per §4 (INV-disc-B). Promotion to a shared core primitive is the explicit trigger-on-second-connector.

**No workspace bind-mount in v0** (no `slack_upload` SandboxPath arg yet). When file upload lands (deferred), the container must bind-mount the host `workspace_root` read-only at the same in-container path — a deployment coupling, still not a core code change.

Is a generic management/OAuth mechanism needed? **No, not for v0**, and when it is, it is a **generalization** of the existing `submit_call` plane (already connector-agnostic), not an accretion — the only new code is the thin per-connector veneer, which itself should generalize at the third instance.

---

## 6. What the adversarial review changed

Top confirmed findings, by refined severity, with what each changed:

| # | Lens | Sev | What it changed |
|---|---|---|---|
| Socket-Mode 3s ack | correctness | 82 | **Ack-first in the listener** (`send_socket_mode_response` before parse); struck the false "Socket Mode eliminates the 3s-ack surface" claim. |
| Dead-worker zombie | operability | 76 | `serve_connection` **blocks-and-retries** on non-recoverable auth (re-fetch secrets) instead of dying; named recovery as an obligation, not "the SDK handles it." Filed SDK fix separately. |
| Mention-gate thread-continuation | product-scope | 82 | Added the **`bot_thread_participant` implicit-mention bypass** to v0 — the gate no longer makes the bot ghost after the first reply. |
| `message_changed` self-filter | security/correctness | 70/68 | Self/bot filter reads **nested `event.message.*`** for edits (or diverts `message_changed` to a non-emitting system path); edit `event_id` from `message.ts`+`edited.ts`. Closes the self-reply loop. |
| Socket-Mode event loss | operability | 68 | Named at-most-once inbound loss as a third "where AIOS can't"; **on-reconnect `conversations.history` backfill** keyed through the dedup ledger. |
| `channel:thread_ts` double-encoding + sprawl | arch-fit / product | 62 / 58 | **Threads share the channel session** (bare `chat_id` + `thread_ts` as model param/metadata); dissolved the double-encoding, the per-thread sprawl, and the `slack_react` decomposition gap. Threads-as-sessions reserved as explicit opt-in. |
| Outbound double-send | correctness | 52 | Documented `slack_send` as **at-least-once**; scoped the spool claim honestly; filed idempotency-key fix as an SDK follow-up. |
| 429 / Retry-After | correctness | 48 | **`AsyncRateLimitErrorRetryHandler`** wired at client construction. |
| `reactions:read` ungated inbound | product-scope | 48 | **Removed `reactions:read`** from the v0 manifest and the inbound-reaction path; kept `reactions:write` for outbound. |
| #794 clamp claim wrong path | security | 42 | **Struck the clamp justification** for `per_chat` Slack sessions; restated the real boundary as operator-binding + tenant/connection scoping. |
| Missing-`thread_ts` backfill | correctness | 42/32 | Confined to the rare shape, **off the ack path**, with a logged channel-scope fallback on failure — never blocks, never silently mis-routes. |
| Install identity gate | operability | 38 | **Fail-closed `team_id` gate** adopted in v0 `serve_connection`. |
| `slack_react` channel recovery | arch-fit | 45/38 | Dissolved by the bare-`chat_id` decision (no decomposition needed). |
| Root `tools_schema` precondition | product-scope | 28 | Promoted to **"Step 0" ordered walkthrough step**; SDK stays fail-hard on non-root 403. |

Dismissed after verification (one-line reasons): **sender `display_name` forgery (sev 20)** — INV-3 was never claimed to authenticate `display_name` content; `from=` is untrusted context and authz keys on opaque id; addressed by hygiene + framing, not a fix. **Bot-token egress/SSRF (sev 22)** — targets the deferred download arm; the *inbound* private-URL fetch obligation is folded into the deferred-arm spec (§7) but is not a v0 defect. **Single-token cross-workspace outbound mis-dispatch (sev 15)** — `connection_id` is server-authoritative on the calls-SSE payload and stripped from the model schema; mis-dispatch cannot occur. **`bot_user_id` drift (sev 10)** — reinstall preserves `bot_user_id`; identity-changing uninstall revokes the token, which the auth fast-fail already handles. **`channel:thread_ts` vs 4-segment convention (sev 12)** — `focal_channel_path`'s docstring is segment-agnostic, not a 4-segment mandate; moot under the bare-`chat_id` decision. **`app_mention` double-fire double-append (sev 12)** — the design already treats app_mention as a hint, and the shared-`ts` `event_id` makes the ledger collapse the pair.

---

## 7. MVP vs deferred

**MVP (v0):**
- `connectors/slack/` src-layout package mirroring telegram; `slack-sdk` (not bolt).
- `SlackConnector(HttpConnector)`, `connector='slack'`, `_SlackConnectionState`.
- `serve_connection`: build clients (with `AsyncRateLimitErrorRetryHandler`), `auth.test` + fail-closed `team_id` gate, ack-first socket listener, drain→`emit_inbound`, on-reconnect `conversations.history` backfill, dead-worker block-and-retry on auth failure.
- Inbound: self/bot/`message_changed`-nested/subtype filter → cross-app/team filter → mention-gate (im→always; mpim/channel/group→require mention, with `bot_thread_participant` bypass) → `sender={id, display_name(sanitized)}` + non-reserved `connector_metadata` → bare `chat_id` + `thread_ts` in metadata → `event_id` (edit-suffixed) → `emit_inbound`.
- Two `@tool`s: `slack_send` (text + optional `thread_ts`), `slack_react`.
- `format.py` markdown→mrkdwn + 8000-char/≤50-block/section-3000/label-75 clamps before every Web API call.
- `prompts.py` identity prelude + `SLACK_SERVER_INSTRUCTIONS` via MCP-init (not the tools_schema PUT).
- `SqliteAnsweredSpool` wired; `emit_lifecycle` on socket death / token revoke / identity mismatch.
- Least-privilege manifest (no `reactions:read`, no `files:*`/`pins:*`/`commands`).
- Operator walkthrough: **Step 0 root `tools_schema` publish** → manifest → install → paste tokens as secrets → `runtime-tokens issue --connector slack` → `attach` (lead) or `configure-per-chat` → `docker run` with `AIOS_URL`+`AIOS_RUNTIME_TOKEN` → DM or @mention.
- Parametrized inbound contract test (validated sender, renderable body, group label for non-DM, gate outcomes incl. bot-authored `message_changed` dropped and thread-continuation bypass) + parse/format/prompts unit tests, all no-network.

**Deferred (with named obligations):**
- `slack_edit_message` (`chat.update`) / `slack_delete_message` (`chat.delete`) — trivial; add once the floor is smoked.
- `slack_upload_file` + inbound attachments — the 3-step `files.getUploadURLExternal`→presigned PUT→`completeUploadExternal`, U-id→DM `conversations.open` resolution, 5 MiB cap, and the workspace bind-mount. **Obligation (red-team, arch-fit sev 40):** the **inbound** Slack private-URL download must use the **same `*.slack.com`/`*.slack-files.com`/`*.slack-edge.com` allowlist + no-cross-host-redirect pinning** as the outbound PUT, and the bot token must be attached **only** to allowlisted hosts (the URL is Slack-API-returned, but the guard is named on both directions).
- Block Kit (`blocks` escape hatch + `[[slack_buttons]]`/`[[slack_select]]` directives + `block_actions` round-trip via synthetic inbound).
- always_ask → Slack approval buttons (depends on Block Kit; opaque-id approver authz).
- Native streaming (`chat.startStream/appendStream/stopStream`) — genuine "where AIOS can't" #1.
- Modals (`views.open` / `view_submission`) — genuine "where AIOS can't" #2.
- HTTP/Events-API transport — gated on HMAC + replay window + `url_verification` echo + ack-then-process.
- Hosted 3-legged OAuth install + management-call trio + fail-closed `team_id` gate (generalize the veneer at the third instance).
- Per-sender allowlists, `dmPolicy`/`groupPolicy`, channel-config security-audit findings — connection-level auth + mention-gate suffices for v0.
- Slash commands — privilege-escalation surface; out until command authz is designed.
- **Threads-as-sessions** (`chat_id=channel:thread_ts`) as an explicit per-connection opt-in — requires `chat_sessions` GC + per-template idle-TTL first.
- Mention-gating as a shared core primitive (`ChatType` + `resolveInboundMentionDecision`) — on the second channel connector.

---

## 8. Residual risks & open design forks for the human

**Residual risks (accepted for v0, documented):**

- **`from=<sender_name>` is untrusted context, not authenticated identity** *(sev 20).* A Slack display name is reassignable; the model sees it as descriptive. Authorization keys only on opaque ids. Mitigated by parse-step newline-strip + length-cap; not a foreclosure. Identical to shipped telegram/signal/whatsapp.
- **`slack_send` is at-least-once** *(sev 52).* A rare AIOS-side outage between Slack-send-success and result-POST yields one duplicate channel message. The clean idempotency-key fix is an SDK follow-up (benefits telegram).
- **Inbound at-most-once loss on socket gaps** *(sev 68).* The reconnect `conversations.history` backfill heals most of it; a message posted exactly in a sub-second gap on a channel with no recent activity watermark can still be missed. Observable via disconnect lifecycle signal.
- **No `per_chat` capability clamp** *(sev 42).* A `per_chat` Slack session inherits the bound template agent's surface **unattenuated** (the #794 clamp is workflow-only). Safe because the operator chooses the template and INV-1/INV-2 scope it; but operators must understand that binding a broad-surface template per_chat grants that full surface to anyone who can reach the bot.
- **Multi-workspace token rotation blast radius** *(sev 36/52).* The dead-worker block-and-retry loop (§3.3) lets a single workspace's corrected token be picked up without a full restart; a *deliberate* `docker restart` still bounces all N sockets. For high-workspace-count containers, prefer per-workspace `connection_ids`-scoped tokens (one container per workspace) — documented in the multi-workspace walkthrough.
- **Bot-token blast radius** *(sev 22).* The `xoxb-` token grants cross-channel `*:history` reads workspace-wide (inherent to Slack's token model; AIOS cannot confine it). Mitigated by issuing `connection_ids`-scoped tokens and least-privilege manifest scopes; the inbound-download SSRF obligation is named on the deferred attachment arm.
- **No operator-facing per-connection health surface** *(sev 34).* A silently-dead workspace shows green on process-level health. v0 mitigation is **structured-log discipline**: every disconnect/reconnect/`serve_failed`/identity-mismatch logs `connection_id` **and** `team_id` (the connector caches `team_id`, so this is free) for greppability. A first-class per-connection liveness read-model/CLI field is a **substrate-wide** observability gap (all connectors share it), filed as its own cross-connector issue, not solved in this single connector.

**Open design forks needing a human decision:**

1. **Default binding mode for the walkthrough.** `single_session` (whole workspace → one session, personal-assistant) vs `per_chat` (each conversation → its own session, assistant-in-many-conversations), or a split-by-`chat_kind` (DMs `single_session`, channels `per_chat` via `routing_rules`). The design supports all three with no code difference. v0 leads with `single_session`; confirm this is the product default.

2. **Thread granularity (the central fork).** v0 chooses **threads share the channel session** (bounded, no sprawl, single source of thread truth). The strictly-more-capable **threads-as-sessions** (`chat_id=channel:thread_ts`) is reserved as an explicit opt-in pending `chat_sessions` GC + per-template idle-TTL. Confirm the shared-channel default, and whether/when to invest in the GC that would make per-thread sessions safe to offer.

3. **`mpim` mention policy.** v0 defaults mpim to the **channel rule** (require mention) on the "any room with >1 human" principle. Confirm — an operator who wants a dedicated bot-and-me-plus-one group DM to respond freely would want an opt-in.

4. **`allowBots` opt-in.** v0 ships `allowBots=false` (no bot-to-bot, the safe loop guard). A legitimate bot-relay use case would need an opt-in. Confirm no immediate need.

5. **Inbound contract-test harness: shared now or inline-then-generalize.** Leaning inline-in-slack-tests and generalizing to a parametrized cross-connector suite on the second connector (per "don't pre-abstract"), but it is the single highest-value cross-connector test idea — flagging the timing for an explicit call.
