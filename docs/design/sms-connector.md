# AIOS SMS Connector — Final Design

*Status: implementation-ready · Steelmanned against an independently-verified adversarial review · 2026-06-16*

## 1. Summary

We are building **`aios-sms`**, a credential-isolated connector sidecar that gives an AIOS agent a phone number it can send and receive SMS/MMS on. It is built on the existing `aios-connector-http` SDK (`HttpConnector`, `connector="sms"`), is **Twilio-first but provider-seam-ready**, and adds **zero new aios-api mechanisms on the happy path** — it speaks the existing `/v1/connectors/runtime/*` contract verbatim, exactly as signal/whatsapp/telegram do.

The single most important design insight, and the one place SMS does not fit the connector mold cleanly, is the **direction of the three message paths**:

- **Outbound = a model tool call.** Bare assistant text is *never* delivered. The model emits an `sms_send` `type=custom` tool call → it sits pending in the session log → the `/runtime/calls` SSE backfills it to the sidecar → the sidecar runs the `@tool` and POSTs `/runtime/tool-results`. AIOS never pushes a send.
- **The connector is a credential-isolated sidecar.** Twilio creds (`account_sid`, an **API Key SID+secret** for sending, and the `auth_token` used only to verify inbound signatures) are per-connection secrets, encrypted at rest under the owning account's subkey, readable *only* via the runtime-scoped `GET /v1/connectors/runtime/secrets`. aios-api never sees a Twilio byte.
- **Inbound = a provider webhook — and this is the misfit.** The AIOS runtime contract is single-direction: the connector is always the HTTP *client*; there is no AIOS route a third party POSTs into. Twilio is push-only and POSTs `application/x-www-form-urlencoded` to a URL. **Resolution: the sidecar stands up its own HTTP listener** (one container-wide `aiohttp` server in `setup(tg)`), verifies `X-Twilio-Signature`, parses the Twilio payload, and *then* calls `emit_inbound` → `POST /v1/connectors/runtime/inbound`. The provider webhook lands **inside the sidecar**, not on aios-api. aios-api's trust boundary stays exactly at the runtime bearer token. This makes the SMS sidecar the **only inbound-reachable surface in the entire connector fleet** — a deliberate, contained change to the *fleet* threat model (not a change to aios-api), and the reason this document treats the sidecar's listener as a first-class hardened, monitored unit rather than an ops afterthought.

Identity is the signal/whatsapp E.164 pattern: `external_account_id` = the AIOS-owned Twilio number, `chat_id` = the peer number, `focal_channel = sms/<our_e164>/<peer_e164>`. `event_id = MessageSid` gives durable inbound dedup against Twilio's at-least-once webhook retries.

The two genuinely hard problems are **compliance** (opt-out, registration, spend) and the **at-least-once / at-most-once asymmetry** of the outbound send. The adversarial review's most valuable confirmed findings cluster on exactly these, and §4–§7 fold them as load-bearing design decisions rather than caveats.

---

## 2. Background: the AIOS connector contract

A connector is **not** a server-side plugin. It is: a `connector` *type* string + one shared `connectors.tools_schema` row for that type + a runtime container that speaks an HTTP/SSE protocol. The container subclasses `HttpConnector` and runs three outbound SSE tails plus a set of POSTs, all authenticated by a runtime bearer token (`aios_runtime_<…>`, sha256-hashed in DB) that scopes the caller to **one connector type**, **one account_id** (tenant), and an optional `connection_ids` allowlist (#350).

The contract is **single-direction**: the sidecar is always the client.

- **Discovery** — `GET /v1/connectors/connections` (SSE) backfills active connections of the type as `added`, then tails `connections_<connector>` NOTIFY. The runner spawns `serve_connection(connection_id, secrets)` per added connection.
- **Outbound origination** — `GET /v1/connectors/runtime/calls` (SSE) backfills pending `type=custom` tool calls the model emitted, then tails `connector_calls_<connector>`. The sidecar dispatches the `@tool` and answers with `POST /v1/connectors/runtime/tool-results`.
- **Inbound** — `POST /v1/connectors/runtime/inbound` (multipart) is **the only inbound ingress**, and it is *connector-originated*. Fields: `connection_id`, `event_id` (client dedup key), `chat_id`, `content`, `sender` (typed dict), `metadata`, `timestamp`, `attachments`. `handle_inbound` does dedup (`try_record_inbound_ack`, idempotent on `(account_id, connector, external_account_id, event_id)` *in the same txn as `append_event`*), three-tier session resolution, attachment staging, and `defer_wake`.
- **Lifecycle** — `POST /v1/connectors/runtime/lifecycle` appends a `kind=lifecycle` event onto **every** session bound to the connection (`list_session_ids_for_connection` unions the single_session binding + all per_chat `chat_sessions`). It carries `{connection_id, event, reason?, data?}` and has **no per-session targeting and no idempotency key**.
- **Secrets** — `GET /v1/connectors/runtime/secrets` is the only decrypt path; secrets are cached at `serve_connection` spawn and require a container restart to rotate.
- **Management plane** — `GET /v1/connectors/runtime/management-calls` (SSE) + `POST /v1/connectors/runtime/management-call-results`, bridged from operator-facing Section-2 routes via `management_calls.submit_call` + the `connector_result_<call_id>` long-poll. Operator-initiated, type-wide, *no auto-replay on result-POST failure* (visible operator timeout beats an SMS-storm).

**The webhook-inbound mismatch and its resolution.** The SDK's `serve_connection` docstring models a *polling* feed, and the runner stands up **no** HTTP server — it opens three outbound SSE clients and POSTs. Twilio, however, is push-only. The runtime contract is therefore complete for inbound *only if the connector itself originates the inbound*. The resolution (validated by the review): the sidecar runs **its own** webhook listener via the `setup(tg)` hook, terminating and verifying every Twilio request in-container, then bridging to `emit_inbound`. No new AIOS route, no public ingress on aios-api, trust boundary unmoved.

The HTTP-status retry contract on `/runtime/inbound` is load-bearing for how the sidecar acks Twilio: `413` payload-too-large; `422` DETACHED / ARCHIVED_TEMPLATE (terminal — operator config; stop retrying); `5xx` SESSION_MISSING / ATTACHMENT_STAGING_FAILED (transient — retry); `200 deduped=true` is success.

---

## 3. Design

### 3.1 Package structure

Path: **`connectors/sms/`** with package `src/aios_sms/` — connector-type-neutral (`connector = "sms"`), **not** `connectors/twilio/`. Rationale (§3.7): the agent-facing tool catalog and the AIOS connection/binding/resolver model are vendor-neutral; the provider is a per-connection discriminator, not a connector type. Mirrors signal/telegram layout:

```
connectors/sms/
  pyproject.toml          # mirrors telegram's; name aios-sms; deps: aios-connector-http, aiohttp, httpx
  src/aios_sms/
    __main__.py           # main() -> asyncio.run(SmsConnector().run_until_stopped())
    connector.py          # SmsConnector(HttpConnector); setup(tg), serve_connection, @tool sms_send
    webhook.py            # the aiohttp app: /twilio/inbound, /twilio/status; verify-before-parse
    verify.py             # WebhookVerifier (Twilio HMAC-SHA1), lifted from openclaw webhook-security.ts
    provider.py           # MessageSender + InboundNormalizer protocols; Twilio arm only in v0
    consent.py            # durable consent ledger gate (see §4)
    outbound.py           # durable outbound-attempt record + send-dedup + MessageSid→session correlation (see §3.6, §5)
    addressing.py         # E.164 normalizer (lifted from whatsapp normalize_phone), focal_channel helpers
    state.py              # _SmsConnectionState dataclass
    management.py         # SmsManagementMixin: deferred provisioning/registration handlers
```

`_SmsConnectionState`: `account_sid`, `api_key_sid`/`api_key_secret` (send creds), `auth_token` (verify key), `from_number`/`messaging_service_sid` (the sender, a discriminated kind), `sender_kind` (`long_code | toll_free | short_code`), `registration_status`, per-connection inbound `asyncio.Queue`, and **handles to the durable stores** (consent ledger, outbound-attempt log, spend ledger) — the in-memory state holds no compliance-critical counters (see §4, §5).

### 3.2 Inbound webhook receiver — where it lands, signature verification, dedup

**One container-wide listener in `setup(tg)`.** Twilio posts to one URL per number/MessagingService, so a single shared `aiohttp` app demuxing by the normalized **`To`-number → connection_id** is the right shape — the exact analog of signal's single shared `signal-cli` daemon + dispatcher fan-out. `setup(tg)` (which runs once, before discovery, and is handed only the TaskGroup) spawns it via `tg.create_task`. `serve_connection` registers its connection's normalized `from_number → (connection_id, auth_token)` into the shared demux map as connections are discovered, then drains its per-connection queue.

**Two routes, both Twilio webhooks carrying `X-Twilio-Signature`, both verified before parse:**

- `POST /twilio/inbound` — MO messages → enqueue → drain → `emit_inbound`.
- `POST /twilio/status` — delivery status callbacks → routed to the originating session (§3.6).

The inbound route returns empty TwiML `<Response></Response>` (`text/xml`) so Twilio sends nothing synchronously; the agent's reply is always async via `sms_send`. A synchronous TwiML reply is structurally impossible because AIOS outbound is model-tool-driven.

**Decouple the Twilio ack from the `emit_inbound` round-trip.** The handler does: verify → enqueue onto the per-connection queue → **immediately return 200 + empty TwiML**. `serve_connection` drains the queue and calls `emit_inbound` off the webhook's critical path (signal's dispatcher→queue→drain shape). This keeps the on-loop time per request bounded to verify+enqueue, removes head-of-line stall across numbers when `emit_inbound` is slow or a per_chat session is being spawned server-side, and lets Twilio's retry + the durable `inbound_acks` ledger absorb any retry that fires before the async `emit_inbound` commits.

**`X-Twilio-Signature` verification — `verify.py`, lifted from openclaw `webhook-security.ts` (`validateTwilioSignature` / `buildTwilioDataToSign` / `reconstructWebhookUrl`):**

1. Read the **raw** body bytes (re-serializing breaks HMAC equality).
2. Route by the normalized `To`-number to the connection; fetch its cached `auth_token`. This is **route-then-verify**, which is correct-by-construction: `To` is one of the signed POST params, so an attacker cannot alter it without holding the `auth_token`. The verify key and the routing key are the same signed value; a misroute (e.g. normalizer drift) verifies against the *wrong* connection's distinct token and **fails closed (403)** — never a silent cross-connection accept.
3. Reconstruct the exact public URL Twilio signed. **Prefer the operator-configured public base URL** (the registered webhook hostname) as the canonical signing URL; only fall back to `X-Forwarded-Proto/Host` when no configured base exists, and then require **both** a non-empty `allowedHosts` allowlist **and** a socket-peer-IP match against the known proxy, reject `@` in host, validate RFC-1123. For SMS over HTTPS the **port is kept** (Twilio drops the port only for Voice). Pin the expected port from config; the port-variant retry set is bounded to the configured port.
4. `HMAC-SHA1(auth_token, url + concat(sortedByKey(name+value)))`, base64, **constant-time compare**.
5. **Fail closed (403)** on missing/invalid signature. **No skip-verification-when-no-token fallback** (the GHSA-4hg8-92x6-h2f3 vuln class). For the brief per-number **cold-start window** — a webhook arrives for a `To` whose connection has not yet been discovered and whose `auth_token` is therefore not yet in the demux map — return a **transient 5xx** (not 403) so Twilio retries once discovery completes. The property to document and uphold: *cold-start inbound is briefly unavailable per number, never briefly unauthenticated.*

**Retry/dedup via `MessageSid → event_id`.** `emit_inbound` always passes `event_id = MessageSid`, pinned as a single-source invariant (a test asserts the deprecated `SmsSid`/`SmsMessageSid` aliases are never used). Two layers, distinct roles:
- *Fast path*: a process-local replay cache keyed `sha256(url | canonical-params | signature)`, ~10-min window — an optimization, lost on restart, **not** the durable guarantee.
- *Durable guarantee*: AIOS's `try_record_inbound_ack` ledger on `(account_id, connector, external_account_id, event_id)`. A Twilio at-least-once retry collapses to `deduped=true` even across a container restart.

**Inbound segmentation.** Twilio normally reassembles a concatenated MO message into one webhook (one `MessageSid`, full `Body`, `NumSegments` set). But carriers/handsets that don't preserve the UDH concat header deliver segments as separate webhooks with distinct `MessageSids` → distinct `event_ids` → no dedup collision → fragmented user turns and a model that may reply to a partial message. **Decision:** set `inbound_debounce_seconds > 0` for SMS connections (the existing #799 primitive — `queueing_lock=session_id` coalesces a sender's burst into one wake), and document inbound-segmentation semantics explicitly. Debounce coalesces the *wakes* (the model sees the burst as one turn, no premature reply, no double-bill) without a new reassembly mechanism.

### 3.3 Identity / `focal_channel`

- `external_account_id` = the AIOS-owned Twilio number, stored slash-free normalized E.164 (e.g. `+18005551234` — leading `+` and digits satisfy `ConnectionCreate._no_slash`).
- `chat_id` = the peer number, E.164 (no `/`, ≤512 chars).
- `focal_channel = sms/+18005551234/+14155550000`. The runtime re-parses it into injected `external_account_id`/`chat_id` kwargs (`_FOCAL_INJECTABLE`), so `@tool` methods declare them keyword-only and never receive them from the model.
- A **single E.164 normalizer** (lifted from whatsapp `normalize_phone`: strip spaces/dashes, force leading `+`) applied **symmetrically** at the webhook `From`/`To` parse and at any management-handler `external_account_id` lookup, with **digits-only compare** — or operator formatting drift silently misroutes (signal's `account.strip()` message-loss lesson).
- Sender display name flows **only** through the typed `sender={"display_name": From}` dict, never `connector_metadata["sender_name"]` (reserved + stripped server-side; forging it spoofs a trusted identity in the renderer's `from=` clause).
- Two tenants can independently own the same number (migration 0060); the runtime bearer token resolves to exactly one `account_id` and every `/runtime/*` route scopes `get_connection` by it, so the same-number demux within a sidecar is unambiguous and the tenant boundary is server-enforced regardless of sidecar routing.

**`From` is a routing key, not a trust anchor.** SMS `From` is unauthenticated and trivially spoofable (no SPF/DMARC analog); a valid `X-Twilio-Signature` attests only that *Twilio relayed* the webhook, not that the claimed `From` owns the number. Two consequences are designed-for, not waved away (§5).

### 3.4 The `@tool` outbound vocabulary

| Tool | MVP? | Justification |
|---|---|---|
| `sms_send(body, *, connection_id, chat_id, external_account_id)` | **MVP** | The one irreducible outbound primitive. SMS has **no edit/delete/react/typing/read-receipt** wire protocol, so per *compose-don't-accrete* it is the only model-callable tool. ~80–90% of telegram's surface (`format.py`/`parse_mode`/reactions/edits/media-groups/typing/`prompts.py`) has no SMS analog and is deleted, not ported. |
| `mms_send(body, media: list[SandboxPath], *, …)` | **deferred** | A *separate* tool: the MMS 5 MB / ≤10-media / restricted-MIME envelope, the different endpoint, and the higher cost differ enough that the constraints belong in the schema the model sees. |

`sms_send` flow: model emits the call → `/runtime/calls` SSE → `dispatch_call` → **durable outbound-attempt write (pre-POST, §3.6/§5)** → **consent-ledger gate** (terminal `is_error` "recipient opted out" if opted_out) → **spend/throughput gate** (terminal `is_error` "send budget exhausted" if paused) → `POST /Messages.json` (HTTP Basic auth with the **API Key SID+secret**, form-encoded `To=chat_id`, `From`|`MessagingServiceSid`, `Body`, `StatusCallback` pointing at the connection-scoped status URL) → record `MessageSid` on the attempt row → `/runtime/tool-results`.

`body` is run through a **strip-markdown** normalizer (lift openclaw `stripMarkdown` — SMS is **plain** text, *not* WhatsApp/Signal asterisk-markup) + markdown-table→plain degradation, and chunked on paragraph boundaries for the 1600-char hard limit (defer carrier segmentation). The result stamps `{message_sid, channel: focal_channel(...), segments}`.

The `@tool` **docstring** is the sole agent-facing affordance prose (`prompts.py` is vestigial in HTTP connectors). It must state: *"Your text responses are NOT delivered automatically — call this tool or no one sees your reply"*; *"SMS is plain UTF-8, no markdown (asterisks render literally); ~160 GSM-7 chars per billed segment, 70 if any emoji/accent, hard cap 1600"*; and surface provider failures as **stable runtime concepts** ("recipient opted out", "sender not registered for messaging", "last send was blocked"), never raw `21610`/`30034`.

### 3.5 Status-callback → lifecycle events

Twilio POSTs `application/x-www-form-urlencoded` status callbacks on each status change (`queued → sending → sent → delivered | undelivered | failed`). These are Twilio webhooks and carry `X-Twilio-Signature` — verified with the **same HMAC algorithm** as inbound, **but a different key-lookup axis** (this is a must-fix the review surfaced; "verified identically" was false as originally stated):

- On an **outbound** status callback, `From` = the AIOS-owned number (= `external_account_id` = the connection) and `To` = the peer. The verify key is therefore the **`From`-number's** connection's `auth_token`, the *opposite* axis from inbound (where `To` = our number). The status route routes by `From`; this is restart-robust because the `from_number → connection` demux map is rebuilt deterministically from the discovery SSE on every restart.
- Map the message to its **originating session** via the durable `MessageSid → (connection_id, session_id, chat_id, tool_call_id)` correlation written at send time (§3.6) — emphatically **not** the connection-wide `emit_lifecycle` fan-out, which would inject a per-peer `failed` into every per_chat session.

Surfacing (model-consciousness heuristic): a **terminal carrier-block / delivery-failure** (`failed`/`undelivered`, e.g. `30007` carrier-filtered, `30034`/`30032` registration) is **not** billing churn — it means the message the model *consciously sent* did not arrive. It must reach the **originating session** in a form the model acts on. Two implementation requirements fall out:
1. A **session-targeted** lifecycle append (a `session_id`-scoped variant of the lifecycle route, or route the per-peer failure through the resolver on the callback's `To`), since the existing `/runtime/lifecycle` route has no per-session targeting.
2. The carrier-block error classes must be model-visible. The only path that surfaces non-`message` events to the model today is the hardcoded `MODEL_VISIBLE_LIFECYCLE_EVENTS` allowlist (three sandbox-FS events). **AIOS-side change (§6):** add the SMS delivery-failure lifecycle kind to that allowlist, or deliver the failure as a `tool_result`-shaped follow-up the model reasons about. Pure `queued`/`sent`/`delivered` churn stays in the ops log; if a correlation row is genuinely missing (past retention), log to ops and **drop** rather than fan a spurious cross-peer failure.

### 3.6 The durable outbound record — one row, three jobs

A single durable store (the connector-owned SQLite the consent ledger uses, or — preferred long-term — an AIOS-side per-connection store, §6) holds `outbound_attempts(tool_call_id PK, connection_id, session_id, chat_id, message_sid, status, est_cost)`. It is written **before** the Twilio POST and updated with the `MessageSid` after the 201. It serves three jobs that the review proved cannot be served by the SDK's post-dispatch answered-set:

1. **Send idempotency** (§5, must-fix): on a backfill replay after a crash, an existing attempt row with no confirmed `message_sid` triggers reconciliation against Twilio (`GET /Messages` by the recorded SID or a deterministic correlation), **not** a re-POST.
2. **Status-callback correlation** (§3.5): `MessageSid → session_id` for session-targeted delivery-failure surfacing.
3. **Durable spend accounting** (§4): `est_cost` debits a durable per-connection spend ledger in the same critical section as the attempt write.

### 3.7 Provider strategy — Twilio-first, agnostic seam

`connector = "sms"` is a **capability type**, provider-neutral. The agent-facing catalog and the connection/binding/resolver model never change when a second provider lands; the provider is a per-connection `provider` discriminator in the secrets.

Per AIOS doctrine (*compose, don't accrete; unify only once you've found the right primitive; three similar lines beat one over-engineered helper*), **v0 ships only the Twilio arm and does not build a one-impl `Protocol` abstraction.** The connector fleet (signal/whatsapp/telegram/echo) uses plain modules and free functions for exactly this parse/normalize/send factoring — never an interface-with-one-impl. Keep only the genuinely-orthogonal split that pays off regardless of providers: **verify/normalize** (`verify.py` + an inbound normalizer mapping the Twilio form to a common `{from, to, body, media[], provider_message_id, status?}`) is distinct from **send** (`outbound.py`).

When a *real* second provider is requested, ask the AIOS question first: is it a new `connector` type, or an arm inside this sidecar? For interchangeable SMS *transports* (Twilio / Telnyx / Bandwidth / Vonage — same agent semantics) the answer is an arm behind the verify/send seam (a mechanical extraction once the second provider's real requirements exist: Telnyx Ed25519/JSON, Bandwidth no-sig+direction-check, Vonage JWT). For a *different platform* with distinct semantics, it is a new connector type. The decision is made then, against real requirements, not pre-built now.

---

## 4. Compliance model

Compliance is not a layer on top of the connector; it is the connector's reason for being careful. Three items are **non-negotiable even in MVP**.

### 4.1 Opt-out as a correctness invariant — with the spoof and durability holes closed

A connector-owned **consent ledger**, keyed at **Messaging-Service/Sender + peer** granularity (a STOP to one number in a Messaging Service opts the peer out of the *entire* service; cross-channel RCS/SMS/MMS from 2026-03-16), gates every `sms_send`. The ledger is consulted before any provider call, so an opted-out `(sender, peer)` pair raises a terminal `is_error` before any send.

The review forced four corrections that turn the keystone claim from aspirational into actually-holding:

1. **Opt-out is monotonic and NOT clearable by an inbound `From`** (must-fix, sev 91). The original "inbound START is recipient-controlled" claim was false: `From` is spoofable, so a spoofed single-word `START` from an attacker would lift a victim's opt-out — a TCPA violation by construction. **Decision:** an inbound `START` does **not** auto-clear an opt-out in the ledger. Opt-out is terminal for the `(sender, peer)` pair; re-enabling requires an out-of-band/operator action. We additionally lean on Twilio's carrier-layer record as the authoritative cutoff, but we never let a spoofable signal *raise* authority.
2. **STOP/HELP/START are control signals, consumed by the connector — not conversational content delivered to the model** (fold, sev 66). A STOP-family / HELP / START inbound (or one carrying `OptOutType` when Advanced Opt-Out is on) updates the ledger and is **not** `emit_inbound`'d as a user turn (optionally `emit_lifecycle` `sms.peer.opted_out` for connection-state awareness). HELP is answered with the mandated help text by the connector/carrier, never routed to the model. This removes the double-handling, the wasted inference, and the model-replies-to-STOP failure mode, and is consistent with how status callbacks are already kept out of the model's view.
3. **The matcher must be broader than single-word standard keywords** (fold, sev 62). Twilio's carrier auto-STOP fires only on single-word English keywords on long-code/toll-free; "please stop texting me" / non-English opt-outs evade both the carrier *and* a single-word matcher → the bot keeps sending. **Decision:** match STOP-family as a leading/standalone token case-insensitively; consume the authoritative `OptOutType` param when Advanced Opt-Out is enabled rather than re-deriving intent; flag ambiguous opt-out-intent inbound for conservative treatment.
4. **The ledger must be durable and coherent** (fold, sev 60). The original sidecar-local-SQLite home contradicted "same footing as gapless-seq": a container rebuild without a persisted volume, or a multi-container scale-out, can **resurrect an opted-out recipient**. **Decision:** the inbound STOP→`opted_out` write **commits durably before the inbound webhook is acked** (per-insert `commit()` gives this on one host), and the ledger's durable home is promoted to an **AIOS-side, account-scoped, per-connection store** (the #462 inbox-gating direction) so opt-out survives redeploy and coheres across containers by construction. If v0 must ship sidecar-local first, a persisted volume for the ledger DB is a **hard release-gate prerequisite** documented in eumemic-ops (a redeploy without it is a compliance incident).

**On the framing.** This is honestly a *best-effort runtime gate over an at-least-once dispatcher*, made durable and monotonic. The irreducible residual is the in-flight reply: a message already dispatched when STOP arrives cannot be recalled. For long-code/toll-free that racing message hits Twilio's server-side opt-out record (which Twilio learns *before* it forwards the STOP webhook to us) and returns **21610 — blocked, not billed, not delivered**. So the carrier is the authoritative cutoff for the race; the ledger makes the *next* send unrepresentable. We do not claim unrepresentability the storage layer can't uphold; we claim a durable, monotonic gate backstopped by the carrier on the configurations v0 ships.

**Carrier auto-STOP is sender-type-conditional.** It covers single-word STOP on long-code/toll-free only — **not** short codes without Advanced Opt-Out. **Decision:** `sender_kind` is a discriminated kind (`long_code | toll_free | short_code`), and a **`short_code` without Advanced-Opt-Out wiring is unrepresentable as a live v0 binding.** Short-code support is deferred until that wiring lands.

### 4.2 Registration state (A2P 10DLC / toll-free) — a fail-closed gate, not a prose precondition

Unregistered +1 10DLC traffic is fully blocked (`30034`) **and billed** since 2023-09-01; unverified toll-free is blocked (`30032`) **and billed** since 2024-01-31. These fail **asynchronously** after the API 202, so `sms_send` returns success and the model believes delivery happened while every message is silently dropped and billed during the multi-day TCR/verification window.

The original "an unregistered config is simply not a valid live binding" was an assertion with no mechanism (`models/connections.py` has no registration dimension). **Decision (close the gap):** add `registration_status (pending | approved | rejected)` to the SMS connection, polled from Twilio's A2P/TFN status API (the deferred `a2p_status` management handler is the read path). **Refuse to mark the binding live, and refuse the first send, until `status=approved`.** `sms_send` returns a terminal `is_error` "sender not yet approved for messaging" — a stable runtime concept — rather than letting `30034`/`30032` burn money async. This is the SMS analog of whatsapp `confirm-pairing`'s fail-closed JID check, applied to registration state.

### 4.3 Spend / throughput budget — a first-class, durable, AIOS-visible ceiling

Per-segment billing (~$0.011–0.015; Unicode silently halves the segment to 70 chars; blocked-but-billed `30034`/`30032` charge anyway) gives a model reply-loop or prompt-injected burst an uncapped real-money blast radius. A hard per-connection spend ceiling + per-number throughput awareness (TFN ~3 MPS, 10DLC ~75–150 MPS, T-Mobile per-brand daily caps) **pauses** outbound (a `paused_reason` discriminator, modeled on the workflow per-run compute ceiling) rather than looping; `sms_send` returns a terminal `is_error` the model can reason about.

The review confirmed the budget must be more than an in-memory counter:

- **Durable.** The counter lived in in-memory `_SmsConnectionState` and reset to zero on the exact event (crash/restart) most correlated with a runaway loop. **Decision:** the spend ledger is durable (same store as the outbound-attempt row), read-and-decremented in the same critical section as the pre-POST attempt write, so a restart resumes from true accumulated spend.
- **AIOS-side and operator-visible.** The existing `default_spend_limit_usd` gate (`config.py`, `harness/loop.py`) meters **model-inference cost only** and runs in a different process; it sees **$0** of Twilio egress. **Decision:** the sidecar reports each send's estimated cost via a runtime route that debits a Postgres-backed per-connection budget, so the ceiling survives restart, is enforced server-side, and surfaces SMS spend alongside inference spend on the operator dashboard. "$0-while-billing-thousands" must be impossible.
- **The ceiling value** is operator policy: an operator-facing default plus a per-connection override (too low silently drops legitimate replies; too high defeats the guard). This is an open fork (§9).

### 4.4 Quiet hours + prior-express-written-consent — deferred, flagged as legal duties

TCPA quiet hours (8 AM–9 PM recipient-local; FL 8–8 + 3/24h; TX 9–9) and prior-express-written-consent provenance are **operator/agent policy above the transport** — they need recipient-timezone resolution and consent provenance the connector doesn't hold, and they bind only **connector-initiated/marketing** messages (a purely reactive reply-bot is far lower-risk). Deferred, with the `triggers` `paused_reason` pattern as their home. **They become mandatory the moment the agent initiates outreach.**

---

## 5. Security model

Stated as invariants the implementation must hold.

1. **Verify before parse, fail closed.** Every inbound and status-callback request is HMAC-SHA1-verified over the **raw** body against the per-connection `auth_token` before any field is trusted; missing/invalid → 403; **no** skip-verification fallback. Cold-start (token not yet discovered) → transient 5xx, never accept.
2. **`From`-spoofing / trust — `From` carries zero capability.** A valid signature attests Twilio relayed the message, not that the `From` owns the number. Two designed consequences:
   - **Confused-deputy into a trusted session** (fold, sev 62): a spoofed `From` of a known peer routes attacker text into that peer's single_session / per_chat session. **Mechanization, not just documentation:** stamp SMS sender provenance as **explicitly unverified** toward the model (a metadata flag the system prompt teaches the model to treat as low-trust) so identity-gated tools refuse on SMS-origin alone; identity-gated actions require out-of-band verification, enforced in agent/harness policy, never assumed from `From`. Session binding on a spoofable `chat_id` is **not** an identity assertion — documented as a security-model invariant (the per_chat auto-spawn-on-first-contact note is a residual, §9).
   - **Opt-out cannot be raised by `From`** (§4.1).
3. **Public-ingress authentication & DoS posture.** The sidecar listener is the only public surface in the fleet. **Decisions:** (a) cap request body size **before** parsing; (b) require the `X-Twilio-Signature` header's presence as a cheap first filter and put the listener behind the eumemic-ops Traefik route with a Twilio source-IP allowlist + rate limiting; (c) run the synchronous HMAC/raw-body work — and any MMS byte download — off-loop via `asyncio.to_thread` (or `await`-ed async `httpx`), per the single-event-loop SDK constraint, so a flood cannot starve every number's inbound; (d) return a **uniform** response on any unverified/unroutable request so the endpoint is not a number-enumeration oracle (low-value but free); (e) bound the per-connection inbound queue and ack-200/shed on overflow rather than growing it unbounded (a dropped inbound is recoverable via Twilio retry; an OOM is not), and rate-limit unverified-`From` inbound at the listener edge so a spoofed-`From` flood can't grow queues for the whole container.
4. **Host-header-injection defense** (fold, sev 22 — the design already specifies the fail-closed gate; this hardens it): prefer the operator-configured public base URL as the canonical signing URL over header reconstruction; when reconstructing, require non-empty `allowedHosts` **and** socket-peer-IP (not header-derived) proxy match; the eumemic-ops ingress strips client-supplied `X-Forwarded-Host` and sets it authoritatively. Because the HMAC key is the per-connection `auth_token` selected by the signed `To`, controlling the URL component alone cannot forge a signature — a misconfig fails *closed* (availability), never open.
5. **`_RESERVED_METADATA_KEYS` boundary.** `channel`, `sender_name`, `attachments`, `platform_timestamp` are stripped from connector-supplied `connector_metadata` server-side. The sidecar passes the peer name via the typed `sender` dict and the Twilio timestamp via the typed `timestamp` field — never via metadata. Forging `sender_name` = spoofed trusted identity in the renderer's `from=` clause; forging `attachments` (with an `in_sandbox_path=/workspace/...`) = arbitrary `/workspace` file exfil via the harness vision renderer. The server alone writes attachment records.
6. **MMS-download SSRF** (residual until MMS ships, but a hard gate then): the deferred inbound-MMS fetch pulls `MediaUrlN` from `mms.twiliocdn.com` with Basic auth — and the credential it carries doubles as the HMAC verify key, so a leak = forge-any-inbound. **Requirements for the MMS slice:** use `await`-ed async `httpx` with `follow_redirects=False` (or strip `Authorization` on any cross-host redirect); pin fetches to a Twilio media-host allowlist with DNS-rebinding-safe resolution refusing private/link-local/loopback (mirror `aios.tools.url_safety.is_safe_url`, which the connector must replicate since connectors don't import `aios.*`); fetch + re-host server-side (media expires at 13 months); honor the 5 MiB SDK cap.
7. **Multi-number tenant isolation.** The runtime bearer resolves to one `account_id`; every `/runtime/*` route scopes by it; migration 0060 lets two tenants own the same number with **distinct** `auth_tokens`. Never resolve/dedup/decrypt/append across the account boundary.
8. **#794 capability-lattice position.** The sidecar adds **no ambient authority**. Its reach is bounded by the runtime bearer token's #350 scope (one connector type + optional `connection_ids`); per-connection routes enforce both type-scope and connection-scope. The novel capability it *holds* — egress authority to spend money via Twilio — is explicitly clamped: the consent ledger is the lattice **meet** on the recipient axis (opted-out = bottom = no representable send, and **not raisable by a spoofable inbound**), and the durable spend/throughput budget is the meet on the volume axis. The inbound `From` carries **zero** trust authority; the peer allowlist is the explicit grant. No part of the sidecar's authority composes upward into aios-api beyond the bearer-token scope.
9. **Provider-credential handling.** Twilio creds are per-connection secrets, write-only on the operator surface (`secrets_set: bool`), encrypted per-account-subkey, decrypted only via `/runtime/secrets`. **Decision (fold, sev 55):** use a **Twilio API Key (SID+secret) for outbound REST** so the *send* credential is independently revocable and is **not** the HMAC key; keep the master **`auth_token` solely for signature verification**. Add a structured-logging redaction invariant (never log secrets). Document explicitly that the spend/throughput budget protects against a runaway *model*, not a leaked-credential *attacker* (who bypasses the sidecar entirely) — credential confidentiality is the real send-abuse control. **Rotation runbook** (fold, sev 38): the `auth_token` is doubly-load-bearing (verify key + the verify side cannot be hot-reloaded) and cached at spawn; rotate via Twilio's primary+secondary grace window — rotate to a secondary token, restart the sidecar to pick it up while the old token is still valid for both verify and send, then deactivate the old token after the restart is confirmed healthy. The "no restart for credential rotation" memory note applies to the LiteLLM pool path, **not** connector secrets.

---

## 6. AIOS-side changes

**Pure-sidecar, no aios-api change, on the happy path.** The SMS connector speaks `/v1/connectors/runtime/*` verbatim (`inbound`, `secrets`, `tool-results`, `lifecycle`, discovery SSE, calls SSE). **aios-api gains NO public provider-webhook ingress** — the decisive resolution of the central question. The Twilio webhook lands on a listener the sidecar stands up itself; aios-api never sees a Twilio byte; its trust boundary stays at the runtime bearer token (grep-confirmed: zero signature-verify/`x-twilio`/`hmac` code in `src/*.py`). This mirrors signal/whatsapp/telegram exactly.

The review surfaced a small number of **genuine AIOS-side changes** the steelmanned design now requires (none a new public ingress):

1. **Model-visible SMS delivery failures** (from §3.5 must-fix). The status-callback `failed`/`undelivered` carrier-block classes must reach the originating session in a form the model acts on, but the only path that surfaces non-`message` events is the hardcoded `MODEL_VISIBLE_LIFECYCLE_EVENTS` allowlist. Either **add the SMS delivery-failure lifecycle kind to that allowlist** and give it stimulus (a wake), or add a `tool_result`-shaped follow-up channel for it.
2. **Session-targeted lifecycle append** (from §3.5). The existing `/runtime/lifecycle` route fans out to *every* bound session and carries no `session_id`. A session-scoped variant (or routing the per-peer failure through the resolver on the callback `To`) is required so a per-peer delivery failure does not pollute unrelated per_chat sessions.
3. **AIOS-side durable consent + spend stores** (from §4.1, §4.3). Promote the consent ledger and the spend budget to **account-scoped, per-connection** stores written via a runtime route — so opt-out survives redeploy and coheres multi-container, and SMS spend is enforced server-side and operator-visible alongside inference spend. This is the #462 inbox-gating direction; v0 may ship sidecar-local-with-persisted-volume as a stopgap behind a hard release gate, but the durable AIOS-side home is the correct shape and is on the critical path for the compliance invariant to actually hold.
4. **`registration_status` on the SMS connection** (from §4.2): a representable approval state + a binding-layer/first-send fail-closed gate.

**Deferred AIOS-side, additive:** a Section-2 operator-facing route block in `src/aios/api/routers/connectors.py` (mirroring signal `register`/`verify` and whatsapp `start-pairing`) bridging to `@management_handler` methods via the **existing** `management_calls.submit_call` + `connector_result_<call_id>` plumbing — for self-service number provisioning + A2P/TFN registration. This adds operator routes, **not** a public ingress, and reuses the management plane unchanged.

**The public ingress is a first-class deliverable, not an ops footnote.** It is the only inbound-reachable surface in the fleet, and the `X-Twilio-Signature` correctness depends on the TLS-termination point, the configured public base URL, the trusted-proxy set, the `allowedHosts` list, and the port-keeping rule. **Decision:** specify all of these as design artifacts the verifier is written against; co-deliver the eumemic-ops Traefik/Coolify route + TLS in lockstep (the durable-sandboxes promote-gate precedent: never ship the code half without the ops half); add a **startup self-test** that POSTs a synthetic signed request through the sidecar's own public URL and asserts it verifies (catching host/port/proto/cert drift before it silently eats traffic); monitor cert expiry and alert on a `403`-signature-failure rate `>0` and on a sustained inbound-rate drop. Re-frame the summary from "zero new core mechanisms" to the accurate **"no new aios-api mechanism, but the first inbound surface in the connector fleet — a deliberate, contained change to the fleet threat model."**

---

## 7. What the adversarial review changed — the steelman record

The review produced 30+ findings, independently verified with calibrated verdicts. The top confirmed findings and their effect on this design:

| Lens | Finding (refined sev) | What it changed |
|---|---|---|
| security | Spoofed `START` clears a victim's opt-out → TCPA-by-construction (91) | **Opt-out is monotonic; inbound `START` no longer auto-clears.** Killed the false "recipient-controlled, one-directional" claim. §4.1, §5.2 |
| compliance / arch-fit | Outbound `sms_send` is at-least-once: spool written **after** the billable POST → crash-window double-send + double-bill (78–82) | **Durable pre-POST outbound-attempt row + reconcile-not-resend on replay.** Dropped the false "spool prevents re-send" guarantee. §3.6, §5 (the #1/#19/#27/#33 cluster) |
| operability-cost | Unregistered-10DLC/unverified-TFN blocked-AND-billed async, no gate (72) | **`registration_status` + fail-closed bind/first-send gate**; `sms_send` returns "sender not yet approved". §4.2 |
| operability-cost | Status-callback delivery failure depends on a volatile, restart-fragile `MessageSid→session` map; lifecycle fans out to all sessions (70) | **Durable `MessageSid→session` correlation + session-targeted lifecycle append.** §3.5, §3.6, §6 |
| compliance | Unregistered/carrier-block failures surface only as `emit_lifecycle`, which the model **can't see** (70) | **Add SMS failure kind to `MODEL_VISIBLE_LIFECYCLE_EVENTS` / `tool_result`-shaped surfacing.** §3.5, §6 |
| compliance | STOP double-handled — mutates ledger **and** is delivered to the model as a user turn; HELP unanswered (66) | **STOP/HELP/START are connector-consumed control signals, not user messages.** §4.1 |
| operability-cost / compliance | Consent ledger + spend budget sidecar-local SQLite → rebuild/scale-out resurrects opt-out & resets budget (60, 62) | **Durable, commit-before-ack, AIOS-side account-scoped stores; persisted-volume release gate as stopgap.** §4.1, §4.3, §6 |
| security | Spoofed `From` injects attacker content into a trusted peer's session — confused deputy (62) | **Stamp SMS provenance as unverified to the model; identity-gated actions need out-of-band verification.** §5.2 |
| compliance | Carrier auto-STOP is sender-type-conditional; single-word-only matcher misses multi-word/non-English opt-outs (62) | **Broader opt-out matcher + consume `OptOutType`; short-code-without-Advanced-Opt-Out unrepresentable in v0.** §4.1 |
| security | `auth_token` is BOTH outbound send credential AND HMAC verify key (55) | **API Key SID+secret for send, `auth_token` verify-only; redaction invariant; rotation runbook.** §5.9 |
| security | Status-callback route has no durable dedup; forged/replayed signed callbacks fan spurious failures to all sessions (55) | **Durable `(connection, MessageSid, MessageStatus)` idempotency on the status path; correlate strictly.** §3.5 |
| arch-fit | "Verified identically" false — status callbacks route by `From`, not `To` (52) | **Status route keys on `From`-number connection; inbound on `To`.** §3.5 |
| security | Public webhook ingress: pre-verification DoS, single-loop starvation, enumeration oracle (52) | **Pre-parse body cap, IP-allowlist + rate-limit, off-loop HMAC/MMS, uniform responses, bounded queue.** §5.3 |
| correctness | Concatenated inbound SMS fragments into N user turns/wakes (48) | **`inbound_debounce_seconds > 0` for SMS + documented segmentation semantics.** §3.2 |
| arch-fit | "Zero new core mechanisms" understates the first inbound surface in the fleet; ingress under-specified (34) | **Re-framed; ingress promoted to a hardened, monitored, lockstep-delivered unit with a startup self-test.** §6 |

**Rejected / downgraded to residual (considered-and-dismissed):**

- *To-confusion / route-then-verify token confusion (sev 12)* — refuted: `To` is a signed param; a misroute fails closed against the wrong token.
- *Proxy `X-Forwarded-*` signature bypass (sev 22)* — refuted: the HMAC key is the per-connection `auth_token` selected by the signed `To`, not attacker-chosen; URL control alone can't forge a signature. The fail-closed gate is already specified; hardening folded into §5.4.
- *MMS-download SSRF (sev 38)* — MMS is deferred; folded as a hard gate for when it ships (§5.6), residual until then.
- *Proxy URL host-header injection at sev 64 / cold-start fail-open at sev 90* — both materially overstated; the design already mandates fail-closed and route-then-verify. Folded as the transient-5xx cold-start refinement (§3.2) and the configured-base-URL preference (§5.4).
- *`event_id` SmsSid/MessageSid alias drift (sev 8)* — speculation about unwritten code; foreclosed by the single-source `event_id = MessageSid` invariant + test (§3.2).
- *Single-loop MMS byte-read stall (sev 12)* — MMS deferred; async `httpx` doesn't block the loop; folded into §5.3.
- *Provider two-seam abstraction premature (sev 28)* — the `connector="sms"` decision is correct; the only fold is to express the verify/send split as plain functions, not a one-impl `Protocol` (§3.7).
- *Spoofed-first-contact per_chat TOCTOU (sev 18)* — not a TOCTOU; the spoof-as-routing-key property is the documented zero-trust stance. Residual (§9).
- *Send-time STOP TOCTOU as a delivered-after-STOP violation (sev 28)* — the racing message hits Twilio's server-side 21610; no illegal delivery on v0's sender types. Framing corrected (§4.1).

---

## 8. MVP vs deferred

**MVP (compliance items here are non-negotiable):**

- `SmsConnector(HttpConnector)`, `connector="sms"`, `_SmsConnectionState`; echo-http skeleton + signal fan-out shape.
- `setup(tg)` → one shared `aiohttp` app, `POST /twilio/inbound` + `POST /twilio/status`; verify-before-parse (raw body, route-by-`To`/`From`, configured-base-URL-preferred reconstruction, keep-port, constant-time, fail-closed 403, transient-5xx cold-start); pre-parse body cap; off-loop crypto; bounded queues.
- `serve_connection` → fetch creds via `/runtime/secrets`, normalize `from_number`, register demux mapping, drain queue → `emit_inbound(chat_id=From, sender={"display_name":From}, content=Body, event_id=MessageSid)`. `inbound_debounce_seconds > 0`.
- `sms_send` `@tool`: durable pre-POST attempt row → **consent gate** → **spend gate** → strip-markdown + chunk → `POST /Messages.json` (API Key Basic auth, `StatusCallback` set) → record `MessageSid` → stamp result; docstring carries all affordance prose.
- **Consent ledger** (Messaging-Service/Sender+peer keyed): STOP/HELP/START **consumed as control signals** (not emitted to the model), broad matcher, `OptOutType` when present, **opt-out monotonic & not raised by inbound `From`**, **commit-before-ack**, durable home.
- **Registration gate**: `registration_status`, fail-closed bind/first-send.
- **Spend/throughput budget**: durable, AIOS-side, operator-visible, pause-not-loop.
- **Status callback** → verify (by `From`) → durable correlate `MessageSid→session` → **session-targeted** delivery-failure surfacing that the model can see.
- Reuse `resolve_target_session` unchanged (`chat_id` = peer E.164); connection created detached, operator binds single_session/per_chat; default per_chat. Honor drop→HTTP-status when acking Twilio.
- Peer allowlist (`id`=E.164 + wildcard) enforced in-sidecar at the listener edge before `emit_inbound`.
- API Key for send / `auth_token` verify-only; structured-logging redaction.
- `SqliteAnsweredSpool` wired via `load_answered`/`save_answered`.
- eumemic-ops: one public HTTPS ingress route + TLS, delivered in lockstep, with startup self-test + cert/`403`-rate/inbound-rate monitoring.

**Deferred (additive):**

- MMS (`mms_send` separate tool; inbound `MediaUrlN` download behind the SSRF gate; `<media:image> (N)` placeholder; 13-month re-host).
- Group MMS as a second `chat_id` `chat_type` arm (validated reversible encoder, group = stable thread id).
- Self-service number provisioning + A2P/TFN registration `@management_handler` flow (fail-closed number-ownership verification, no-auto-replay posture) bridged from new Section-2 routes.
- Quiet-hours + prior-express-written-consent provenance (operator/agent policy above the transport) — mandatory once the agent initiates outreach.
- Messaging Services beyond the `From|MessagingService` kind (sticky-sender, geomatch, Smart Encoding, full Advanced-Opt-Out).
- Short-code senders + Advanced-Opt-Out wiring (a `short_code` binding is unrepresentable until this lands).
- Additional provider arms behind the verify/send seam (Telnyx/Bandwidth/Vonage).
- Hot secret reload — *rejected for v0*: it would special-case SMS against the deliberate restart-to-rotate contract every connector follows; use the primary/secondary grace-window runbook instead.

---

## 9. Residual risks & open design forks for the human

**Residual risks (accepted, documented, monitored):**

- **In-flight reply vs STOP race.** A message already dispatched when STOP arrives cannot be recalled at the application layer. On long-code/toll-free it hits Twilio's server-side `21610` (blocked, unbilled). Irreducible; documented honestly, not claimed-away.
- **Spoofed `From` as a routing key.** A spoofed `From` of a known peer lands attacker text in that peer's session (the generic untrusted-inbound property shared by all connectors). Mitigated by stamping provenance as unverified to the model + out-of-band verification for identity-gated actions; not eliminable (SMS has no sender authentication).
- **MMS SSRF / credential-on-redirect** — residual only because MMS is deferred; a hard gate (§5.6) is a prerequisite the moment `mms_send`/inbound MMS ships.
- **Number-enumeration oracle on the public listener** — low value (DIDs are public); uniform responses (§5.3) close it cheaply.
- **Status-callback field-set drift** — Twilio adds/removes callback fields without notice; parse tolerantly, key off `MessageSid` + `MessageStatus`, never branch business logic on specific error codes.

**Open forks requiring a human decision:**

1. **Consent + spend durable home — commit now or stopgap?** The steelmanned position is AIOS-side per-connection stores (#462 inbox-gating). The fork: build that AIOS-side primitive as a v0 prerequisite (correct, broader, slower), or ship sidecar-local-with-persisted-volume behind a hard release gate and promote later? *Recommendation: build the AIOS-side store for the consent ledger in v0 (it is the compliance invariant), spend budget can follow.*
2. **Peer-allowlist placement.** v0 enforces it in-sidecar (fast, no schema change). Promoting it to a first-class connection-level `allow_from`/`dm_policy` (#462) benefits all connectors. *Recommendation: sidecar now, promote once a second connector's requirements exist — building the core primitive correctly needs more than one connector's input.*
3. **Spend-ceiling default value + override authority.** Who sets the per-connection default, and at what number (segments/window, $/window)? Too low silently drops legitimate replies; too high defeats the guard. Needs an operator-facing default + per-connection override.
4. **Pre-provisioned vs self-service number for first ship.** v0 presumes a pre-provisioned, registered/verified Twilio number with the `registration_status` gate refusing sends until approved. Confirm this is an acceptable operator prerequisite, or pull the provisioning/A2P `@management_handler` flow into v0.
5. **`mms_send` as a separate tool vs `sms_send` with optional attachments.** Recommended split (constraints differ enough to live in the schema the model sees); confirm — a single tool is simpler if MMS is rare.
6. **Public-ingress ownership.** Confirm the dedicated eumemic-ops Coolify/Traefik route + TLS + monitoring (cert expiry, `403`-signature-rate, inbound-rate) is provisioned and signed off explicitly — the one net-new infra dependency vs the outbound-only connectors, and load-bearing for `X-Twilio-Signature` correctness.