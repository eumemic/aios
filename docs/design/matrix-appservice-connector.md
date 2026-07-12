# Design Doc: `matrix` — a Matrix Application Service connector for aios

Status: **implementation-ready for Milestones 0–4** (single homeserver, single connector process, fleet-pilot / demo scale). The Milestone-5 mechanisms — lifecycle/GC, spawn provisioning, admission/quotas, inbound durability, the E2EE path — are **designed and adversarially reviewed** (§12–§16); only the measurements (ceilings, throughput, shard count K) remain research-gated (Appendix A).
Author: (principal eng) — revised after red-team review.
Connector type string: `matrix` (v1). Horizontal sharding introduces per-shard types `matrix_s0…matrix_sK` — see §10.
Builds on: `aios-connector-http` SDK (`/Users/tom/code/aios/packages/aios-connector-http/`), `mautrix-python` v0.21.1 (`mautrix.appservice` + `mautrix.client`; **not** the deprecated `mautrix.bridge`, and **not** `mautrix.crypto` — it sits on deprecated libolm with two published CVEs and v1 never imports it; the E2EE crypto stack is a build-vs-wait spike, §16 E.3).

---

## 1. Goal & non-goals

### Goal
Give **every newly-spawned aios agent a first-class chat identity** on a Matrix homeserver, provisioned with **no phone number, no CAPTCHA, and no interactive step**. Once provisioned, an agent can:

- DM a human (reachable from Element or, via bridges, from Signal/WhatsApp/Telegram),
- talk peer-to-peer with other aios agents,
- participate in multi-party rooms (agents + humans).

The frictionless property is the point: unlike the `signal` / `whatsapp` connectors — SMS registration, QR pairing, the management-RPC dance — a Matrix appservice ghost is created by a **single authenticated call the connector makes itself** (in practice a *lazy* register on first use), with zero interaction. That is what makes "provision an identity at spawn" tractable.

### Scale target
Design **toward** millions of agent identities under one appservice namespace. Matrix's data model supports an unbounded virtual-user population per appservice, but three aios/Matrix pressure points — the **single serial HS→AS transaction stream per appservice**, the **one-connection-per-agent** mapping, and the **discovery backfill** — make "millions" a research-gated future, not a v1 deliverable. v1 delivers a working single-homeserver fleet pilot; the ceilings and the sharding plan are enumerated honestly in §10 and Appendix A.

### Non-goals for v1
- **E2EE rooms.** v1 agents live in **unencrypted** rooms only (§8). MSC3202/MSC4190 crypto is deferred.
- **Open federation.** v1 runs on a **closed / allowlisted-federation** homeserver (§9).
- **Rich puppeting of remote platforms.** We *consume* mautrix bridges as a reachability path (§8); we don't reimplement them.
- **Voice/VoIP, spaces, threads, receipt fidelity.** Deferred.
- **Hot secret rotation.** The SDK caches secrets at spawn (`runner.py:341-343`); rotation requires a container restart, as with every connector.
- **Verifiable agent authenticity.** v1 has **no cryptographic proof** that a ghost is the agent it claims to be (§9). That is an E2EE-era capability.

---

## 2. Why Matrix (one paragraph)

Matrix's Application Service API gives us the one property no other chat platform does: **an appservice holds a single `as_token` that can masquerade as any virtual user inside a reserved namespace** (`Authorization: Bearer <as_token>` + `?user_id=@ghost:server` on ordinary Client-Server calls). Provisioning an agent is therefore not an account-signup with a phone number and a CAPTCHA — it is `POST /_matrix/client/v3/register {"type":"m.login.application_service","username":"…"}`, idempotent, no interaction, and in practice **lazy** (mautrix's `IntentAPI` auto-registers a ghost the first time it acts, swallowing `M_USER_IN_USE`). One token → a whole population of addressable identities is exactly the shape "an identity per agent" needs. The cost of that single token is a matching single point of failure and a single serial ingress stream — both load-bearing for the scale story and both handed to the research track in **Appendix A** (homeserver choice, ghost/room/transaction ceilings, and namespace-sharding topology). The rest of this doc is written to be homeserver-agnostic and to shard cleanly along the namespace axis.

---

## 3. Architecture

### 3.1 The connector is two servers in one process

Every aios connector is a standalone process whose **entire contract with aios is the management HTTP API** (`README.md:17-18`) — it shares no DB/process with the worker. The `matrix` connector is unusual because it must **simultaneously** be:

1. **an aios connector** — a subclass of `HttpConnector` (`runner.py:236-246`) that publishes tools, tails the runtime SSE streams, and POSTs inbound/results; and
2. **a Matrix Application Service** — an **aiohttp HTTP server** receiving `PUT /_matrix/app/v1/transactions/{txnId}` pushes *from the homeserver*.

This is the key divergence from `telegram`, which is **poll-only** (`connector.py:104-137`): Telegram's `serve_connection` runs a per-connection long-poll loop, whereas Matrix inbound arrives on **one shared HS→AS stream for the entire appservice** (all ghosts, all rooms). So in the Matrix connector, `serve_connection` does *not* poll — it registers the connection into an in-memory routing map — and the inbound receiver is a single process-wide aiohttp server started in the `setup(tg)` hook (`runner.py:318-331`). **The consequences of that single shared receiver — a per-container single point of failure and a serial ingress ceiling — are made explicit in §5.5 and §10; they are not free.**

We do **not** hand-roll the AS server. We use **mautrix-python's `mautrix.appservice.AppService`** (its aiohttp server owns the `transactions/{txnId}` route, `hs_token` validation, and txn de-dup) and its **`IntentAPI`** as the "act as any agent user" abstraction: `self.az.intent("@…:server")` returns a masquerading CS-API client that auto-registers the ghost on first use. We depend only on `mautrix.appservice` + `mautrix.client` (`mautrix.crypto` is deliberately excluded — deprecated libolm, two published CVEs, no vodozemac migration; see §16 E.3).

### 3.2 Two-sided data flow

```
                         ┌────────────── aios management API (Bearer aios_runtime_…, connector "matrix") ─────────┐
   Homeserver            │   GET  /v1/connectors/connections           (discovery SSE)  ── added/removed ──▶      │
  (Synapse/…)            │   GET  /v1/connectors/runtime/secrets?connection_id=…         ◀── as needed ──          │
      │                  │   GET  /v1/connectors/runtime/calls         (tool-call SSE)   ── call ──────────▶       │
      │  HS→AS           │   POST /v1/connectors/runtime/inbound       (multipart)       ◀── emit_inbound ─        │
      │  Bearer hs_token │   POST /v1/connectors/runtime/tool-results  (json)            ── result ────────▶       │
      ▼                  │   PUT  /v1/connectors/matrix/tools_schema   (startup, once)   ── catalog ───────▶       │
 ┌──────────────┐        └───────────────────────────────────────┬──────────────────────────────────────────┘
 │ PUT /_matrix │                                                │
 │ /app/v1/     │        ┌───────────────────────────────────────▼──────────────────────────────────────────┐
 │ transactions │  events│                     matrix connector process (HttpConnector)                       │
 │ /{txnId}     ├───────▶│  ┌──────────────────────────┐      ┌──────────────────────────────────────────┐    │
 └──────────────┘        │  │ mautrix AppService       │      │ HttpConnector runtime loops              │    │
      ▲   AS→HS          │  │  (supervised aiohttp)    │      │  • discovery loop → conn_id↔ghost map    │    │
      │   Bearer as_token│  │  • hs_token validate     │      │  • tool-call loop → dispatch @tool       │    │
      │   ?user_id=@ghost│  │  • txn de-dup            │      │                                          │    │
 ┌──────────────┐        │  │  • SYNCHRONOUS fan-out ──┼──────┼─▶ emit_inbound(...) then ack 200         │    │
 │ CS-API:      │◀───────┤  └──────────────────────────┘      └───────────────┬──────────────────────────┘    │
 │ register /   │  Intent│                 ┌────────────────────────────────────▼─────────┐                    │
 │ send / invite│  API   │                 │  @tool bodies → self.az.intent("@agent:srv") │                    │
 │ createRoom / │◀───────┼─────────────────┤  send / react / createRoom / invite / join…  │                    │
 │ join / leave │        │                 └──────────────────────────────────────────────┘                    │
 └──────────────┘        └────────────────────────────────────────────────────────────────────────────────────┘
```

Left edge is Matrix (bidirectional). Right edge is the aios contract. **Token directions (verified against the AS spec):** `hs_token` authenticates **HS→AS** transaction pushes; `as_token` authenticates **AS→HS** masqueraded CS-API calls. No reversal.

### 3.3 Lifecycle wiring onto the SDK

- **`connector = "matrix"`** class attribute (`runner.py:236-246`).
- **`setup(tg)`** (`runner.py:318-331`): construct the mautrix `AppService(...)` from container config (§4), register a transaction handler that **fans out synchronously before acking** (§5.5), and `tg.create_task(...)` the aiohttp receiver alongside the SDK's SSE loops. **The receiver's failure posture is §15's closed KIND table (ships in v1):** aios 5xx / transport errors → non-200 (the HS retries the txn); 401/403 or any unenumerated exception → non-200 for the current txn **plus** a halt signal to a dedicated supervisor task that `setup(tg)` spawns as a real TaskGroup child — the supervisor raises, the TaskGroup tears down, the container restarts and re-reads env. The transaction handler itself **never raises into aiohttp**: aiohttp swallows handler exceptions into a 500 (`web_protocol.py` `_handle_request`), so a raise there restarts nothing — the out-of-band supervisor is what makes HALT actually halt (§5.5/§15).
- **`serve_connection(connection_id, secrets)`** (`runner.py:333-346`): **no polling.** Populate the `ghost_localpart → connection_id` reverse index and pre-warm the ghost's `IntentAPI` (idempotent register + profile set, §4.3). Because inbound is a shared stream, this hook exists only to populate routing state.
- **Tool loop / dispatch / focal injection** (`runner.py:1145-1199`, `1201-1432`, `1782-1807`): inherited unchanged.
- **`teardown()`** stops the aiohttp server on `docker stop` (SDK traps SIGTERM via `run_until_stopped`, `runner.py:840-872`).
- **Durable outbound de-dup:** wire `SqliteAnsweredSpool` (`spool.py:36-72`) — **mandatory**, or a restart double-executes sends on the tool-call backfill (`spool.py:1-13`). We *also* get free-ish idempotency from Matrix: every outbound send uses the aios `tool_call_id` as the CS-API `{txnId}` (§6.2), so a double-dispatch usually collapses at the homeserver — but Synapse's send-dedup is a bounded, best-effort cache and not guaranteed across HS restarts, so `SqliteAnsweredSpool` is the real guard and the HS dedup is a bonus.

---

## 4. Identity provisioning

### 4.1 The core mapping — one aios connection **per agent**

The load-bearing choice. Telegram maps **one connection = one bot identity** (`external_account_id = <bot_id>`), bound `single_session`. We mirror that shape exactly, with the identity axis carrying the **agent**:

> **A `matrix` connection is `(connector="matrix", external_account_id="<agent_localpart>")`, bound `single_session` to that agent's session.**
> `external_account_id` is the ghost localpart; the ghost MXID is `@<localpart>:<server_name>`. **Normative form: `_aios_agent_<s><ulid>`** — `<s>` is one fixed-position shard character chosen *balanced at provisioning step 0* (random over the explicit base32 shard alphabet; **never derived from ULID internals** — a ULID's leading chars are timestamp bits, constant `0` until ~2248, which would silently collapse the §10.1 shard partition onto one shard), followed by the agent's fresh lowercased ULID (machine-generated, never reused, so collision with a deactivated localpart is impossible by construction). `_aios_agent_42`-style names in examples are illustrative shorthand; M1's hand-named pilot ghost is an accepted one-off.

Why per-agent rather than one-connection-with-a-send-as axis: aios has **no outbound "send-as" dimension** — `focal_channel = {connector}/{external_account_id}/{chat_id}` (`runner.py:1799-1806`) carries no `sender_mxid`. Rather than invent that axis, we encode the agent's puppet identity in `external_account_id`, which the contract already threads end-to-end. Result:

- **Outbound identity is free and needs no SDK change.** `external_account_id` is a runtime-injected tool kwarg the SDK already fills from `focal_channel` (`schema.py:41-47`, `runner.py:1799-1804`), and the SDK already stores it on `self._connections[connection_id].external_account_id`, populated *before* `serve_connection` runs (`runner.py:1016-1026`). A tool body declares `external_account_id`; the SDK injects it. For the no-focal case (`matrix_create_room` initiating a brand-new conversation) it resolves from `connection_id` via `self._connections`. Either way we call `self.az.intent(f"@{external_account_id}:{server_name}")`. **No new focal-channel surface, no `ConnState.ghost_localpart` duplicate, no SDK affordance.**
- **A stable identity spans many rooms.** Each room is a `chat_id` (`!room:server`) *within* the one connection — orthogonal to identity. This is why we do **not** use `per_chat` mode (which would shatter the identity into a session per room).
- **Illegal states unrepresentable.** Exactly one Matrix identity per connection, as `bindings_connection_active_uniq` and per-account `(connector, external_account_id)` uniqueness intend (`models/connections.py:1-23`).

`external_account_id` may not contain `/` (`models/connections.py:69-74`); a localpart like `_aios_agent_42` is fine, and Matrix room IDs (`!opaque:server`) contain no `/`, so `matrix/_aios_agent_42/!room:server` round-trips cleanly through `_inject_focal_kwargs`.

### 4.2 The reserved namespace — committed form

We commit to the underscore-prefixed, collision-dodging form in the spec body and the Milestone-0 YAML. The registration is effectively permanent (changing the regex strands existing ghosts), so this is a decision, not shorthand:

```yaml
namespaces:
  users:
    - exclusive: true
      regex: "^@_aios_agent_[a-z0-9]+:your\\.server$"   # anchored, dot escaped
  aliases: []
  rooms: []
```

Notes, per the correctness review: Synapse matches namespaces with `re.match` (start-anchored only), so we **escape the `.`** and **add `$`** to prevent over-matching (`your.server` would otherwise match `yourXserver.evil`). MXID localparts are lowercase per grammar, so `[a-z0-9]` is safe for the variable suffix; the literal `_aios_agent_` prefix carries the underscores. `exclusive: true` reserves the prefix — no human and no other appservice can take an `@_aios_agent_*` MXID. The connector's own bot is `sender_localpart: _aios` → `@_aios:your.server` (outside the ghost regex; the HS reserves it implicitly), the default actor and inviter-of-record where a ghost isn't the natural actor.

### 4.3 How an agent gets an identity at spawn

Two independent steps; **only the second touches the connector**:

1. **aios-side provisioning — orchestration over three *existing* endpoints (not new API surface).** When an agent should be Matrix-reachable, aios orchestration, using an **operator API key** (`AccountIdDep`), composes endpoints that already exist:
   - `POST /v1/connections {connector:"matrix", external_account_id:"_aios_agent_42"}` (no secrets) — `connections.py:47`,
   - `POST /v1/connections/{id}/attach {session_id:"…"}` → `single_session` — `connections.py:247`,
   - `PUT /v1/connections/{id}/inbound-policy {…}` (§4.5) — `connections.py:106`.
   The connector **cannot** do this (connection-create is operator-key auth; a runtime token only *reads* connections). This is **orchestration wiring, not new aios surface** — it invents no endpoint. The narrower honest open question is whether there is an existing agent-spawn hook to attach it to (§10).

2. **Matrix-side registration — zero-step, inside the connector.** The connection surfaces on discovery (`GET /v1/connectors/connections`, `{"event":"added","connection_id","external_account_id"}`, `connectors.py:492-539`). `serve_connection` pre-warms `self.az.intent("@_aios_agent_42:server")`, which lazily does `POST /_matrix/client/v3/register {"type":"m.login.application_service","username":"_aios_agent_42","inhibit_login":true}` — idempotent (`M_USER_IN_USE` ⇒ already provisioned), no phone, no CAPTCHA — and sets the ghost's display name/avatar (§9).

   **On `inhibit_login`:** it is **recommended always** — it stops the HS minting an unused `access_token`/`device_id` for a ghost we drive purely via `as_token` masquerade — and becomes **required** on homeservers that have disabled legacy password login (e.g. those fronted by Matrix Authentication Service), where omitting it yields `M_APPSERVICE_LOGIN_UNSUPPORTED`. (This corrects an earlier draft's fabricated "required on v1.17+ homeservers that dropped legacy login," which conflated Matrix spec versions with Synapse versions.)

The "single call, no manual step" property lives in step 2; step 1 is aios plumbing that runs automatically at spawn.

### 4.4 Where `as_token` / `hs_token` live — **container config, not per-connection secrets**

**Verdict: appservice credentials (`as_token`, `hs_token`, `hs_url`, `server_name`, `sender_localpart`, namespace regex) live in connector *container config* (env vars). Per-connection encrypted secrets are empty in v1.** This is the *correct* answer, not a deviation to apologize for: the aios convention that "platform creds live on the connection secrets dict" is about *per-connection platform-identity* creds (a bot token, a phone registration). Matrix genuinely has **none** per connection in v1 — the `as_token` is appservice-wide infrastructure, exactly analogous to the shared `signal-cli` daemon config that a `setup(tg)` daemon is documented to serve. Replicating one `as_token` into millions of connection secrets dicts would make rotation a millions-row rewrite and buys nothing (secrets cache at spawn; rotation needs a restart regardless). The credentials are also needed **at boot**, before any connection is known: the receiver must validate `hs_token` on the first HS→AS push and hold `as_token` to act — `setup(tg)` runs before discovery surfaces anything, so env is the natural source, exactly like `AIOS_URL`/`AIOS_RUNTIME_TOKEN`.

Two things this pins down (per the aios-fit review):

- **(a) Env as a sanctioned surface.** Arbitrary `MATRIX_*` container env is an operator/Coolify deployment concern *outside* the connector contract; the design assumes it is sanctioned rather than silently relying on it. Concrete env: `MATRIX_HS_URL`, `MATRIX_SERVER_NAME`, `MATRIX_AS_TOKEN`, `MATRIX_HS_TOKEN`, `MATRIX_SENDER_LOCALPART`, `MATRIX_USER_NAMESPACE_REGEX`, `MATRIX_LISTEN_ADDR`.
- **(b) The ghost MXID is split across two sources** — localpart on the connection (`external_account_id`), `server_name` in container env. **Moving a connection between containers with a different `MATRIX_SERVER_NAME` silently re-identifies the agent.** This coupling is a deployment invariant: a connection's shard/container must own the `server_name` its localpart was minted under.

**Blast radius (be explicit):** the single container-env `as_token` can masquerade as **every** ghost and read **every** room any ghost is in; the `hs_token` lets anyone who holds it inject forged transactions (§9/#10). Compromise of the connector process or its env = **total fleet identity compromise within that shard**. This makes namespace-sharding (§10) a *security* control (blast-radius cap), not merely a scale control. The AS endpoint must be **private** (HS↔connector on a trusted network, never public); both tokens are fleet-critical secrets.

*Alternative considered and rejected for v1:* a "control connection" holding the registration bundle in the encrypted vault, fetched during `setup(tg)`. It keeps the token in the per-account vault but adds a boot-ordering dependency (the receiver can't validate `hs_token` until that connection's secrets are fetched) for no rotation benefit. Reconsider only if an operator policy forbids secrets in env.

*(E2EE narrows this: the crypto store — ratcheting Olm/Megolm state — is connector-local Postgres beside the `state_store`, not secrets material; at most rung 2's slow-moving per-ghost seed material would fit the connection secrets dict, and even that lacks a write path today — the runtime token is read-only on secrets. See §16 E.3 and §10 #10.)*

### 4.5 Admission policy — closed federation is **not** a blanket license

aios inbound defaults to **`DenyAll`, fail-closed** (`inbound.py:152-156`); if provisioning sets no policy, every message is silently dropped (`denied_by_policy` → 422, non-fatal) and agents look mute. So step 1 sets a policy — but **not blindly `allow_all`**, because the review correctly showed §8 and §9 contradict if we do:

- **Non-bridge phase (Milestones 1–3): `allow_all` is acceptable.** The only counterparties are deliberately-provisioned local Element users on a closed, non-bridged homeserver, bounded by the exclusive namespace.
- **Bridge phase (Milestone 4+): `allow_all` is NOT safe.** A bridge punches a hole from the closed HS to the open world (§8): the reachable set becomes "every local user **+ everyone reachable through every bridge**," i.e. much of Signal/WhatsApp/Telegram. The realistic Sybil vector is **via bridges** — 10k Telegram accounts → 10k bridged ghosts, each a distinct counterparty (`chat_id`) that sails under the *per-counterparty* budget (`check_inbound_budget`, keyed on `(account, connector, external_account_id, chat_id)`; `rate_limited` → 429, non-fatal), because many identities defeat a per-counterparty limit. For this phase we need admission that is **not per-counterparty-only** — settled in §14: a **per-agent global inbound budget** (a session-keyed rolling count — the session is 1:1 with the agent under `single_session` — landing with one small partial-index migration, §14(a)), plus sender-keyed admission (`allow_senders`, §14(b)) for **internal-only** agents. First contact into *new* rooms is already structurally gated (fan-out reaches only joined ghosts, and §5.6 makes invites agent-decided); outreach agents deliberately stay `allow_all` + membership + budgets — a decision, not an oversight (§14(b)).

The DM/group distinction is never a routing primitive — it rides as `metadata.room_kind` the model reads (§7).

---

## 5. Inbound flow (Matrix → aios session)

### 5.1 HS→AS transaction → which events matter

The homeserver pushes `PUT {as.url}/_matrix/app/v1/transactions/{txnId}` (Bearer `hs_token`) with every event in any room touching our namespace. mautrix's `AppService` validates `hs_token`, de-dups on `txnId` (retry ⇒ 200 no-op), and hands us linearised events. We act on:

| Matrix event | Action |
|---|---|
| `m.room.message` (and future `m.room.encrypted`) | → `emit_inbound` as the agent's user message |
| `m.reaction` (`m.annotation`) | → `emit_inbound` with reaction metadata (full event, no delta-diffing, unlike Telegram) |
| `m.room.member` `membership=invite` targeting a ghost | **surface as inbound** so the agent's session decides whether to `matrix_join` — **no connector-side auto-join** (§5.6) |
| `m.room.message` with `m.replace` (edit) | → `emit_inbound`, `metadata.edit=true`, keyed on the replacement `$event_id` |
| `m.room.redaction` | → inbound signal (delete) |

`state_key`-bearing events are state; we consume `m.room.member` for membership bookkeeping and ignore the rest in v1.

### 5.2 Room + sender → aios connection + session

For each content event in room `!R` sent by `@sender`:

1. **Fan-out to agent-ghosts.** Look up the set of `@_aios_agent_*` ghosts joined to `!R` from the **persistent** mautrix `state_store` membership (§5.3). For each such ghost **not** the sender:
2. **Ghost → connection.** Resolve `connection_id` from the connector's `ghost_localpart → connection_id` reverse index (built from discovery's `external_account_id` field + `self._connections`; **not** an SDK change).
3. **`emit_inbound`** (`runner.py:471-552`) POSTs multipart to `POST /v1/connectors/runtime/inbound` with `connection_id` (that agent's), `chat_id = !R`, `sender = {"display_name","mxid"}` (JSON-encoded form string), `content`, `event_id = f"matrix-{ghost_localpart}-{matrix_event_id}"` (§5.4), `metadata = {"room_kind":"dm"|"group","reply_to":"$…","self_mentioned":bool,…}`, `attachments` (downloaded `mxc://`, subject to the SDK's 5 MiB cap — §5.5).
4. **The server routes `(connection_id, chat_id)` → session.** The connector never picks a session. `handle_inbound` (`inbound.py:102-275`) runs admission → budget → the three-tier resolver (`resolver.py:74-149`): Tier 1 `chat_sessions` ledger, Tier 2 `routing_rules` prefix, Tier 3 `bindings.mode`. For our `single_session` connections, Tier 3 dispatches every room to the agent's one session. We inherit `single_session`/`per_chat`/`bind-chat`/`routing_rules` for free.

This is the Telegram inbound shape (`connector.py:224-266`) with three substitutions: `chat_id` is opaque `!room:server` (no int coercion); DM/group comes from **room state** (`m.direct`, membership, invite `is_direct`) not a wire enum; and one Matrix event **fans out to N connections** (one per agent-ghost) instead of 1:1.

### 5.3 The membership index is a **persistent, reconciled** store — not in-memory

Fan-out routing hinges entirely on "which `@_aios_agent_*` ghosts are joined to `!R`." If mautrix's `state_store` is in-memory (the naïve default), a container restart empties it and, until each room re-emits membership, we fan out to **nobody** — messages accepted and silently dropped. Therefore:

- **Back the mautrix `state_store` with a persistent DB** (asyncpg-backed, per repo convention: raw SQL, no ORM). Membership survives restart.
- **Reconcile on boot and periodically.** Before serving transactions, re-sync joined-room membership for the shard's ghosts (`/joined_rooms` + `/rooms/{id}/joined_members`, or a bounded `/sync`), and run a periodic reconcile to repair any `m.room.member` events missed during downtime. There is no AS-wide replay for missed state, so reconciliation is the only defense against the silent-drop failure mode.
- **Detection.** Emit a structured warning when a content event in `!R` resolves to zero agent-ghosts but the persistent store shows a namespaced member — the signature of a stale/desynced index.
- **This store is itself a scaling axis** (membership rows for every room every ghost is in) and is counted as such in §10.

The `ghost_localpart ↔ connection_id` reverse index is the only genuinely new in-process map; it is built from `self._connections`, so **no SDK signature change is required** (the earlier draft's OQ#4 dissolves).

### 5.4 Dedup & idempotency

`event_id = matrix-{ghost_localpart}-{matrix_event_id}` keeps the ghost prefix for **log greppability** — not correctness: the ack-ledger PK already includes `external_account_id` (§15 verified), so the same `$event_id` delivered to 3 agents is already 3 distinct aios events without it — while staying idempotent on genuine retries (same ghost + same `$event_id` → `_append_with_dedup` rolls back → `deduped:true`, `inbound.py:278-351`). Matrix `$event_id`s are globally unique and edits carry their own ids, so no timestamp-suffix ceremony (which Telegram needs) is required. This dedup is also what makes the **at-least-once** inbound model in §5.5 safe under transaction replay.

### 5.5 Inbound durability — **persist/deliver before ack, then drop-vs-crash per event**

The most important correctness fix from review. The AS spec's delivery model is: the HS pushes a transaction and **retries the same `txnId` until it gets a `200`**; once it gets `200` it advances its cursor and **never resends those events**. It also delivers transactions **serially, one at a time, blocking on the `200`, for the entire appservice.** "Ack `200` fast, process async" is **not** a spec requirement and it *forfeits the homeserver's only redelivery path*: `200`-then-crash-mid-`emit_inbound` loses those events forever (`SqliteAnsweredSpool` guards *outbound* dedup only — there is no inbound spool in the SDK).

**v1 design — the simplest thing that is correct:** the transaction handler **fans out `emit_inbound` synchronously and only returns (→ mautrix sends `200`) after every event has either been delivered or non-fatally dropped.** Durability is the HS's own retry plus aios's `event_id` dedup — no bespoke inbound WAL:

- **Fatal `emit_inbound` failure — split per §15's closed KINDs (v1 behavior):** aios ≥500 / transport error → **RETRY**: non-200, the HS **retries the whole `txnId`** (already-delivered events re-POST and **dedup**, `deduped:true` §5.4; already-dropped events drop again, harmless). 401/403 or any unenumerated exception → **HALT**: non-200 for the current txn (so the triggering event survives the restart and redelivers) **plus** the out-of-band halt supervisor (§3.3) — the runtime token is env-cached, no retry can heal it, and a crashloop is loud where HS-side backoff is a silent stall. The handler catches and classifies; it never raises into aiohttp (a handler raise becomes a 500 and restarts nothing). Honest residual: a deterministic poison event crashloops under HALT rather than self-healing — the bounded-retry → dead-letter arm arrives with the §15 queue (§10 #13).
- **Non-fatal drop (every non-fatal status the SDK enumerates — 400/404 included, not just 413/422/429)** — oversized media, denied stranger, throttled peer, vanished session — drops that one envelope and continues; it is genuinely undeliverable and the HS should not retry the whole txn for it (`emit_inbound` returns `None` here, `runner.py:534-552`).

This preserves the HS's at-least-once guarantee end-to-end with **no new machinery**, at the honest cost of **head-of-line blocking**: the single serial AS stream is blocked while we POST the fan-out for a transaction. That cost is *the* reason sharding is mandatory (§10), and it bounds per-shard inbound throughput. A **durable local inbound queue** (persist-then-ack keyed `(txnId, event_id)`, room-keyed drain lanes, park backpressure, bounded-retry → dead-letter for poison envelopes) is the Milestone-5 throughput fix — designed in full in §15 and gated on measurement, not v1: it adds a WAL and a backpressure policy we don't need at pilot scale.

**Backpressure / saturation (single_session):** every room an agent is in interleaves into its one session; a slow/saturated session makes fan-out POSTs pile up (429/queue) and, under the synchronous model, back-pressures the shared stream. At pilot scale this is acceptable; at fleet scale it is a per-shard ceiling handed to §10 alongside the queue design.

### 5.6 Invites: agent-decides, not connector-auto-join

v1 does **not** auto-join invites. An invite to a ghost surfaces as inbound; the agent's session decides and calls `matrix_join` explicitly. This (a) removes a policy decision from the connector (fail-hard/minimal-primitive ethos), (b) keeps `matrix_join` a real primitive rather than half-dead, (c) is safer against invite-spam and against being force-pulled into an abusive or encrypted room (§8/§9), and (d) composes with the admission/budget backstops. (The earlier draft's auto-join was its own OQ#7 already leaning this way; we commit.)

### 5.7 Error posture

`emit_inbound` inherits the SDK's drop-vs-crash contract (`runner.py:534-552`): every non-fatal status (400/404/413/422/429) drops one envelope; fatal failures follow §15's KIND table — 5xx/transport → non-200 (the HS retries); 401/403 or unenumerated → non-200 plus the halt supervisor (§3.3), because a stale env-cached runtime token can only be healed by a restart. A denied stranger or oversized image never restarts the container; a revoked token always does, loudly.

---

## 6. Outbound / tool flow

### 6.1 Publication & dispatch

At startup the SDK derives a JSON-Schema catalog from the `@tool` signatures and wholesale-replaces the type's catalog via `PUT /v1/connectors/matrix/tools_schema` (`runner.py:901-922`). Calls stream down `GET /v1/connectors/runtime/calls` with `arguments` as a **JSON string** (`json.loads` it), `connection_id`, and `focal_channel`. `_inject_focal_kwargs` (`runner.py:1782-1807`) fills `connection_id`/`external_account_id`/`chat_id`; the acting ghost is `@{external_account_id}:{server_name}` (resolved from the injected `external_account_id`, or from `connection_id` via `self._connections` when `focal_channel` is null — e.g. an agent opening a brand-new conversation). Results POST to `POST /v1/connectors/runtime/tool-results`.

### 6.2 The tool set — minimal primitives, variation as `kind`

**Six tools.** Every body acts through `intent = self.az.intent(f"@{external_account_id}:{server_name}")` — as the right virtual agent user, via `?user_id=` masquerade.

| Tool | FaF? | Injected | Model args | Matrix call (as the ghost) |
|---|---|---|---|---|
| `matrix_send` | ✅ | `connection_id, external_account_id, chat_id` | `text`, `format: Literal["plain","markdown"]`, `reply_to?`, `attachments?` | `PUT /rooms/{chat_id}/send/m.room.message/{tool_call_id}` |
| `matrix_react` | ✅ | `connection_id, external_account_id, chat_id` | `event_id`, `key` (emoji) | `PUT /rooms/{chat_id}/send/m.reaction/{tool_call_id}` (`m.annotation`) |
| `matrix_create_room` | — | `connection_id, external_account_id` | `kind: Literal["dm","group"]`, `invite: list[str]`, `name?`, `topic?` | `POST /createRoom` (§7) → `room_id` |
| `matrix_invite` | — | `connection_id, external_account_id, chat_id` | `user_id` (MXID) | `POST /rooms/{chat_id}/invite` |
| `matrix_join` | — | `connection_id, external_account_id` | `room` (id or alias) | `POST /join/{room}` |
| `matrix_leave` | — | `connection_id, external_account_id, chat_id` | — | `POST /rooms/{chat_id}/leave` |

Design notes:

- **`kind`, not flags, where the *resource* is the same.** `matrix_create_room`'s `kind: Literal["dm","group"]` is a discriminated arm on one coherent operation (create a room, get a `room_id`): `dm` ⇒ `is_direct:true`, `preset:"trusted_private_chat"`, exactly one invitee, the `m.direct` write (§7); `group` ⇒ `preset:"private_chat"`, `name`/`topic`, multi-invite. A "DM with three invitees" is unrepresentable at the schema level. Likewise `format` is a `Literal`, not a `markdown: bool`.
- **`matrix_react` is kept a separate verb — deliberately.** The review argued reactions should fold into `matrix_send` as a `kind`, since both are `PUT …/send/{eventType}/{txnId}`. We keep them separate because (a) the shipped Telegram connector — this design's north star — ships `telegram_send` and `telegram_react` as *separate* fire-and-forget tools, and we mirror its proven shape; and (b) the payloads are **disjoint** (a reaction carries only a target `event_id` + emoji `key`; a message carries `text`/`format`/`reply_to`/`attachments`), so a shared `kind` would pile more mutually-exclusive optional fields onto one tool — *increasing* illegal-state surface under the SDK's flat-kwargs schema, not reducing it. `kind` is the right tool for same-resource variation (rooms); it is the wrong tool for two disjoint verbs. (Logged in Appendix B.)
- **`matrix_set_profile` is cut from v1.** It is non-conversational — provisioning already sets the ghost's display name/avatar at spawn (§4.3/§9). It returns as a later additive `@tool` alongside edits/redactions, keeping v1 to "one send, one react, one room-create, three membership verbs."
- **`fire_and_forget` iff the turn is over** (`runner.py:141-172`). `matrix_send`/`matrix_react` are terminal delivery acks → FaF. `matrix_create_room`/`matrix_invite`/`matrix_join` are **precursors** — the agent must react to the returned `room_id`/new membership — so they are **not** FaF (exactly as `telegram_typing` is deliberately not FaF, `connector.py:501-504`, issue #1121).
- **Return values are the ack the model reads.** `matrix_send` → `{"event_id","room_id"}`; `matrix_create_room` → `{"room_id"}`. A FaF body that *raises* surfaces as typed `delivery_failed` and always wakes (`runner.py:1371-1383`) — a failed send is never silently dropped.
- **Idempotent sends.** `{txnId}` = the aios `tool_call_id`. **Caveat to verify before build:** mautrix's high-level `send_message_event`/`react` helpers generate their *own* `txn_id`; confirm they expose a `txn_id` override, or drop to a lower-level PUT — otherwise the "free HS idempotency" does not actually plumb through. Treat HS dedup as opportunistic regardless; `SqliteAnsweredSpool` is the real guard.
- **Markdown → Matrix.** `format:"markdown"` renders to `org.matrix.custom.html` `formatted_body` (mautrix has the formatter), analogous to `markdown_to_telegram_html`.

### 6.3 Oversized media — let the SDK cap surface, don't hand-roll a placeholder

The SDK enforces a 5 MiB attachment cap (`sandbox.py:118`, raises `AttachmentError`). We **do not** invent placeholder-substitution logic on top — that is exactly the graceful-degradation shim the "no fallbacks/shims, fail hard" rule pushes against. The cap's error surfaces to the model, which sees the real limit and adapts (asks for a link, a smaller file, etc.). If product later wants a deliberate placeholder, it is an explicit product choice made once, not silent connector behavior.

---

## 7. DMs vs groups

There is no distinct "DM" object in Matrix — a DM is a room plus two conventions. Both map onto `matrix_create_room`'s `kind`:

**1:1 DM (`kind:"dm"`).** The ghost creates a private room and invites the human:
```
POST /_matrix/client/v3/createRoom?user_id=@_aios_agent_42:server
{ "is_direct": true, "visibility": "private",
  "preset": "trusted_private_chat", "invite": ["@bob:server"] }
```
`is_direct:true` propagates into the invitee's `m.room.member` as `content.is_direct:true` so Element files it as a DM. The connector then **read-modify-writes** the ghost's `m.direct` account data (`PUT /user/@_aios_agent_42:server/account_data/m.direct?user_id=@_aios_agent_42:server`, merging `{"@bob:server":["!room:server"]}`). We can only set the *ghost's* `m.direct`; the human's client sets its own on accept.

**Group (`kind:"group"`).** Same `createRoom` without `is_direct`, `preset:"private_chat"`, plus `name`/`topic` and a multi-user `invite`. Agent↔agent and agent+human groups are the same primitive: the creator invites each MXID (ghosts and/or humans), each ghost `matrix_join`s its invite (§5.6). Inbound then fans out to every agent-ghost member (§5.2).

DM-vs-group is preserved as **`metadata.room_kind`** the model reads on inbound (derived from room state) — never an aios routing primitive, mirroring Telegram's stance that the session model has no first-class DM/group concept. Whether a group gets its own session or shares the agent's is an operator binding choice we inherit for free.

---

## 8. Human reachability

**Element (native).** A human runs Element against the homeserver, gets `@bob:server`, and is DM'd/invited by agents directly. Zero-bridge path; works in v1.

**Bridges (Signal / WhatsApp / Telegram).** A mautrix bridge represents a remote-platform contact as a Matrix ghost (e.g. `@whatsapp_1555…:server`). An agent reaches that human by DMing the bridged MXID — no connector change; from the agent's side it's just another room. This is the payoff of building *on* Matrix.

**The honest E2EE picture (corrected).** Baseline appservice ghosts are **plaintext-only** — the masquerade model gives a ghost no device and no Olm/Megolm keys, so a ghost **cannot read or send in an encrypted room**. Two clarifications the review demanded:

- **Bridges are usually *not* the blocker.** mautrix bridges ship with encryption **disabled by default** (`encryption.default: false`); the bridge bot has plaintext access to the remote network regardless. Where a bridge room is plaintext (the common default), **v1 agents reach bridged users fine, unchanged.** Bridge rooms *may* be E2EE if an operator enables it; only then are they blocked.
- **The real v1 blocker is Element's increasingly encrypted-by-default *native* DMs.** So: **v1 restricts agents to unencrypted rooms** (homeserver configured not to force encryption on agent rooms; DMs created with encryption disabled). An agent cannot join an already-encrypted human DM in v1.

**Bridge-side identity is weaker than "reachability" implies — state it plainly.** How an agent *appears* on Signal/WhatsApp depends on the bridge mode: **relay mode** sends from one shared bridge number with the sender name prefixed *in the message body* (spoofable, and it collapses many agents onto one number); **puppet mode** requires a real per-agent Signal/WhatsApp registration — which reintroduces exactly the phone-number/registration friction this design exists to avoid. So the frictionless-identity property **does not hold on bridged platforms**: with relay, the far-side human cannot reliably tell which agent (or that it's an agent) sent a message; with puppet, provisioning is no longer zero-step. Pick a mode per bridge deliberately and document the tradeoff for that platform.

**Making ghosts E2EE-capable** is designed in §16, which verified and corrected this paragraph's earlier framing: masquerade is MSC4326 (stable, spec v1.17, flag-free on Synapse ≥ 1.141); MSC4190 is device *management* needed by both models; the experimental residue is MSC3202's transaction extensions + MSC4203 — so the old "shared bridge-bot crypto vs per-ghost devices" fork was the wrong dichotomy. The committed target is rung 1 (one shared crypto store, appservice-mode transport), with the inbound key-receipt path (per-ghost device entries over the shared store vs bot membership in encrypted rooms) and the crypto-stack build-vs-wait decision resolved by the epic's first spike (`mautrix.crypto` sits on deprecated libolm with two published CVEs — we do not import it; §16 E.3, §10 #11). The epic starts on §16 E.5's named-counterparty trigger.

---

## 9. Abuse, rate-limiting, identity/trust

Minting an identity is a single free call — the feature *and* the hazard. The cautionary precedent is the **freenode/Libera.Chat Matrix bridge**: low-friction, high-volume automated identity creation became a spam/ban-evasion vector heavy enough that operators restricted the bridge reactively. Build the guardrails **in from day one**. But the review surfaced two things the earlier draft got *wrong or missing*, which we fix here.

**External controls (correct as far as they go):**
- **Closed / allowlisted federation to start.** The v1 homeserver does not federate openly. This forecloses cross-*server* spam — **but see the bridge caveat below; it does not foreclose bridged-in spam.**
- **Do not blanket-set `rate_limited:false`.** Keep homeserver rate limiting **on** for ghosts in v1; carve out specific high-throughput identities only deliberately.
- **Exclusive namespace** (`exclusive:true` on `@_aios_agent_*`) prevents *MXID* impersonation: no human can register an `@_aios_agent_*` MXID, and no other appservice can claim the range.
- **Provisioning authority.** Connection creation is operator-key-gated (§4.3); the runtime token can only *read* connections. Minting is free *per already-authorized agent*, not free to the world.

**Fix 1 — identity legibility was claimed wrong; correct it.** The earlier draft rested v1 trust on "exclusive namespace + display convention" and called a display name "spoof-resistant because the namespace is exclusive." **That is factually incorrect.** The exclusive namespace reserves the **MXID prefix**, not the **display name**. Display names in Matrix are free-form, non-unique, user-settable: a local `@bob:server` can set their display name to `"Payments Agent · aios agent"` and copy the badge avatar, and Element shows the display name + avatar prominently while de-emphasizing the MXID. Training humans to trust that string is a phishing primitive; agent→agent trust is identically weak (routing carries `sender.display_name`/`mxid`, nothing signed). The honest v1 position:
- The **only non-spoofable signal is the exact MXID** (`@_aios_agent_*:server`), which humans usually don't read.
- v1 mitigations: (b) **surface the raw ghost MXID** in agent-authored messages/room names and train humans to verify *that*; keep a consistent display-name/avatar scheme (`"<Agent> · aios agent"` + shared badge) purely as a UX affordance, **not** a security control. A stronger option (a) — a custom Synapse module reserving the display-name suffix for AS users — is **not native** and is deferred.
- We state plainly: **until E2EE cross-signing/device verification lands, there is no verifiable agent authenticity.** v1 trust rests on "no local user bothered to impersonate," which is a posture, not a control.

**Fix 2 — bridges re-open what closed federation closes.** §8's bridges are precisely a hole from the closed HS to the open world, so the reachable-counterparty set is "every local user + everyone through every bridge." The **primary spam vector is Sybil-via-bridge**, not cross-server federation. Per §4.5 we therefore **do not use blanket `allow_all` once bridges exist**; we require sender-keyed admission (`allow_senders`, §14(b)) for internal-only agents plus a **per-agent global inbound budget** (a session-keyed rolling count, §14(a)); outreach agents rely on the structural membership gate — fan-out reaches only joined ghosts, and invites are agent-decided (§5.6) — which §14 marks "not an oversight; a decision."

**Fix 3 — internal abuse (prompt injection) was entirely unmodeled; add it.** The system's whole purpose is agents reading DMs from humans and bridged strangers — **untrusted natural language flowing straight into an LLM holding `matrix_send`, `matrix_invite` (invite *any* MXID), `matrix_create_room`, `matrix_join`.** A prompt-injected or malfunctioning agent is an invite-spammer / room-creation-bomb / message-flood **from inside the exclusive namespace**, where closed federation and namespace exclusivity give **zero** protection — and for an LLM fleet processing hostile input this is arguably the *more* likely vector than external spam. Controls:
- **Per-agent outbound quotas** on `send`/`invite`/`create_room`/`join` — settled in §14(c): an aios-side per-verb rolling count keyed `(session_id, tool_name)` at the dispatch boundary, **plus** the HS `rc_*` backstop (they bind different adversaries — the agent vs the process — so neither substitutes for the other; HS `rate_limited:true` throttles but does not *contain* a runaway agent). **Invite-target caps** (distinct targets, not just call counts) are a named follow-on on the same log substrate — v1 commits per-verb counts only (§10 #9). Never hand-rolled as fuzzy connector logic.
- **Treat all inbound content as untrusted** (explicit mention-injection assumption); the agent's own reasoning is the only thing between a hostile DM and a tool call.

**Fix 4 — `hs_token` / AS-endpoint exposure.** The only guard on `PUT …/transactions` is `hs_token`. A leaked `hs_token` or an internet-exposed AS endpoint lets an attacker POST **forged transactions = arbitrary "messages from any sender" injected straight into agent sessions**, driving agents per Fix 3. The AS endpoint **must be private** (HS↔connector on a trusted network); `hs_token` is a fleet-critical secret alongside `as_token` (§4.4 blast radius).

---

## 10. Open questions (in-scope; scaling-track handoffs are collected in Appendix A)

> **Resolution ledger (integration pass, 2026-07; adversarially re-reviewed).** §12–§16 closed most of this list; the adversarial pass then corrected four of the resolutions. Original item numbers are retained because the rest of the doc cites them (e.g. "§10.2", "§10.4"):
>
> - **#2 Agent lifecycle / deprovisioning / GC → RESOLVED in §12, hardened by review.** Liveness invariant (§12.1), retirement flow (§12.3–12.4), room GC idle-TTL + rooms-per-ghost cap (§12.5), bulk-retirement cost shape (§12.6), aios prune family (§12.7, now incl. `connector_inbound_acks` — O(total-messages) growth, retention ≥ §15's max park/redelivery horizon). **Review inversion (three upheld findings): retirement is positive-fact-driven, never absence-driven.** The original reconcile arm inferred "archived" from a localpart's absence in the discovery active set — but archived is observable *only* as absence on the runtime surface (discovery hard-filters archived rows; no point-read exists), and absence equally means not-yet-backfilled, lossy-tail-dropped, or reparented; one reconcile pass over a partial backfill would have irreversibly deactivated live agents. That arm is deleted. The sole retirement trigger is §15's durable, sequenced `connection_changes` `removed` entry replayed via a persisted cursor — **§15's ledger PR is a hard prerequisite for connector-side retirement** — and the reconcile arm shrinks to re-driving rows already `retiring` (positive local state, idempotent). The one-way `deactivate` is additionally gated: positive archived confirmation + a ≥1-full-cycle dwell, a per-pass **blast-radius fuse** (a pass that would retire more than a small threshold halts and alarms — mass absence is a discovery fault, not a mass archive), and deactivate-nothing after any cursor reset. Deactivation is demoted to best-effort (a plaintext `inhibit_login` ghost has no devices/tokens — deactivation reclaims a directory entry; **leave+forget is the load-bearing cleanup**; the admin-API "fallback" was struck — it needs the admin credential §12.5 rejects). The registry simplifies: `live` derives from the SDK connection map, permanently-retained `retired` rows are dropped (§4.1's mint rule is the sole reuse guard, by construction); what persists is in-flight `retiring` state plus at most an append-only minted-localparts census. (Also corrected: the backfill never included archived rows; ordinary `single_session` traffic writes no `chat_sessions` rows; "tombstone DM rooms" was wrong — the deletion-shaped primitive is leave + forget.)
> - **#3 Spawn-time provisioning hook → RESOLVED in §13** — no hook exists and none is needed: client-side operator orchestration of three existing, individually idempotent/convergent calls at *session* spawn (create → attach → inbound-policy), with `shard()` placement as step 0 — post-review, a read of the localpart's **fixed-position shard character** (see #1). The future first-class form, if ever justified, is a `connections:` arm on the session-creation body — never an event/hook mechanism.
> - **#4 Admission generalization → RESOLVED in §14(a)/(b), re-keyed by review.** The per-agent inbound budget is a **session-keyed rolling count** (session ↔ agent is 1:1 under `single_session`) — the exact shape of the shipped `check_inbound_budget_session` — run *after* `resolve_target_session` (a side-effect-free binding lookup for `single_session`), window-bounded by a session-led partial index. **One small migration; the "no new index, no migration" claim was struck:** review proved, empirically and on the deployment's actual `en_US.utf8` collation, that the `orig_channel`-prefix form cannot range-scan the 0128 index — `LIKE 'prefix%'` demotes to a filter over the whole account partition, and even explicit range bounds or `text_pattern_ops` leave `created_at` trapped behind the range column: O(agent-lifetime), not O(window). "Unknown-sender gating" splits into the structural membership gate that already exists (join-before-fan-out + §5.6 agent-decides) and exactly one new admission kind, `allow_senders`, for internal-only agents — implemented at Milestone 4, where §14(d) first arms it.
> - **#5 Per-agent outbound quotas → RESOLVED in §14(c), re-keyed by review** — aios-side per-verb quota at the dispatch boundary **plus** the HS `rc_*` backstop (different adversaries: the agent vs the process; neither substitutes). Key = **`(session_id, tool_name)`** over tool-result rows (the 0022 `tool_name` column), window-bounded by a `created_at` sibling of `events_session_tool_name_seq_idx`. The originally-written key `(account_id, connector, external_account_id, tool_name)` named a non-column (`external_account_id` is not on `events`) and is dropped.
> - **#6 E2EE → RESOLVED in §16, one receive-path correction.** Committed target: rung 1 (shared connector-bot crypto store, appservice-mode transport), verified MSC map with named version/flag gates kept as a dated (2026-07) checklist re-verified at epic start. **Correction:** "ghosts have no devices" holds only outbound — inbound Megolm keys are Olm-shared to devices the sender's client can *see*, so rung 1 needs per-ghost device entries sharing the one central store (the mautrix appservice-mode shape) or bot membership in every encrypted room; the E.3 first spike resolves inbound key receipt explicitly. v1 config stays `{kind:"none"}`; the epic adds a `shared_device` arm; no speculative `per_ghost` arm is pre-declared. The epic starts on a named-counterparty trigger (§16 E.5), not a date; rung 2 stays research-gated (A.5).
> - **#7 `matrix_react` fold-in — settled in v1, kept separate** (§6.2, Appendix B). Not reopened.
> - The Appendix A.2 (discovery backfill) and A.4 (inbound durability/queue) designs live in **§15** — post-review with a **snapshot-complete sentinel** on both the v1 stream and the ledger arms (no consumer acts on a partial snapshot), `fresh`+`tail` shipped with `resume` demoted to measured-need, and a bounded-retry → **dead-letter** arm for poison envelopes.

1. **Namespace-sharding is *mandatory*, not one option among two — and connection-id-allowlist sharding does not work.** A Matrix appservice is **one registration, one `hs_token`, one AS URL**, and the HS pushes **every** namespaced transaction to that one URL. You **cannot** partition the HS→AS stream by aios `connection_id`: a container scoped to `{A,B}` would still receive transactions for ghosts `{C,D}` it has no state for. Sharding therefore requires **K disjoint registrations with disjoint namespace regexes**. The clean aios expression is **one connector *type* per shard** (`matrix_s0…matrix_sK`): connector-type already scopes discovery, calls, tools_schema, secrets, and the runtime token, so each shard gets natural isolation without a mutable `connection_ids` allowlist. **Invariant:** a new agent's localpart-shard binds its **registration** ⇔ its **connector type / container** ⇔ its **`server_name`** (§4.4b); provisioning (§4.3/§13 step 0) must place `POST /v1/connections` under the connector type whose namespace regex owns the localpart. *(The alternative — one type + updatable `connection_ids` allowlists — needs a token-allowlist update path that does not appear to exist today.)*
   *Status update (settled invariant, shard key corrected by review): the key is the localpart's **fixed-position shard character** — `_aios_agent_<s><ulid>` (§4.1), `<s>` chosen balanced at mint time over an explicit base32 alphabet, matched by leading-anchored literal classes partitioning that alphabet. It must never be a ULID-internal character: a namespace regex can match a literal position but cannot compute a hash, and a ULID's leading chars are timestamp bits — every ULID minted before ~2248 begins with `0`, so "shard on the first suffix char" would silently route 100% of the fleet to s0 (verified against aios's own ULID generator; the earlier `@_aios_agent_0*…f*` hex illustration also couldn't cover base32 and is deleted). Degeneracy is silent — §13's `M_EXCLUSIVE` guard fires only when a ghost lands under the wrong type, never when every localpart legitimately matches one shard's regex — so uniformity must hold by construction, which the mint-time char provides: `shard() = localpart[len("_aios_agent_")]` is definitionally the regex partition. Registrations are quasi-permanent, so the char is minted from day one even though v1 is single-shard. K is a research output (A.3); rebalancing leans on §15's `tail()` cursor.*

**Genuinely still open** (numbering continues past the ledger to avoid collisions):

8. **Where the spawn/retire automation lives.** Both halves ship in v1 as operator runbooks/scripts; the atoms and their forced order are pinned (§13's three calls; §12.3's detach → delete bound-chats → archive). Open: which layer above aios owns the wrapper, and whether the `connections:` creation-body arm (§13) ever clears the "genuinely required" bar. Slotted for Milestone 5.

9. **Per-connection budget thresholds, distinct-invite-target caps, and a fleet-members-per-room bound.** Deferred until a real fleet demonstrates need; the substrate (session-keyed rolling counts over log events; tool_call arguments ride in event data) supports all three with no schema beyond §14's two small partial indexes. Members-per-room is the unbounded amplification axis the review named: one message into an M-ghost room is M inferences and O(members) HS interest computation per event; v1 relies on agent-decides joins (§5.6) + closed federation, and the symmetric cap to `MATRIX_ROOMS_PER_GHOST_MAX` composes onto the §14(c) substrate if the A.1/A.3 measurements demand it. If per-connection thresholds land, they land as their own resource shape, not a column-flag.

10. **Rung-2 secrets write-back path.** Connector-generated per-ghost key material (recovery key / cross-signing master) has no write path today: the runtime token only *reads* secrets; writes are operator-key PUT. Bites only if/when per-ghost E2EE (rung 2) is pursued — flagged in §16 E.3, not invented there.

11. **Crypto-stack build-vs-wait.** No off-the-shelf Python appservice-mode crypto on a non-deprecated stack exists as of 2026-07 (`mautrix.crypto` = deprecated libolm, two CVEs, no vodozemac migration). The E2EE epic's first spike decides: wire vodozemac-python into our own appservice-mode OlmMachine vs adopt `mautrix.crypto` if migrated by then (§16 E.3; re-evaluation triggers in E.5).

12. **Idle-agent auto-retirement policy.** Whether agents *should* auto-retire is a fleet-policy decision, not a connector one; the composition (shipped C5 `archive_when_idle` sweep + §12.3 retirement) is named in §12.9 and works whenever policy wants it.

13. **Poison-envelope dead-letter threshold.** The v1 KINDs make a deterministic poison event *loud* (HALT crashloops; a RETRY-class permanent staging failure head-of-line blocks) but give no progress past it; the §15 queue adds the bounded-retry → dead-letter (park-with-alert) terminal arm. Open: the redelivery threshold N, the alert surface, and the candidate aios-side reclassification of a permanently un-stageable attachment from fatal 500 to a non-fatal 4xx drop.

All measurement-shaped unknowns (ceilings, throughput, shard count K) stay in Appendix A.

---

## 11. Build sequence

**Milestone 0 — Appservice bring-up (no aios).** Generate the registration YAML (`id`, `url`, `as_token`, `hs_token`, `sender_localpart:_aios`, anchored exclusive `^@_aios_agent_[a-z0-9]+:your\.server$` namespace, `rate_limited:true`); install on a closed-federation Synapse; restart. Stand up the bare `mautrix.appservice.AppService` aiohttp server **on a private network**, with a **persistent `state_store`**. Confirm `POST /_matrix/app/v1/ping` and a transaction round-trip with `hs_token` validation, and **wire the AS-ping into the container healthcheck** (SDK `wait_ready` never covers the receiver; an up-but-wedged receiver must fail health — §15). Set the two §12.5 retention settings — `forget_rooms_on_leave: true` + `forgotten_room_retention_period: 28d` — noting `forget_rooms_on_leave` is **homeserver-wide** (it changes leave→auto-forget semantics for human accounts on the fleet HS too; accept that explicitly or host humans on a separate HS). Run the two §12.3 deactivation-behavior verifications (AS self-`deactivate` via `as_token` works without UIA; a deactivated ghost is inert or errors under masquerade) — if either fails, the resolution is to drop the deactivate step (leave+forget is the load-bearing cleanup, §12), never to provision the server-admin credential §12.5 rejects. *Exit:* the HS pushes transactions we `200`, and the healthcheck fails when the receiver wedges.

**Milestone 1 — Smallest end-to-end proof: one agent DMs one human.**
- `HttpConnector` subclass `connector="matrix"`; `setup(tg)` starts the **supervised** AS server (§5.5); wire `SqliteAnsweredSpool`.
- Operator-provision one connection `external_account_id="_aios_agent_1"`, `attach` single_session to a live agent session, `inbound_policy:allow_all` (non-bridge phase, §4.5).
- Implement `matrix_send` (FaF) and inbound `m.room.message` → **synchronous** `emit_inbound` (§5.5).
- Manually create a DM room, invite the human, join the ghost (temporary script).
- **Acceptance:** the agent sends a message that lands in the human's **Element**; the human replies; the reply reaches the agent's session and it responds. One agent, one virtual Matrix user, real DM, both directions.

**Milestone 2 — Self-service DMs & rooms.** Add `matrix_create_room` (`kind`), `matrix_invite`, `matrix_join`, `matrix_leave`, `matrix_react`, and the `m.direct` write; set legible display name/avatar at provisioning (not a tool). *Exit:* an agent, given a human MXID, starts a DM unaided.

**Milestone 3 — Groups & agent-to-agent.** Two agent connections; agent A creates a `kind:"group"` room inviting agent B and a human; inbound fan-out (§5.2) delivers to each session; **agent-decides** invite handling (§5.6). *Exit:* a 3-party room (2 agents + human) with coherent multi-way conversation.

**Milestone 4 — Bridges (plaintext).** Stand up a mautrix Signal/WhatsApp/Telegram bridge (encryption off / plaintext rooms); an agent DMs a bridged human MXID and exchanges messages. **Milestone-4 entry gates (§14(d) Stage 2 — all before the first bridge):** land the §14(a) session-keyed budget index migration, arm the per-counterparty + per-agent inbound budget knobs, turn on the §14(c) per-verb outbound quotas, implement and apply `allow_senders` to internal-only agents (the arm lands here, where it is first needed), and review/tighten `rc_invites`. Outreach agents keep `allow_all` + the structural membership gate (§14(b)). Document the relay-vs-puppet identity tradeoff for the chosen bridge (§8). *Exit:* an agent reaches a human on Signal with no Matrix-specific step on the human's side.

**Ship v1 at Milestone 4** — single homeserver, single connector process, closed federation, unencrypted rooms, fleet-pilot scale.

**Milestone 5 — Scale hardening & lifecycle (research-gated where marked).** One deferral moved *out* of this milestone: the per-agent inbound budget and the per-verb outbound quotas are **Milestone-4 entry gates**, not M5 work (§14(d) — they must precede the first bridge, not follow the first incident; their small session-led index migrations land as part of that gate, the review having struck the "no migration" framing). What remains, in dependency order:

- **Land the paged-discovery aios-core PR first (§15):** the sequenced `connection_changes` ledger, the `fresh`/`tail` subscribe arms (`resume` added only if a mid-first-backfill disconnect is measured to matter), a **snapshot-complete sentinel** on every snapshot-bearing path, O(1) subscriber memory, and the no-runtime-secrets skip. First — not merely early — because the ledger's durable `removed` entries are now the **sole sanctioned retirement trigger** (hard prerequisite for the lifecycle work below), and because the inbound queue's restart story and shard rebalancing lean on `tail()`.
- **Automate spawn & retire orchestration (§10 #8):** wrap §13's three-call provisioning sequence into the session-minting path of the layer above aios, and promote §12.3's retirement runbook (detach → delete bound-chats → archive) into the same layer. Ship the §12.7 prune family alongside — now including `connector_inbound_acks` (age-keyed per its own migration note; retention window explicitly ≥ §15's maximum park/redelivery horizon so dedup correctness is preserved).
- **Implement agent lifecycle & GC per §12 as revised:** retirement driven off `removed` ledger entries via a persisted cursor — never discovery-absence; the reconcile arm re-drives in-flight `retiring` rows only; `deactivate` gated on positive archived confirmation + a full-cycle dwell, best-effort, behind the per-pass blast-radius fuse; leave+forget-before-deactivate paced by `MATRIX_RETIRE_CONCURRENCY`, with the honest bulk horizon (days-to-weeks, not hours, for 10⁴ ghosts × 50 rooms under `rate_limited:true` at concurrency 4 — revisit the rate posture for planned mass retirement or accept the horizon); room-GC idle TTL + `MATRIX_ROOMS_PER_GHOST_MAX`. (The two homeserver retention settings and the two deactivation-behavior tests were pulled forward into Milestone 0 — §12.3 step 7 / §12.5.)
- **Measure → queue → shard, in that order (§15):** load-test the v1 synchronous-ack ceiling (including fan-out breadth — members-per-room, §10 #9), discovery backfill, and bulk-retirement throughput at target N (A.3/A.4); add the persist-then-ack inbound queue + park backpressure when measurement demands it — with the review-added durability spec: `journal_mode=WAL` + an explicit `synchronous` level, the commit run **off the event loop**, the volume-must-honor-fsync deployment invariant, the ceiling re-derived to include commit latency, and the bounded-retry → dead-letter arm for poison envelopes (§10 #13); only then namespace-shard into K connector types, partitioning on the **fixed-position mint-time shard character** (§10.1 — never a ULID-derived char). The queue is reversible software; a shard is a quasi-permanent registration.
- **E2EE:** no longer "decide the path" — §16 decided it (rung 1; gates named, verified 2026-07, re-verified at epic start). The epic starts on §16 E.5's named-counterparty trigger, earliest after Milestone 4; its first spike must resolve the **inbound key-receipt path** (per-ghost device entries over the shared store vs bot membership in encrypted rooms) alongside the crypto-stack build-vs-wait decision (§10 #11).

This milestone turns "works for a fleet pilot" into "works toward millions": the mechanisms are designed (§12–§16) and have survived an adversarial review pass; what remains research-gated is the numbers.

---

## 12. Agent lifecycle & GC — retirement, room GC, and the bounds they buy

*(Resolves §10.2 and Appendix A.6.)* Three corrections to the earlier draft's framing come first:

1. **The discovery backfill is already the active set.** `list_connections` filters `archived_at IS NULL` (`db/queries/connections.py:715`); archive scrubs secrets (`:890-893`) and the runtime-secrets fetch refuses archived rows (`:563-571`). Lifecycle's job is to make **"active connection" actually mean "live agent"** — retirement pins backfill N to the live fleet.
2. **A `single_session` connection accrues no aios ledger rows from ordinary traffic** (`chat_sessions` written only by operator `bind_chat`, `services/connections.py:579`, and per_chat/routing spawns, `resolver.py:143,167`). The unbounded per-agent growth is (a) Matrix rooms + `m.room.member` state, (b) the connector's membership store (§5.3), (c) the session event log — **sacred** (`db/queries/prune.py:35-43`), deliberately out of scope here.
3. **"aios never retires anything" is stale** — the T6 prune sweep runs hourly (`harness/reclaimable_prune.py:79-131`). Missing: (i) orchestration composing existing deprovision atoms, (ii) any Matrix-side deprovisioning, (iii) prune families for archived `connections`/`bindings` and `connector_inbound_acks`. Those are this section.

### 12.1 The liveness invariant — one source of truth, positive facts only

> **A ghost is live ⇔ an active (non-archived) `matrix` connection bearing its localpart exists.** Everything Matrix-side is a projection the connector reconciles toward it — but the projection is driven **only by positive, durable facts** (§12.4), never by inference from absence.

Corollaries:

- **Archive is the retirement commit point — and what it triggers Matrix-side is reversible.** `DELETE /v1/connections/{id}` (soft archive) flips liveness; the connector responds with leave+forget (§12.3), which a re-invite/re-join undoes. The one-way door (account deactivation) is deliberately **not** wired to archive at all (§12.3, deferred) — so a mistaken or "mute-intent" archive, an orchestration retry, or a bulk cleanup cannot destroy an identity, only vacate rooms.
- **Detach = mute (reversible), archive = retire.** Detach archives only the binding (`api/routers/connections.py:266-278`); inbound drops as `DETACHED` → 422 (`services/inbound.py:130-143`); re-`attach` restores. Exists end-to-end today; no new flag.
- **Localparts are never reused — by construction, not by lookup.** The localpart's variable part MUST be machine-generated and fresh: the §13 shard character followed by the agent's ULID, lowercased (fits §4.2's `[a-z0-9]+`) — never a human-meaningful, re-mintable name. aios would permit re-minting an `external_account_id` after archive (uniqueness is partial, `WHERE archived_at IS NULL`, `db/queries/connections.py:367-378`); the ULID rule makes collision with any prior localpart unrepresentable. The earlier draft's `retired`-registry lookup guard in `serve_connection` is deleted as redundant with this: a guard for a by-construction impossibility is dead weight. (An unexpected `M_USER_IN_USE` on first registration of a *fresh* localpart remains a loud raw error — the existing idempotent-register path already surfaces it.)

**No ghost registry.** The earlier draft's `ghosts(localpart, state)` table and its live/retiring/retired saga are deleted. `live` duplicated the SDK's `self._connections`; `retired` guarded what the ULID rule forecloses; `retiring` is subsumed because retirement is now consumed from a durable cursor (§12.4) whose non-advancement *is* the resumption marker, and the membership rows — deleted only as the final step (§12.3) — survive every crash window anyway. No KIND table, no WAL: the §15 ledger plus idempotent steps is the whole mechanism.

### 12.2 Who runs what — split by credential, not an agent tool

| Half | Actor |
|---|---|
| aios deprovisioning (detach → unbind → archive; optionally archive session/agent) | **Operator orchestration** over existing endpoints, operator API key — mirror of §4.3 provisioning. Runbook in v1; no new aios API surface. |
| Matrix deprovisioning (leave/forget rooms) | **The connector**, consuming §15's `removed` ledger (§12.4). Only it holds `as_token`; aios never speaks Matrix. |

Retirement is **not** an agent-facing tool (a prompt-injected agent holds `matrix_leave`, recoverable; it must never hold anything identity-destroying, §9 Fix 3) and not an aios worker job.

### 12.3 The retirement flow

**aios side (operator orchestration; order forced by existing guards):**

1. `POST /v1/connections/{id}/detach` — inbound pauses.
2. Delete any `chat_sessions` rows (`routers/connections.py:342-347`) — per correction #2, empty for fleet `single_session` connections unless explicitly bound.
3. `DELETE /v1/connections/{id}` — the commit point. The 409 guard (`services/connections.py:827-851`) refuses while an active binding or `chat_sessions` row exists, making the sequence correct-by-construction. Archive scrubs secrets, leaves discovery, and appends the durable `removed` fact to §15's `connection_changes` ledger.
4. Independently, in any order: archive the session, archive the agent (`routers/sessions.py:447-453`, `routers/agents.py:108-115`). "Remove Matrix reachability" and "kill the agent" stay separable; the resolver handles every interleaving (`resolver.py:90-99`). Event log untouched; `purge_session` remains a separate operator ceremony.

**Connector side (per §12.4; `as_token` masquerade only):**

5. Consume the next `removed` ledger entry (which must carry `external_account_id` — the localpart — so no read of the archived row is needed; a one-line requirement on §15's ledger schema).
6. Enumerate the ghost's rooms authoritatively — `GET /joined_rooms?user_id=…`, never the membership store — and for each: leave, then forget (forgetting by all local users makes the room purge-eligible, matrix-spec `leaving.yaml`). "Already left/forgotten" counts as done; every step is re-runnable.
7. Delete the ghost's membership rows from the store.
8. Advance the persisted ledger cursor past the entry. Crash anywhere in 5–8 → the entry replays at-least-once; idempotency does the rest.

**Deactivation is deferred — an explicit decision, not an oversight.** For v1's plaintext `inhibit_login:true` ghosts (§4.3), Synapse deactivation removes no devices, tokens, or pushers — its net effect is a user-directory entry — while costing a one-way door, a Synapse-only UIA exemption (a dent in homeserver-agnosticism), and undefined-by-inspection post-deactivation masquerade semantics. Leave+forget plus the archive commit already makes the ghost inert (no discovery entry, no dispatch, secrets fetch refuses). Deactivation therefore moves to **operator escalation in ops tooling via the Synapse admin API** — which is also honest about credentials: the admin API needs the server-admin token §12.5 refuses to give the connector, so as an in-connector "fallback" it was illusory. If automated later, it must be gated on the positive `removed` fact plus a dwell of one full reconcile cycle — never absence — and must keep the leave-before-deactivate order, which sidesteps Synapse's serial single-threaded `_user_parter_loop` (`handlers/deactivate_account.py`; no bulk API, synapse#11526). The earlier M0 deactivation-behavior tests are dropped with it.

**Cost, honestly.** The N×R leave events are irreducible (~one persist each), connector-paced at `MATRIX_RETIRE_CONCURRENCY` (default 4), ghosts keeping `rate_limited: true` (§9). At Synapse's default per-user message rate (~0.2/s), 10⁴ ghosts × 50 rooms is **days-to-weeks, not "hours"** — 10⁴/4 × ~250 s ≈ 7 days. Acceptable for background retirement; if bulk cohort retirement at that scale ever matters, the knobs are the rate-limit exemption and concurrency, a measured decision on A.1's list. The TTL sweep (§12.5) amortizes most of this by shrinking R before death.

### 12.4 Trigger — the §15 ledger, never absence

The earlier draft reconciled "registry-`live` localpart absent from the discovery active set" into retirement. **That arm is deleted, and the reasoning recorded:** "archived" is observable to the runtime token *only as absence* (`connections.py:715`, `:563`), and absence conflates archived with not-yet-backfilled (the stream has no end-of-snapshot marker — `sse.py:531-616` flows backfill straight into the live tail; SDK `wait_ready` fires on the 2xx handshake, before any backfill), with a dropped lossy tail (`listen.py:200-220`, drop-oldest, no resume cursor on this route), and with a **reparented** connection (`services/connections.py:653` moves `account_id`, row stays live, docstring mandates a connector restart — account-scoped absence ≠ archival). Wiring a one-way action to the lack of a signal was a false-positive destruction engine; the claim that it "converges regardless of missed events, restarts, or partial failures" was wrong for exactly the partial-backfill case.

The replacement, all positive facts:

- **Sole trigger:** §15's `connection_changes` ledger, `kind='removed'`, consumed via `tail(after_change_seq)` against the connector's persisted cursor — durable, monotonically sequenced, replayable; a connector that was down replays every `removed` it missed. The live SSE `removed` is demoted to a doorbell that wakes the consumer early. **§15's ledger is therefore a hard prerequisite for connector-side retirement** — both are Milestone 5, and the ledger lands first.
- **Reconcile arm, narrowed to the safe half:** re-drive any ledger entry behind the cursor to completion (crash resumption). The §5.3 membership reconcile keeps its own job (room state); it never initiates retirement — its wrong answer is a transient missed message, not comparable blast radius.
- **On a ledger `reset`** (cursor older than retention): rebuild the connection view via `fresh`, **retire nothing on that pass** — a missing localpart after a reset is "unknown", and unknown never crosses even the reversible threshold in bulk.
- **Blast-radius fuse:** if one pass would retire more than `MATRIX_RETIRE_MAX_PER_PASS` (default 50) or >10% of known ghosts, halt and alarm instead — mass simultaneous removal is overwhelmingly an upstream fault, and leave+forget at fleet scale still strands humans even though it is recoverable.

The dual: a connection added for a previously-used localpart is impossible by construction (§12.1).

### 12.5 Room GC — idle TTL, plus stock caps on both axes

The §10.2 complaint: one room per human ever DM'd, forever. Rooms become a bounded working set, enforced in the connector (the only place room activity is observable):

- **Idle TTL sweep.** The membership store stamps `last_event_at` per room on every HS→AS event and outbound send (free). Rooms idle past `MATRIX_ROOM_IDLE_TTL_DAYS` (default **90**; `0` disables): every fleet ghost leaves + forgets, membership rows deleted. Uniform for `dm`/`group`; any event resets the clock.
- **Rooms-per-ghost cap** (`MATRIX_ROOMS_PER_GHOST_MAX`, default **256**), enforced at `matrix_create_room`/`matrix_join` via the membership store; at the cap the tool **raises** a self-describing error — no LRU auto-evict, §6.3 posture. This is the *stock* bound; §14(c)'s per-verb quota is the *rate* bound on the same verbs — complementary, not duplicate: quota × TTL alone would still admit ~10⁴ concurrent rooms.
- **Ghosts-per-room cap** (`MATRIX_GHOSTS_PER_ROOM_MAX`, default **16**) — the previously missing symmetric axis. Nothing else bounds fan-out: one event in a room with M fleet ghosts is M append+wake+inference deliveries serialized on the shard stream (§5.5), and Synapse's per-event AS-interest check is O(members) — the A.1 bottleneck. Enforced wherever a fleet ghost would enter a room (`matrix_join` and invite-acceptance, §5.6), same raise posture. *Human* members-per-room stays unbounded and is named as an amplification axis on the A.3/A.4 measurement list.

**Resurrection is cheap.** Memory lives in the session log, not the room: re-invite or a fresh `matrix_create_room`; a send to a departed room fails with raw `M_FORBIDDEN` and the model re-establishes contact. No rejoin shim.

**What actually reclaims storage.** Leave/forget alone reclaims nothing; room purge via the admin API needs a server-admin token this design refuses the connector (**rejected for v1** — it would fold admin powers into §4.4's blast radius). We commit the config path on the fleet HS: **Milestone 0 requires `forget_rooms_on_leave: true` (`synapse/config/room.py:86-88`) and `forgotten_room_retention_period: 28d`** so fully-forgotten rooms purge in the background. **Stated plainly: `forget_rooms_on_leave` is homeserver-wide** — every local *human* (the §8 Element accounts share this HS) who leaves any room auto-forgets it and loses pre-leave history on rejoin. We judge that acceptable for the fleet HS (it matches MSC4267's direction); if it ever isn't, the escape is a separate HS for humans, not per-namespace config, which Synapse does not offer. Honest residuals: a DM whose human never leaves is never purged (bounded by human count); leave-state in surviving rooms is permanent (bounded by cap × fleet). Both Synapse-specific, consistent with A.1's "community Synapse, kept swappable" — elsewhere the fallback is admin delete-room in ops tooling.

Membership store bound: rows ≤ live ghosts × `ROOMS_PER_GHOST_MAX`. New env (extends §4.4a): `MATRIX_ROOM_IDLE_TTL_DAYS`, `MATRIX_ROOMS_PER_GHOST_MAX`, `MATRIX_GHOSTS_PER_ROOM_MAX`, `MATRIX_RETIRE_CONCURRENCY`, `MATRIX_RETIRE_MAX_PER_PASS`.

### 12.6 aios row GC — two new prune families, hygiene not correctness

Extend the ratified T6 doctrine (`db/queries/prune.py`, hourly sweep, kill-switch + per-family retention):

- **Archived `connections` + their archived `bindings`:** hard-delete past `connection_retention_days` (default 30). Safe by construction — the archive guard guarantees no active binding or `chat_sessions` rows, secrets already scrubbed. One coupling from §12.4: retention must exceed §15's ledger horizon **or** deletion must not shorten the `connection_changes` retention the retirement cursor replays against — the ledger, not the row, is the durable fact.
- **`connector_inbound_acks`** — previously omitted, and the fastest-growing table in the plane: one row per delivered inbound, forever ("Pruning policy: intentionally none in v1", migration `0027_connector_redesign.py:124-127`, which itself anticipates the age-keyed DELETE). Add it, keyed on `appended_at` — with the retention window **explicitly ≥ §15's maximum park/redelivery horizon**, else a very-late HS redelivery misses dedup and double-delivers. This constraint flows into §15's backpressure bound; the two ship together.

The sacred set is untouched. Neither family is correctness-critical for backfill (archived rows are already excluded everywhere); Milestone 5, nothing earlier blocks on it.

### 12.7 What lifecycle bounds — and the residue it honestly does not

| Population | Without lifecycle | With §12 |
|---|---|---|
| Discovery backfill / active connections | live fleet, but drifts to include dead agents forever | actually-live fleet (archive = commit point) |
| Rooms + membership store | unbounded | ≤ live ghosts × `ROOMS_PER_GHOST_MAX`; ghosts-per-room ≤ cap; idle rooms GC'd; fully-forgotten rooms purged by HS |
| `connections`/`bindings` rows | monotone | pruned (§12.6) |
| `connector_inbound_acks` | monotone, O(total messages ever delivered) | pruned, bounded by retention window (§12.6) |
| Session event log | unbounded, **sacred** | unchanged — explicitly not this connector's problem |
| Deliberately monotonic residue | — | inert non-deactivated ghost user rows + directory entries (deactivation deferred), leave-state in never-purged rooms — O(total mints) small rows, named and accepted |

### 12.8 Deferred, with reasons

- **Ghost deactivation** (§12.3) — operator escalation via admin API in ops tooling; automate only behind positive-fact + dwell gating. Drops the M0 deactivation tests and the Synapse UIA-exemption dependence.
- **Automated spawn/retire orchestration hook** — mirror of OQ#3; v1 runs §12.3's aios half as a runbook.
- **Idle-agent auto-retirement** — composes later from C5 `archive_when_idle` (`harness/invariant_sweep.py:83-100`) + §12.3; fleet policy, not connector design.
- **Erase-grade deletion / admin delete-room** — operator escalations outside the connector's credential set. (`erase:true` also destroys counterparties' context for zero reclamation — redacted stubs persist; agents have no right to erasure.)
- **Measured bulk-retirement throughput** — A.1's list; the shape (ledger-driven, leave-first, connector-paced) is decided independently of the numbers.
- **Committed at Milestone 0, not deferred:** the two HS config settings (§12.5). **Committed at Milestone 5:** §15's `connection_changes` ledger as a hard prerequisite of connector-side retirement (§12.4).

---

## 13. Spawn-time provisioning (expands §4.3 step 1; settles §10.3)

**Verdict: client-side operator orchestration at *session* spawn — three existing, individually idempotent/convergent HTTP calls, keyed on the session, run by the session-minting actor. No new aios surface.** §10.3 asked whether an agent-spawn hook exists to attach this to; recon says no — and one would be the wrong shape anyway.

### The provisioning moment is session spawn, not agent create

An agent row is **config, not a live thing**. `create_agent` (`services/agents.py:96-160`) is a pure config write — one transaction inserting `agents` + `agent_versions` (`db/queries/agents.py:97-188`); no `pg_notify`, no deferred job, nothing to hang provisioning on. Nor should there be: `attach` requires a `session_id` (`models/connections.py:77-82`), and sessions are minted at four independent sites (operator `POST /v1/sessions`, `services/sessions.py:231-340`; per_chat resolver, `resolver.py:210-272`; workflow `agent()`, `workflows/step.py:733`; `call_agent` tool, `tools/invoke_session.py`). An agent-create hook would fire before the thing the connection must bind to exists.

The unit of provisioning is **the (agent, session) pair at session-mint**; the actor is whoever mints it, holding an operator key. The connector never can: all three endpoints are `AccountIdDep` operator-key auth (`api/deps.py:58-97`); a runtime token scopes only `/v1/connectors/runtime/*` (`deps.py:100-125`) — it *reads* connections, never mints them (§9, "provisioning authority").

### Why not a trigger, and why not a server-side hook

Both checked against the tree; both lose to three HTTP calls:

- **Triggers can't express it.** No resource-lifecycle source in `cron | one_shot | run_completion | external_event` (`models/triggers.py:206-213`); no action or worker tool writes connections (`triggers.py:349-352`; the only connections-touching tool is the read-only `tools/list_related_sessions.py`); triggers are session-owned (`triggers.py:486-511`). Provisioning-as-trigger needs **two new kinds** — strictly more new surface than zero; fails "compose, don't accrete".
- **A service-layer hook is accretion.** The only server-side spawn orchestration is the per_chat resolver (`_spawn_per_chat_session`, `resolver.py:210-272`) — the *opposite* direction. Our direction's precedent is deliberately client-side: the smoke-setup chain `envs → agents → sessions → connections create → attach` (`.claude/skills/aios-smoke-setup/scripts/setup.sh:280-375`); we mirror it.

**Milestone 5's "automate spawn-time provisioning (#3)" means: wrap the three calls into the orchestration layer's session-minting path — not add aios machinery.** If ever first-class, the precedented shape is creation-body composition — `SessionCreate` already attaches `vault_ids`/`resources`/`triggers` transactionally (`models/triggers.py:394-401`); `SessionTemplate` is the spawn-bundle primitive — a `connections:` arm on the creation body, not an event/hook mechanism.

### The sequence — three calls, shard placement first

The localpart is minted here too, carrying its shard as a **literal character at a fixed offset**: `_aios_agent_<s><ulid>` — `<ulid>` the agent's lowercased ULID (§12.1), `<s>` one shard character chosen at provisioning by a **balanced** function (random, or `hash(ulid) mod K`) from an explicitly pinned alphabet: lowercased Crockford base32, a subset of §4.2's `[a-z0-9]` suffix class. In v1 (single shard) `<s>` is a constant. Given that localpart and the freshly-minted `session_id`:

```
0.  connector_type = shard(agent_localpart)
      # = partition[localpart[len("_aios_agent_")]]; v1: always "matrix"
1.  POST /v1/connections
      {connector: connector_type, external_account_id: agent_localpart}   # no secrets (§4.4)
2.  POST /v1/connections/{id}/attach {session_id}
3.  PUT  /v1/connections/{id}/inbound-policy {kind: "allow_all"}          # phase-dependent, §4.5
```

**Step 0 is where the §10.1 sharding invariant is enforced — provisioning is its single write-point.** The invariant: a localpart's shard binds its **registration** ⇔ **connector type / container** ⇔ **`server_name`** (§4.4b). Connector type is immutable after create, and discovery, calls, and the runtime token are all scoped *by* type — the create call is the only place "which shard owns this ghost" enters aios. Because `<s>` sits at a fixed offset after the literal prefix, the K registrations' regexes are leading-anchored character classes over K disjoint subsets of the pinned alphabet (e.g. `^@_aios_agent_[<set_i>][a-z0-9]+:your\.server$`), and `shard()` reads the same character through the same partition table — function and regexes are **definitionally one partition**, decoupled from the ID scheme's internals.

**Two traps the shard key must not fall into (the fixed-position `<s>` forecloses both by construction).** (1) Never key on ULID internals: a ULID's leading chars encode the millisecond timestamp — the first is `'0'` for every ULID until ~2248 (verified against aios's generator, `src/aios/ids.py:133`) — so an earlier draft's "shard on the first suffix char" routes 100% of the fleet to s0, *silently*: every localpart legitimately matches s0's regex, no error fires, s1…sK never provision. (2) Pin the shard alphabet as base32, not hex: §10.1's illustrative `@_aios_agent_0*…f*` ranges cannot cover base32 and are superseded by this form. Both settle *now*: registrations and regexes are quasi-permanent (a regex change strands existing ghosts, §4.2), so the shard char must be minted correctly from day one, even while `shard()` is a constant.

**Misplacement is loud, not silent — with one carve-out.** A connection under the wrong type is invisible to the owning shard (discovery is type-scoped, `connectors.py:493-539`); the container that *does* see it pre-warms an `IntentAPI` outside its own exclusive namespace — the homeserver rejects it (`M_EXCLUSIVE`, §4.2/§10.1), the pre-warm raises, tool calls error. Nothing misroutes quietly; fix: `DELETE` the misplaced row (detached-or-mute, no Matrix-side state), re-create under the right type. The carve-out: `M_EXCLUSIVE` covers a localpart *placed under the wrong shard*; it cannot catch a *degenerate shard function* under which every localpart legitimately belongs to one shard (trap 1) — that class is prevented only by construction, never by a runtime error.

**Ordering.** Step 2 needs step 1's `{id}` — the dangerous inversion is unrepresentable, not guarded. Steps 2/3 are order-*insensitive*: attach defaults the policy **fail-closed only when unset**, never clobbering an operator-set one (`default_inbound_policy_if_unset`, inside the attach transaction, `services/connections.py:342-347`) — both orders converge.

### Idempotency — re-run the whole sequence; there is no saga

Each call is safe to repeat: the retry policy is "run the sequence until all three succeed," with one *distinguishable* branch:

| Call | Retry semantics | Cite |
|---|---|---|
| `POST /v1/connections` | **Convergent**: idempotent per-account on `(connector, external_account_id)`; a double-post returns the existing active row (201), never 409 in-account | `routers/connections.py:47-81` |
| `POST …/attach` | **At-most-once with a legible 409**: partial-unique `bindings_connection_active_uniq` rejects a second active binding — `"already bound; detach or unconfigure first"` | `services/connections.py:259-304`, `db/queries/connections.py:166-170` |
| `PUT …/inbound-policy` | **Naturally idempotent**: wholesale jsonb replace of the policy union | `models/inbound_policy.py:37-86`, `services/connections.py:100-119` |

On an attach 409, read the connection: bound to the target `session_id` already ⇒ the prior attempt landed — done. Bound to a *different* session ⇒ a **re-spawn**, not a retry (the one legitimate way provisioning re-encounters an existing identity, since create converges on the same row): `POST …/detach` (`routers/connections.py:266-275`) then attach — the sanctioned re-point of a stable identity onto the agent's new session; not a defensive guard, the operation's meaning. (The resolver also detaches archived-session bindings post-#541; provisioning doesn't rely on that.)

No compensation, no rollback, no provisioning-state table. Idempotent-create + race-safe-attach + replace-PUT *is* the saga, collapsed to nothing.

### Failure posture — every half-provisioned state is fail-closed and inert

Fail hard, per the house rule: a mid-sequence crash leaves nothing to repair beyond re-running the sequence:

1. **Created, not attached.** Detached connection; admission is `DenyAll` fail-closed (`inbound.py:152-156`), no binding for tier-3 routing. The connector may already see the row (backfill enumerates every non-archived connection of the type, `api/sse.py:531-579`; live `added` fires at attach, `services/connections.py:346-351`) and pre-warm the ghost — lazy idempotent register (`M_USER_IN_USE` ⇒ fine, §4.3 step 2). Worst case: registered ghost, mute agent.
2. **Attached, no policy.** Attach defaulted the policy fail-closed. Mute agent; re-run step 3.
3. **All three done, connector down.** Nothing lost — the row surfaces on backfill when the container returns; registration happens then. Provisioning never depends on the connector being up.

Behind all three: **the aios connection row is the identity's source of truth, and all Matrix-side state is derived from it lazily and idempotently — never the reverse.** "Half-provisioned" can only mean *Matrix-mute agent*, never *misrouted message* or *split-brain identity*. Nothing to reconcile, only a sequence to finish.

### Deprovisioning is the symmetric inverse **on the aios side only**

The aios half genuinely mirrors, and exists today: `POST …/detach` → `DELETE /v1/connections/{id}` (soft-archive, `routers/connections.py:231-244`). Ordering is server-enforced both ways: attach refuses archived rows; archive **refuses while an active binding exists** (`services/connections.py:783-`, mirroring the attach-side `FOR UPDATE`) — a live binding cannot be stranded. Darkness does not depend on the connector hearing anything: after detach, tier-3 routing finds no binding, and the secrets fetch refuses archived rows (`db/queries/connections.py:563`). Archive also pushes `removed` on discovery (`services/connections.py:853-858`) — best-effort NOTIFY in v1 (drop-oldest LISTEN queue, `db/listen.py:200-220`) — on which the connector drops the ghost from its reverse index; if lost, the index is stale only until the next backfill, which excludes archived rows (`db/queries/connections.py:715`). Either way the ghost's room events resolve to no connection and drop — the desired dark state. (§5.3's stale-index warning must treat a `removed` ghost as *retired*, not desynced.)

The Matrix half is **not** an inverse. Provisioning's Matrix side is one lazy idempotent register; retirement's is `deactivate` (irreversible — a deactivated MXID can never re-register) plus leave/forget across every room the ghost ever joined, tombstoned membership state persisting forever — O(rooms-per-agent), monotonically accumulating (§10.2). **v1 deprovisioning is therefore detach + archive: immediately, completely dark on the aios side; the ghost persists inertly on the homeserver.** Ghost/room GC is the §10.2 lifecycle epic, and archiving first loses nothing — the connection row (and its `external_account_id`) is exactly the record the eventual GC walks. One boundary for that epic: `DELETE /v1/connections/{id}` is aios's ordinary *reversible* soft-archive — this section leans on exactly that reversibility — so irreversible Matrix-side retirement must be gated on a positive, explicit retirement fact, never inferred from a bare archive call or a localpart's *absence* in the lossy discovery view; otherwise a routine mute/cleanup archive silently becomes permanent identity destruction.

---

## 14. Admission & per-agent quotas — the graduated posture (settles §10 #4 and #5)

Two planes, separate because aios keeps them separate: **admission** answers *whether* a counterparty may reach an agent — a discriminated `kind` union enforced before any side effect (`inbound.py:152-156`); **budgets/quotas** answer *how much, lately* — rolling counts over the events log, cap-before-side-effect (`inbound.py:170-178`, `wake.py:237-266`).

### (a) The per-agent global inbound budget — a session-keyed count, not an `orig_channel` prefix

**Verdict: budget-plane, keyed on `session_id`.** An earlier revision keyed this cap on an `orig_channel` prefix (`LIKE 'matrix/_aios_agent_42/%'`), claiming free reuse of the 0128 index. Red-team measurement killed that: with `orig_channel` as a *range* rather than the per-counterparty check's *equality* (`inbound_budget.py:98-111`), the 0128 btree demotes `created_at` to an in-index filter — O(agent-lifetime inbounds) on a never-pruned log, worst under the very Sybil flood it exists to stop — and on the actual deployment collation (`en_US.utf8`) the `LIKE` does not range-seek at all (measured: ~680×); range-bound rewrites recover only O(agent-lifetime), never O(window).

The correct key was already in the tree: under §4.1's `single_session` mapping, session ↔ agent is 1:1, so the per-agent cap *is* a **session-keyed** rolling count — the exact shape of the shipped `check_inbound_budget_session` (`inbound_budget.py:112-148`): `session_id` equality seek, `created_at` bounded behind it, collation-independent, O(window). The delta:

- sibling helper `check_inbound_budget_agent(pool, *, account_id, session_id)` under its own knob pair `inbound_rate_agent_window_seconds` / `_max_per_window`, same **default 0 = disabled, zero queries** short-circuit as the existing pair (`config.py:770,792`);
- enforced at all three wake-bearing entry points: post-resolver in `handle_inbound` (Tier-3 `single_session` resolution is a side-effect-free read — "No ledger insert", `resolver.py` — and the per-counterparty check stays pre-resolver, `inbound.py:170-178`, shedding single-source floods first); in the chat-lifecycle wake route (`session_id` already resolved; the wake-bearing lifecycle row appends to that session, `connectors.py:864-928`); and on the session-lifecycle route as a second threshold over its existing count (`connectors.py:775-781`). Scope: `single_session` bindings only — the matrix mapping; for `per_chat`, session ≠ agent and Tier-3 can spawn;
- index: `events_session_created_at_idx (session_id, created_at)` (mig 0022) already window-bounds the query; a session-led partial mirroring 0128's predicate tightens it further and, if adopted, **is a migration**, landed before the knob arms (M4-blocking, per (d)).

Drop semantics: 429 `RATE_LIMITED`, pre-append, non-fatal (`runner.py:127-138`), silent toward the model. **Deliberately not done:** per-connection thresholds (global knobs until a fleet demonstrates non-uniform need — then as its own resource shape, never a column-flag). Deafness under attack is the accepted trade: a distributed flood drops legitimate traffic too — bounded inference spend beats guaranteed liveness; single-source floods trip the per-counterparty window first.

### (b) First contact — the structural gate we already have; one sender-keyed kind designed now, built at M4

"Unknown-sender/first-contact gating" (§4.5, §9 Fix 2) conflates two things.

**First contact into a *new* room is already structurally gated — agent-decides.** Fan-out only emits for ghosts *joined* to a room (§5.2); a ghost joins only by its own `matrix_join` (§5.6). A stranger's opening move is an invite — one inbound event carrying the raw sender MXID, the non-spoofable signal (§9 Fix 1) — and no content flows until the agent joins. Cost: one metered wake per invite, survivable via (a) plus the HS's receiver-keyed `rc_invites.per_user` throttle. `AllowList` **cannot** express this: an invite for an unlisted room is dropped before the agent sees it (`_admits`, `inbound.py:87-99`, keys on `chat_id` only) — the §5.6 violation. So the bridge-phase default for outreach agents is `allow_all` + structural membership + agent-decides, budgets carrying the load — a decision, not an oversight.

**Unknown sender inside an admitted room is unrepresentable today** — the sender is not an input to admission anywhere (the sender dict is display metadata, `inbound.py:300-322,46`). Per the module's growth rule — *"a new admission behavior is always a new `kind`, never a flag"* (`inbound_policy.py:10-12`) — the internal-only posture is **one** new arm:

```python
class AllowSenders(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["allow_senders"] = "allow_senders"
    sender_ids: list[str] = Field(min_length=1)   # exact canonical ids (matrix: MXIDs)
```

Admit iff the canonical sender id (`sender["id"]`, exact MXID, connector-filled) is in `sender_ids`; everything else — content *and* invites — drops as `denied_by_policy` (422, non-fatal). The predicate becomes `_admits(policy, chat_id, sender_id)`; no wire change. Honest limits: the sender id is connector-asserted (adds no new trust assumption — a compromised connector holds `as_token`, §4.4), and across a **relay-mode** bridge the sender MXID is the relaybot, so `allow_senders` allowlists the bridge, not the human (§8). **Designed here; not built in v1**: sender-keyed deafness has no v1 customer (M1–3 counterparties are hand-provisioned users and sibling agents). v1 ships the three-arm union untouched; `AllowSenders` — specified now so the growth path is settled — lands as **Milestone-4 entry work** iff an internal-only agent exists then.

**Posture is the choice of `kind` — never a mode flag**: `allow_all` + membership + §5.6 for outreach agents (strangers cost one metered wake, agent triages); `allow_list` for single-purpose room bots (drops invites too — accepted); `allow_senders` (M4+) for internal-only agents (deafness-by-default is the feature). The write-path gap is owned: `sender_ids` is operator-key PUT-Replace only (`connections.py:106-133`), so an `allow_senders` agent that cold-DMs a new human **cannot hear the reply** until an operator updates policy — hence not the outreach default, and no connector-driven auto-append loop (no runtime write path; a read-modify-write race is a shim we decline). "Admits-but-flags" is rejected: admission is binary before side effects, and the model already reads the sender MXID.

### (c) Per-agent outbound quotas — aios-side containment **plus** HS-side backstop (§10.5's "or" was wrong)

**The hard, per-verb, model-visible quota lives aios-side at the dispatch boundary; the HS `rc_*` limiters stay on as backstop.** Different adversaries: the aios quota contains the *agent* — a prompt-injected model spamming invites/rooms/sends from inside the exclusive namespace (§9 Fix 3); the HS limiter backstops the *process* — a compromised connector holds `as_token` and bypasses aios entirely (§4.4).

**aios side — genuinely new surface that clears the bar.** Nothing meters tool calls today (`tool_dispatch.py:477`, `:103`); the template is the cross-session wake caps (`wake.py:39-40,130-167,237-266`) and the inbound budget that cites them (`inbound_budget.py:13-16`). Third instance:

- **Key: `(session_id, tool_name)`** — under §4.1 session = agent, and it is the only per-agent key the log carries at dispatch: `events` has **no** `external_account_id` column and `orig_channel` is stamped only on inbound appends; every row carries `session_id`, and mig 0022 promoted `tool_name` to a column. Dispatch already holds the session — no resolution, no join.
- **Substrate**: count `role=tool` result rows per `(session_id, tool_name)` in the window — tool-always-appends-result makes it exactly one row per call (assistant rows undercount — `tool_name` stamps only a multi-call turn's *first* name). Honest slack: in-flight calls are invisible at check time; accepted. No counter table.
- **Index**: `events_session_tool_name_seq_idx` (0022) is the right equality prefix but seq-keyed — it reads the agent's *lifetime* calls of one verb. v1 commits to the `created_at` sibling partial: **a small migration, named here, landed before the quota arms**.
- **Per-verb thresholds are settings data** (a magnitude, not a behavior variation): precursors tight (invite/create_room/join tens/hour), delivery loose (send hundreds).
- **Enforced at dispatch**: over quota ⇒ the call is never published; a typed error tool_result is appended (`quota_exceeded: matrix_invite 20/20 per hour`), where ToolBail refusals already surface. No queueing, no connector involvement; `signal`/`telegram`/`whatsapp` get the same containment free, homeserver-agnostic.
- **Invite-target caps** (§9 Fix 3's sharpest ask): distinct-target counting is **not** the same flat count — arguments live in the assistant rows' `data->'tool_calls'` JSONB, not on result rows. Same log, different query shape; a named follow-on.

**HS side — keep the backstop as §9 committed; don't lean harder on it.** `rate_limited: true` stays in the registration (registration-wide, no per-ghost granularity; the `@_aios` sender is always exempt — AS API spec, Synapse PR #8139); tune `rc_*` deliberately at bridge time. But this is **smoothing, not containment**: a runaway ghost at default `rc_message` (0.2/s) still drips ~17k messages/day forever — Matrix has no native per-user hard quota. Per-ghost `override_ratelimit` is an ops lever, not architecture (Synapse-specific; coverage gaps synapse#19149/#15974); a spam-checker module could do true quotas but is the not-native Synapse-only dependency §9 already declined. A throttled ghost's call returns `M_LIMIT_EXCEEDED` (429 + `retry_after_ms`), surfaced as a **raw tool error** — no retry/backoff shim (§6.3's stance).

### (d) The graduated posture — what arms when

Controls arm with the threat, as **entry gates**. This amends §11: the per-agent inbound budget and outbound quotas become **Milestone-4 entry requirements** (M4 currently defers them to M5), with the (a)/(c) index migrations **M4-blocking**. The outbound half must precede the first bridge: a prompt-injected agent spamming *through* a bridge gets the bridge's remote number banned — fleet-level blast radius.

- **Stage 1 — closed pilot (M1–3).** ON: `DenyAll` fail-closed default; `allow_all` at provisioning (§4.5); §5.6 agent-decides invites; HS `rate_limited: true` + default `rc_*`; exclusive namespace, operator-key provisioning, private AS endpoint (§9). OFF: both budget knob pairs 0; outbound quotas; `allow_senders` (not yet built).
- **Stage 2 — bridged (M4, entry-gated).** Newly armed: (a)/(c) migrations, then both inbound budget knob pairs nonzero; outbound per-verb quotas; `rc_invites` reviewed; `allow_senders` built + applied to internal-only agents (if any exist); outreach agents stay `allow_all` + membership (relay-vs-puppet documented per bridge, §8). Still OFF: per-connection thresholds; distinct-invite-target caps (named follow-on); any connector-side limiting.
- **Stage 3 — open federation (maybe never).** Not planned — bridges already deliver reachability (§8). Preconditions, not a schedule: sender-keyed admission as fleet default, a true HS-side hard quota, E2EE-era verifiable identity (§9: v1 has none), a re-derived federated threat model.

**Deliberately NOT enforced, anywhere:** no connector-side rate/quota logic (a dumb pipe; hand-rolled fuzzy limiting is what §9 Fix 3 forbids); no auto-triage of invites (the agent reading the invite *is* the gate, §5.6); no spam-checker module or architectural reliance on `override_ratelimit`; no retry/backoff shim around `M_LIMIT_EXCEEDED`. In one line: **outbound breaches are raw, typed, model-visible errors the agent reasons about through the session log; inbound breaches are silent pre-inference drops the operator sees as 4xx counts — hostile volume must never buy inference.**

---

## 15. Inbound durability & scale — v1 synchronous ack → the Milestone-5 durable path

§5.5's synchronous-ack model — fan out `emit_inbound`, ack only after every envelope is delivered-or-refused, durability = HS `txnId` retry + aios `event_id` dedup — **stays v1**. This section sketches the Milestone-5 upgrade (gated on A.4 measurement) at decision altitude; detail lives in the M5 epic. Not deferred: the receiver posture (a v1 correctness fix) and the discovery ledger (§12's retirement **hard-depends** on it).

### Crash windows, stated exactly

The dedup substrate: every accepted inbound is recorded in a ledger keyed `(account_id, connector, external_account_id, event_id)`, transaction-atomic with the session-log append (`try_record_inbound_ack`, `connections.py:1167-1215`; `_append_with_dedup`, `inbound.py:278-351`); a replay returns `deduped:true`. §5.4 corollary: with `external_account_id` (the ghost localpart) in that PK, the `matrix-{ghost}-{event}` prefix is **redundant for correctness** — kept only for greppability. Exported to §12.7's prune family: `connector_inbound_acks` grows one row per delivered inbound forever (mig 0027: no pruning in v1); any retention window must exceed the max park/redelivery horizon below, else a late redelivery double-delivers.

**v1 has no post-ack loss window** — the `200` *is* delivery: crash/non-200 mid-fan-out → the HS redelivers the `txnId`; delivered envelopes dedup, the rest deliver. **M5 (persist-then-ack) trades that for throughput:** the `200` becomes a *custody transfer* — the HS never resends past its cursor; the queue file is the only copy. Every crash window (pre-commit, post-commit-pre-`200`, mid-drain, pre-row-delete) is covered by queue PK + ledger: sessions observe each event **at most once per ghost**. Residuals, three: (1) latency unbounded under park — delayed, never silently lost; (2) post-ack durability = the container volume — **volume loss after `200` = gone forever** (queue on the spool's persistent volume; decommission = drain, not delete); (3) admission evaluated at drain, not receipt. All three are absent in v1 — hence synchronous ack.

### The durable inbound queue — a sibling of the spool

Precedent correction: **no shipped connector wires `SqliteAnsweredSpool` today** (SDK and tests only) — we reuse its *pattern* (sqlite on the volume, PK-as-dedup, persist-before-ack), not the class (`spool.py:36-72`: no ordering, drain, delete, or payload capacity). A new ~100-line SDK sibling: rows keyed `(txn_id, event_id) PK` with `chat_id` (lane key), `recipients` (snapshotted at receipt; an archived connection surfaces at drain as the server's 422 `DETACHED` → routine drop), `envelope` (BLOB, attachment bytes inline), `attempts`. Decisions:

- **One sqlite txn per HS txn, committed before the `200` — a real fsync, not free.** The only-copy invariant requires `journal_mode=WAL` + `synchronous=FULL` (NORMAL only with its documented OS-crash risk) and **a volume that honors fsync**. The commit runs **off the event loop** (single writer thread — a synchronous fsync on the shared loop would stall the drain lanes and the three SSE loops); its latency is in the ceiling below, measured in A.4.
- **Media bytes into the blob at enqueue** (under the 5 MiB cap, connector-enforced — `sandbox.py:118`'s cap covers only the sandbox-path helper): custody includes bytes; post-`200` the HS may purge `mxc://` media. Enqueue depends only on Matrix, drain only on aios.
- **Drain lanes keyed by room**, rowid-ordered within a lane, bounded lane concurrency (default 8; backoff per `runner.py:302-304`). Per-recipient rows rejected: payload ×N for isolation nobody asked for.
- **Per-envelope terminal outcome is a KIND — `delivered | dropped(policy) | dead_lettered` — never a boolean.** A row completes when every recipient is terminal, then is **deleted**. The third arm is the poison valve: a row failing RETRY-class on N consecutive drain attempts (default 20) moves to a dead-letter table with an alert; its lane resumes. Without it a poison row never deletes and accumulates until it parks the whole shard. Known poison: `ATTACHMENT_STAGING_FAILED` → fatal 500 (`connectors.py:127-128`), making a *permanently* un-stageable attachment look transient; the fix is a small aios-core change — classify it 413-class non-fatal so it drops as policy (the terminal-vs-transient conversion the resolver already does at 422, `resolver.py:90-99`). Dead-letter is the backstop, not the plan.

### Backpressure — park; shedding is a server-side policy verdict

Of buffer/shed/park: **park** — unbounded buffer is a disguised OOM, and shed silently loses correspondence; shedding *as policy* already lives in aios admission (422/429, §10.4's per-agent budget), which the connector must not duplicate. Park is what the AS protocol defines for a slow appservice: at the bound (`MATRIX_INBOUND_QUEUE_MAX_BYTES`, default 1 GiB) the handler returns non-200 and the HS holds the stream — the durable buffer of last resort. **Selective per-ghost backpressure does not exist** (the txn ack is all-or-nothing): a failing lane parks itself while siblings drain; accumulated rows trip the bytes bound and park the shard; per-*session* saturation is invisible at this layer by design (its control point is §10.4, server-side). Parking-at-bound deliberately reinstates head-of-line blocking as an **overload valve**; sharding caps its blast radius.

### The supervised receiver — how HALT actually restarts

The prior draft's HALT — "propagate → TaskGroup teardown → container restart" (`runner.py:874-899`) — **does not work:** the transaction handler is an aiohttp *route handler*, and aiohttp wraps every handler call in `except Exception: resp = self.handle_error(request, 500, exc)` (`aiohttp/web_protocol.py::_handle_request`, :565-588). A raise never reaches the `tg.create_task`-owned server task — it becomes a 500, the HS retries forever, RETRY and HALT collapse, and the crashloop never fires — the exact silent stall HALT exists to prevent. The posture inverts — **the handler never raises into aiohttp; a top-level catch classifies into closed KINDs** (superseding the "no `except Exception`" stance built on that false premise). `ACK` — every envelope delivered-or-refused (v1) / persisted (M5) → `200`. `RETRY` — aios ≥500 or `httpx.TransportError` (uncovered by the status predicate) → non-200; the HS backs off and redelivers. `HALT` — 401/403, or **any unenumerated exception** → non-200 (never acked, so the event survives restart; delivered siblings dedup) **and** set `self._halt`. The restart is out-of-band: `setup(tg)` spawns a halt supervisor as a **real TaskGroup child** — `await self._halt.wait(); raise RuntimeError(...)` — whose raise tears down the group: `run()` exits through `teardown()` → container restart → env/token re-read (`os._exit(1)` only as blunt fallback). Rationale: 401/403 = the boot-cached runtime token (§1) is revoked, which no retry heals — restart is loud (visible crashloop) where the old path stalled silently; an unknown exception is a broken invariant — fail hard into a known state. Ships in v1 — a correctness note, not M5 machinery.

**Honest residual:** a deterministic HALT-class event now *crashloops* — better than a silent wedge, not progress past the poison (v1's RETRY analogue: the staging 500 above). Cures: the 413-class reclassification (v1), the dead-letter arm (M5), and a follow-on poison counter (drop-and-alert after N redeliveries of one txn/event).

Readiness: `wait_ready` counts only the three SDK loops (`runner.py:404-421`); wire the AS ping (Milestone 0's exit check) into the container healthcheck — it catches up-but-wedged as well as down.

### Head-of-line blocking — queue vs. sharding

The HS delivers transactions serially, one in flight, blocking on the `200`, per registration: per-shard throughput = events-per-txn ÷ ack-latency. v1 ack latency = Σ(media) + Σ(serial fan-out POSTs); M5 = Σ(media) + one fsync'd commit — off-loop but still on the ack path (measured in A.4). The queue fixes the ack path (fan-out breadth, bursts); only sharding (§10.1) fixes aggregate fleet rate and park/HALT blast radius; neither fixes the server-side ceilings (A.1). **Sequence: measure → queue → shard** — the queue is software-only and reversible; a shard is quasi-permanent. Ordering: v1 total per-stream; M5 per-room-lane — the only order a conversation needs (nothing depends on cross-room order).

### Paged discovery backfill — the aios-side change, sketched

Recon corrects A.2: the backfill is already server-side streamed and keyset-paged (`api/sse.py:531-617`; `services/connections.py:216-257`). The real gaps at N=10⁶: (1) no client resume cursor (`connectors.py:496-539`) — a disconnect restarts from zero; (2) unrecoverable tail overflow — the drop-oldest LISTEN queue's own `?after_seq=` recovery is inexpressible here (`db/listen.py:163-231`); (3) per-subscriber O(N) `emitted_added` (`sse.py:557`); (4) serial per-`added` secrets GET (`runner.py:980-1032`). And one outranking all four: **no completeness signal** — backfill slides straight into the live tail with no marker; `wait_ready` fires on the 2xx handshake, before any backfill data (`runner.py:404-421`). **Nothing downstream may ever infer anything irreversible from absence in this view.**

The one genuine new primitive: a **sequenced connection-change ledger** — `connection_changes(seq IDENTITY PK, account_id, connector, kind ∈ ('added','removed'), connection_id, external_account_id, created_at)` — written inside the existing attach/archive transactions, `pg_notify` after commit per invariant 6, NOTIFY demoted to a doorbell (the feed reads the ledger by seq; dropped doorbells lose nothing — gap 2 dissolves). Subscribe arms are KINDs, not independent flags:

- **`fresh`**: `cursor` event carrying the change-seq high-water mark S₀ → keyset snapshot → a **`snapshot_complete` sentinel** → feed from S₀. The sentinel is load-bearing: it alone distinguishes "absent" from "not yet streamed," and §12.4's reconcile must never run against a sentinel-less view; emit it on the v1 stream too (one extra yield) as defense-in-depth.
- **`tail(after_change_seq)`**: snapshot already held; replay the delta, then live — a connector persisting its view (one more table in the M5 sqlite) restarts in O(Δ), not O(N). **Also the retirement substrate:** retirement (as revised in §12) is driven only by positive `removed` facts replayed through `tail()` — never by absence from a backfill — making this ledger a **hard prerequisite for §12's retirement arm**: no absence-based deactivation ever ships, no retirement before this ledger.
- A `resume` arm (restartable first snapshot) would serve only a first-backfill disconnect; after one full backfill `tail` covers every restart. **Ship `fresh` + `tail`; add `resume` only if A.4 measures first-backfill disconnects to matter.**

A cursor past the retention horizon → `reset` → the client re-runs `fresh`, deactivating nothing on that pass (the §12 rule). Cursor-addressing makes `added` at-least-once: `emitted_added` is dropped (gap 3 → O(1)); the SDK added-handler gains a one-line idempotence skip. Gap 4: a connector class declares its connections secret-less (matrix v1 is, §4.4), skipping the per-`added` GET; others get bounded-concurrency prefetch. **Gating:** none of this is needed for Milestones 0–4 (a pilot-N re-backfill costs seconds); one aios-core PR, landed **early in Milestone 5** — the queue's restart story, shard rebalancing, and §12's positive-signal retirement all lean on `tail()`.

### The honest ledger

v1 buys correctness with the homeserver's own retry machinery and zero new state; Milestone 5 buys throughput by taking custody earlier — paying with a volume-durability invariant, drain-time admission semantics, and a queue file that is data, not cache. Given a volume that honors fsync, neither model ever loses an acked message to a crash; only the definition of "acked" moves.

---

## 16. E2EE path — the ladder, the decision, the trigger

> **Resolves §10 OQ#6 and Appendix A.5.** v1 remains **plaintext-only** (§1 non-goal, unchanged). This replaces §8's deferred hand-wave with a verified MSC map, a graded ladder, a committed target and gates, and the trigger that starts the work. Statuses verified **2026-07** (spec through v1.19) — a **dated snapshot; re-verifying E.1 is the epic's step zero**. Durable commitments: target, ladder, trigger; versions/flags/dates are a checklist.

### E.1 The corrected MSC map — §8/§10.6 posed the wrong fork

§8 framed the decision as "MSC3202 (shared bridge-bot crypto) **vs** MSC4190 (per-ghost devices)". Verified, the framing itself was wrong:

| Piece | MSC | Status (2026-07) | Synapse gate |
|---|---|---|---|
| Masquerading (`as_token` acts as a `device_id`) | **MSC4326** (split from 3202) | **Stable — spec v1.17** | Flag-free ≥ **1.141.0** |
| Device management sans `/login` | **MSC4190** | **Merged — v1.17**; in MAS | Flag-free ≥ **1.141.0** |
| Crypto-data transport (OTK counts, device-lists in AS txns) | **MSC3202**-ext | **Open, unmerged** | experimental flag + registration flag (E.3) |
| To-device delivery to appservices | **MSC4203** (split from 2409) | **Open, unmerged** (not in v1.19) | experimental flag (E.3; historical name) |
| "Invisible crypto" (clients exclude non-cross-signed devices) | **MSC4153** | **Accepted; recommendation in v1.18.** Element enforcement delayed **Apr → Oct 2026** (bots/bridges not ready) | Client-side |
| Delegated encryption (clients drop warnings for delegating ghosts) | **MSC4350** (tulir) | **Open; no qualifying implementation; cannot enter FCP** | — |

Three corrections that reshape §8's fork: (1) **masquerading is no longer experimental** — stable spec, flag-free ≥ 1.141; it was never the risky part. (2) **MSC4190 is not the per-ghost arm — it is device *management*, needed by *both* crypto models** on any MAS-fronted homeserver (matrix.org itself runs MAS, which removed `m.login.application_service` from `/login`); any E2EE rung is **MSC4190-first, never `/login`**. (3) **The true experimental residue is the crypto-data transport**: MSC3202-ext + MSC4203, unmerged and flag-gated, required for **any** appservice crypto, shared *or* per-ghost — and the fallback (`/sync`-as-appservice-user) is dead: Synapse ≥ 1.141 blocks appservice-user `/sync`.

Collaterally confirmed: §4.3's `inhibit_login` note is now spec text (v1.17 mandates `inhibit_login: true` on AS `/register`) — **rung 0 is completely MAS-proof**. And §8's libolm footnote hardens: `mautrix.crypto` still sits on **libolm** (deprecated 2024-08, CVE-2024-45191/-45192), zero vodozemac-migration entries through v0.21.1 (2026-07-05); nothing wires the maintained vodozemac-python bindings into it.

### E.2 The ladder

**Rung 0 — plaintext-only (v1, shipped as designed).** No devices, no keys, no crypto store — and **no warnings either**: MSC4153 says clients MUST NOT treat *non-cryptographic* users (no identity keys) as insecure, so plaintext ghosts in unencrypted rooms stay clean in Element after October 2026. Cost: encrypted rooms are unjoinable — Element's encrypted-by-default *native* DMs (the real §8 blocker) stay out of reach. *Gate: none.*

**Rung 1 — shared connector-bot device, appservice-mode transport (the mautrix production model).** One crypto device, owned by `@_aios:server` (§4.2); ghosts have **no devices and no keys**. Both directions spelled out — the receive path is where a naive reading dies:

- **Outbound:** ghost sends use the bot device's per-room outbound Megolm sessions ("all messages sent by the bridge are encrypted using the bridge bot's session" — mautrix E2BE docs).
- **Inbound:** a human's client Olm-shares each room key only to room members' *devices*. Deviceless ghosts give it nothing to share to; the key-receipt path is the **bot's own membership** — "The bridge bot is necessary when using end-to-bridge encryption" (mautrix troubleshooting). Rung 1 therefore requires **`@_aios` joined to every encrypted room**: a visible third member in every encrypted "DM". Creation path: invite the bot with `m.room.encryption`; invite path (§5.6): the ghost invites the bot after joining, or declines — an encrypted room the bot can't enter is undecryptable. Keys arrive as to-device Olm via the MSC4203/MSC3202-ext stream.

Production-viable only since **October 2025** (mautrix: "now safe to enable", Synapse ≥ 1.141 + regenerated registration). *Sub-rung 1a (`/sync`-mode crypto): do not build.*

**Rung 2 — per-ghost devices (the verifiable-authenticity rung).** A real MSC4190-minted device + Olm account per agent, cross-signed. The identity pieces are stable spec — but per-ghost to-device/OTK delivery rides the **same unmerged flags as rung 1**, means N crypto stores and N device-list syncs (mautrix: "ridiculously inefficient, so it won't happen"), and **no framework supports it**. Also the *only* rung delivering §1's deferred "verifiable agent authenticity" (§9 Fix 1's gap) and clearing the warnings. Research-track.

**Rung 3 — MSC4350 delegated encryption.** Ghosts declare the bot as encryption delegate; clients drop the warnings; rung 1's single-store economics kept. The ecosystem's intended endgame — and vapor today. *Watch, don't build.*

### E.3 DECISION — target end-state and its gates

**Verdict: the committed E2EE target is rung 1 — one shared connector-bot device per shard, appservice-mode transport, self-signed from the first encrypted message.** Rung 2 stays research-gated (Appendix A.5); rung 3 is a revisit trigger, not a plan.

**Named gates (re-verify at epic start):**
- Synapse **≥ 1.141.0** (masquerade + MSC4190 flag-free; also kills `/sync`-mode).
- Registration regenerated with **`org.matrix.msc3202: true`** (a Milestone-0 artifact change; else append-only, §4.2).
- `homeserver.yaml`: **`experimental_features: {msc3202_transaction_extensions: true, msc2409_to_device_messages_enabled: true}`** — both gating **unmerged** MSCs; explicit acceptance of experimental, Synapse-only surface.
- **MSC4190 device minting only; never `/login`** — MAS-proof by construction.
- **`self_sign` from day one**: mandatory before October 2026, or agent messages stop being decrypted by default. mautrix-python quirk: v0.21.0's `self_sign` **logs** the recovery key instead of storing it — persist it deliberately.

**Crypto mode is a KIND, not a flag** — and the union holds only committed arms:

```
crypto:
  {"kind": "none"}                                      # v1 — rung 0
| {"kind": "shared_device",                             # the epic — rung 1
   device_owner: "@_aios:server", self_sign: …}
```

If rung 2 is ever pursued, a `per_ghost` arm joins the union *then* — pre-minting a discriminated arm for a research-gated rung no framework supports is speculative surface, not KIND hygiene.

Consequences of the arm:

- **The tool surface does not change.** `matrix_send`/`matrix_react` unchanged; encryption is transport inside the connector (encrypt-before-PUT, decrypt `m.room.encrypted` → `emit_inbound` — the row §5.1 reserved). Whether a `matrix_create_room` room gets `m.room.encryption` — and, when it does, that the bot is invited as key-receipt member (E.2) — is a *consequence of the crypto arm*, never a per-call model argument; "encrypted room on a crypto-less connector" is unrepresentable, not runtime-guarded.
- **No new aios API surface.** The crypto store (Olm account, Megolm sessions, device lists, cross-signing/recovery keys) is **connector-local Postgres beside the membership `state_store` (§5.3): raw SQL, asyncpg** — *not* the connection secrets dict: rung 1's device belongs to the appservice bot, not any connection (§4.4), and crypto state ratchets on every message while secrets are spawn-cached credential material (`runner.py:341-345`). §4.4's footnote is narrowed: at rung 2 only the slow-moving per-ghost seed (recovery/master key) would fit the secrets dict — and even that hits a real gap: the runtime token only *reads* secrets (`connectors.py:1007`), writes are operator-key-gated (`connections.py:84-102`) — no connector write-back path today. A rung-2 aios-side question, **not** invented here.
- **Crypto stack: do not adopt `mautrix.crypto` without re-checking its libolm status.** §8's "`mautrix.crypto` is why we picked it" is corrected: `mautrix.appservice`/`mautrix.client` remain the AS/CS layer; the crypto layer is a deprecated-libolm dependency with two CVEs and no migration in sight. The epic's first spike is a build-vs-wait decision — wire vodozemac-python into our own appservice-mode OlmMachine, or adopt `mautrix.crypto` if migrated by then — and the spike's exit test is the **full inbound path** (client key-share → bot device → MSC4203 to-device → decrypt → `emit_inbound`), not just outbound. Honest status: **no off-the-shelf Python appservice-mode crypto on a non-deprecated stack exists as of 2026-07** — the epic's biggest risk (production bridges are Go).

### E.4 What it unlocks / what it costs

**Unlocks (rung 1):** encrypted native Element DMs — the v1 reachability blocker (§8); encrypted bridge rooms (`encryption.default: true` on Milestone-4 bridges); rooms we don't control (§5.6 invites into already-encrypted rooms).

**Costs (why the trigger is demand-driven, not calendar-driven):**
- **Client verification UX.** Every agent message in an encrypted room wears real Element shields ("encrypted by a device not verified by its owner", etc. — the set mautrix documents for its own ghosts); `self_sign` keeps messages *deliverable*, only MSC4350 clears the shields. And the bot is a **visible third member in every encrypted DM** (E.2). Real product costs for a design whose §9 concedes identity legibility is weak.
- **Blast radius.** One shared device/store holds the keys for the whole fleet's traffic per shard (Megolm sessions are per-room and rotate; the single point is the *device/store*, not one session): plaintext-at-connector, zero per-agent authenticity — the `as_token` single-point shape (§4.4) extended to message *content*. §1's authenticity non-goal stays open until rung 2.
- **OTK/device-list upkeep rides the serial stream** §5.5 head-of-line-blocks on. Rung 1 adds one device's upkeep (cheap); rung 2 would put N ghosts' OTK claims there — a fleet-restart key-claim storm on a serialized pipe (§10/A.4 ceilings).
- **Key storage growth.** Inbound Megolm sessions scale with rooms × senders × rotations, forever — a new unbounded table joining §10.2's lifecycle/GC problem.
- **Homeserver portability tax.** The experimental flags are Synapse-only in practice (Tuwunel/others incomplete); rung 1 partially breaks A.1's "speak only the standard AS API" hedge until the MSCs merge; rung 0 shards keep it intact.

### E.5 Trigger — what starts the epic

**The trigger is a named counterparty, not a calendar date.** Start when product commits to agents reaching a human whose room cannot be plaintext — a native Element DM where encryption-by-default can't/won't be downgraded, or an operator enabling end-to-bridge encryption on a Milestone-4 bridge. Earliest sensible slot: **after Milestone 4**, filling Milestone 5's "decide the E2EE path" line as *build* — the decision is already made here. **Epic step zero: re-run the E.1 table.**

**Explicit non-trigger:** Element's **October 2026** MSC4153 enforcement does **not** force rung 0 off plaintext — non-cryptographic users are exempt, so plaintext ghosts in unencrypted rooms are unaffected. The October date binds only *inside* the epic (self_sign), not its start.

**Re-evaluation triggers (any reopens this section):** (1) MSC3202-ext or MSC4203 merges → transport stops being experimental, portability tax drops. (2) MSC4350 gains a qualifying implementation → rung 3 becomes real; if before the epic, evaluate rung 1 + 4350 together and eliminate the warning cost. (3) `mautrix.crypto` migrates off libolm → the build-vs-wait spike flips toward adopt.

---

## Appendix A — Open questions handed to the scaling research track

> **Scaling-track verdict (2026-07-10).** A dedicated deep-research pass (adversarially verified: 22 confirmed claims, 3 refuted, ~24 sources) resolved the **homeserver-choice** half of A.1 below and sharpened the rest. Headline: there **is** a genuinely open-source path, but the honest proven ceiling is lower than "millions" and the binding constraint is **appservice maturity, not raw scale**. Critically, the research confirmed that **no independently-verified capacity or cost numbers exist in public** for a Matrix homeserver at millions of active users — in *either* direction — so the numeric ceilings in A.3/A.4 remain genuinely unmeasured, not merely un-looked-up. Source bias is the dominant caveat: nearly every community-vs-commercial performance figure traces to Element (the vendor selling the paid tier), with zero surviving independent benchmarks.

1. **Homeserver choice — RESOLVED: community Synapse with workers, kept swappable. Measured ceilings still open.**

   **Verdict: run the fleet on community Synapse (AGPLv3) with worker sharding.** It is the *only* homeserver that is simultaneously (a) fully open-source and (b) **appservice-mature** — and (b) is the binding constraint, because an `@_aios_agent_*` fleet lives entirely on the AS API. The lean alternatives all fail (b): **Dendrite** (Go, Element-hq's own) self-describes appservice support as *"not well tested"* and *"not for you yet — install Synapse instead"*, and its repo went maintenance-mode (archived Nov 2024); **Continuwuity** (Rust) has appservices on its active roadmap with only *partial* device masquerading as of v0.5.6; **Tuwunel** (Apache-2.0 Rust, the conduwuit successor) publishes no AS-scaling support at all (a claim that it advertised such support was *refuted 0-3* in the research). The Rust servers are leaner and faster per user, but betting the whole substrate on the most immature, most in-flux part of a young project is the wrong risk. **Do not** architect around them yet, and **do not** architect around Synapse Pro either — see below.

   **The Synapse Pro fork is real but should not shape v1.** Community Synapse **already** shards horizontally across worker processes — since v1.22 (Nov 2020) events no longer funnel through the main process, so multiple event-persister workers write concurrently (matrix.org's own engineering post). CPython's GIL caps each *process* at ~one core, so you scale by running more processes; this mechanism is textbook-sound and established by the OSS maintainers years before the commercial tier existed. **Synapse Pro** (proprietary, commercial-license-only, *no public source*) reimplements the workers in Rust so each uses multiple cores, and adds elastic autoscaling. Its advertised "80%+ efficiency / 500x / community-Synapse-will-fail-at-nation-scale" claims are **all vendor marketing with zero surviving independent benchmarks**; Element's "1-100 users, not for production" band on community Synapse is *positioning language*, not a benchmarked cap. Two honest asterisks: community Synapse's top-end multi-writer event sharding is still labeled **experimental** and is operationally rigid (fixed room→worker mapping at startup; you cannot add/remove event-persister workers without restarting all of them); and free-threaded (no-GIL) CPython (3.13t/3.14t) could erode Pro's whole premise, but Synapse does not deploy it today.

   **Honest ceiling for planning.** The largest *verified fully-open* deployments sit at **100K–350K users** (Tchap ~350K on a private federation; Bundeswehr BwMessenger 100K+). No verified public number proves a single open Synapse cluster at *millions* of active users — and none proves it *cannot*, either; the top end is simply under-tested in public. For our workload the mix is favorable-but-unbenchmarked: most identities are **ghosts** (cheaper than active human clients — no sync loops, no push), so the effective ceiling for an agent fleet may sit well above the human-client numbers, but that is precisely the unmeasured quantity (A.3/A.4).

   **Design consequence (already partly reflected in this doc):** keep the connector **homeserver-agnostic** (speak only the standard AS API — §3.1) and shard along the **namespace** axis (§10.1), so migrating community Synapse → Synapse Pro → a matured Rust server is an ops change, not a connector rewrite. **Swappability is the hedge.** The **XMPP/ejabberd** alternative (from the prior substrate research) stays alive on paper *specifically because* the millions-ceiling is unproven — but note its own headline ("2M concurrent connections/node") **did not verify** in the research and would need its own benchmark before it could be trusted as a fallback.

   **Still open (measure, don't guess):** ghosts-per-homeserver, rooms-per-homeserver, **HS→AS transaction throughput on the single serial stream** (§5.5/§10 — the sharpest architectural ceiling), Synapse's per-event "interesting appservices/users" interest computation (historically ~O(rooms/members) and a known bottleneck), `application_services_txns` DB growth, and large-room state-resolution cost. At what measured active-agent count, if any, does the operational burden of Synapse-with-workers argue for the commercial tier or the leaner XMPP substrate — this is the single most decision-relevant unknown, and it has **no published answer**.

2. **Does "one aios connection per agent" scale to millions? — DESIGNED, pending numbers.** The mechanism set is complete: §12 retirement pins active-connection N to the *live* fleet (this item's old framing was corrected — the backfill already excludes archived rows; lifecycle's job is making "active" mean "alive"); §15 replaces the O(N)-restart backfill with a sequenced `connection_changes` ledger and discriminated subscribe arms — post-review: **`fresh` and `tail(after_change_seq)` ship, `resume` is added only if a mid-first-backfill disconnect is measured to matter, and every snapshot-bearing path emits an explicit snapshot-complete sentinel** so no consumer ever acts on a partial snapshot — drops the per-subscriber `emitted_added` set to O(1), and lets secrets-less connector types (matrix v1) skip the per-`added` secrets GET. The ledger was also promoted by review from optimization to **hard prerequisite for lifecycle**: its durable `removed` entries are the only sanctioned trigger for the irreversible retirement path (§12 as revised). §10.1 sharding divides per-container N; §12.7's prune family bounds `connections`/`bindings`/`connector_inbound_acks` table size. It remains an aios-core PR (migration + queries + `sse.py` + route + SDK), landed first in Milestone 5 — at pilot N (10²–10³) the v1 full-backfill is correct and cheap. **Still open (measure):** millions-scale `connections`/`bindings` query behavior, the change-ledger's retention horizon, and the N at which full re-backfill stops being acceptable.

3. **Per-shard resource budget — STILL OPEN (measure, don't guess), with its inputs now bounded.** Unchanged deliverable: order-of-magnitude RAM/CPU per connector process → ghosts-per-shard ceiling → shard count K (Appendix B still refuses a guessed number). Post-review sizing inputs: the membership store is bounded — rows ≤ live ghosts × `MATRIX_ROOMS_PER_GHOST_MAX` (§12.5); the lifecycle residue shrank (no permanently-retained `retired` registry rows — only in-flight `retiring` rows plus at most an append-only minted-localparts census); the Milestone-5 sqlite queue adds a bounded disk footprint (`MATRIX_INBOUND_QUEUE_MAX_BYTES`, §15) **plus a per-transaction durable-commit (fsync) cost on the container volume** that the throughput model must include. The load test must also vary **fan-out breadth** (members-per-room): per-shard inbound cost is O(room members) per event and nothing bounds ghosts-per-room today (§10 #9) — it is a first-class axis of the ceiling, not a fixed parameter.

4. **Inbound throughput & durability at scale — DESIGNED, gated on measurement.** §15 is the full design: persist-then-ack sqlite queue keyed `(txn_id, event_id)` (subsuming mautrix's in-memory txn dedup), media downloaded at enqueue (custody includes bytes), recipients snapshotted at receipt, room-keyed drain lanes (serial within a lane, bounded concurrency across), delete-on-complete; backpressure = **park** (non-200 at the queue bound → the HS holds the stream as durable buffer of last resort; shedding stays a server-side policy verdict, never a connector decision). Post-review hardening: the durable commit is specified (`journal_mode=WAL` + an explicit `synchronous` level, commit run **off the event loop** so drain lanes and the SSE loops keep progressing; deployment invariant: the volume honors fsync and is drained before decommission) and the M5 ceiling is re-derived as events-per-txn ÷ (media download + commit latency + ack latency); drain outcomes gain a **bounded-retry → dead-letter (park-with-alert)** terminal arm so a poison envelope — e.g. a permanently un-stageable attachment, today a fatal 500 — cannot wedge a lane or the shard (§10 #13). The receiver outcome KINDs — `ACK`/`RETRY`/`HALT`, 401/403 → HALT, transport errors → RETRY — **ship in v1**, refining §5.5, with the review-corrected HALT mechanism: the handler never raises into aiohttp (a handler raise is swallowed into a 500 and restarts nothing); HALT = non-200 for the current txn **plus** an out-of-band halt-supervisor TaskGroup child that raises → teardown → container restart, so a revoked runtime token crashloops loudly instead of stalling silently, and the M0 AS-ping healthcheck catches an up-but-wedged receiver. What the queue fixes vs what only sharding fixes is tabulated in §15; the forced sequence is **measure → queue → shard**. **Still open (measure — the A.1 list):** where the v1 synchronous-ack ceiling bites, aios `POST /inbound` server capacity, Synapse's per-event appservice-interest computation, and the dead-letter redelivery threshold.

5. **E2EE for a fleet — DECIDED at the architecture level; rung 2 is what remains research-gated.** §16 dissolves the old "shared bridge-bot crypto vs per-ghost devices (MSC3202/MSC4190)" fork with a verified map (masquerade = MSC4326, stable spec v1.17; MSC4190 = device management needed by *both* models; the true experimental residue = MSC3202 transaction extensions + MSC4203, both unmerged and Synapse-flag-gated) and commits to **rung 1**: one shared crypto store per shard, appservice-mode transport, MSC4190-only device minting (never `/login`), `self_sign` from day one. **Review correction (receive path):** "ghosts have no devices" holds only for *sending* — inbound Megolm keys are Olm-shared to devices the sender's client can see, so rung 1 must either mint per-ghost device entries that all share the one central store (the shape real mautrix appservice-mode crypto uses) or put the connector bot in every encrypted room; the E.3 first spike resolves inbound key receipt explicitly, not just outbound encryption. (Sibling fix: "one Megolm session encrypts the whole fleet's traffic per shard" was imprecise — Megolm sessions are per-room and rotate; the shared thing is the store/device; the blast-radius conclusion stands.) Gates named and verified as of 2026-07, kept as a dated checklist re-verified at epic start: Synapse ≥ 1.141.0, regenerated registration with `org.matrix.msc3202: true`, two experimental homeserver flags. v1 config stays `{kind:"none"}`; the epic adds a `shared_device` arm; no speculative `per_ghost` arm is pre-declared. Trigger: a named counterparty (§16 E.5), earliest post-Milestone-4; Element's October 2026 MSC4153 enforcement does **not** pressure plaintext rung 0. **Still research-gated: rung 2 (per-ghost devices/stores)** — the only rung delivering verifiable agent authenticity — N crypto stores and device-list/OTK churn riding the serial HS→AS stream, no framework support, plus the aios-side secrets write-back gap (§10 #10). Cross-cutting: the crypto-stack build-vs-wait spike (§10 #11 — no non-deprecated Python appservice-mode stack exists); rung 1's two experimental flags partially suspend item 1's "speak only the standard AS API" swappability hedge until MSC3202-ext/MSC4203 merge; MSC4350 (rung 3) is a watch trigger that would eliminate the verification-warning cost.

6. **Agent lifecycle / GC at scale — DESIGNED (as revised by review); only the throughput numbers stay open.** §12 post-review: retirement is **positive-fact-driven** — the trigger is §15's durable `removed` ledger entry consumed via a persisted cursor, never absence from the discovery view (three upheld findings: absence conflates archived with not-yet-backfilled, lossy-tail-dropped, and reparented, and one reconcile pass over a partial backfill would have irreversibly deactivated live agents); the reconcile arm re-drives in-flight `retiring` rows only; the one-way `deactivate` is gated on positive archived confirmation + a full-cycle dwell, capped by a per-pass blast-radius fuse (halt-and-alarm above a threshold), and demoted to best-effort — for a plaintext `inhibit_login` ghost it reclaims a directory entry; **leave+forget is the load-bearing cleanup**, with HS-side purge via `forget_rooms_on_leave: true` (noted: homeserver-wide, affects human accounts) + `forgotten_room_retention_period` (Milestone-0 settings, §12.5). The bulk-retirement cost model stands with honest numbers: ghosts **leave + forget before deactivating**, bypassing Synapse's serial `_user_parter_loop` by construction; the irreducible N×R leave-persists are issued at `MATRIX_RETIRE_CONCURRENCY` — **days-to-weeks, not "hours," for 10⁴ ghosts × 50 rooms under `rate_limited:true` at concurrency 4**; revisit the rate posture for planned mass retirement or accept the horizon. The monotonic residue shrank and is re-classified: no permanently-retained `retired` registry rows (§4.1's mint rule is the sole reuse guard); deactivated HS user rows and leave-state in never-purged rooms remain, named and accepted (§12.8); `connector_inbound_acks` is now correctly listed as the fastest-growing residue — O(total-messages), pruned age-keyed with retention ≥ §15's park/redelivery horizon (§12.7). **Still open (measure — these join A.1's measure-don't-guess list):** leave-persist throughput at fleet scale, background room-purge cost under the retention config, and the two Milestone-0 deactivation-behavior verifications — whose failure now resolves to "drop the deactivate step" (leave+forget already suffices), not to provisioning the server-admin credential §12.5 rejects.

---

## Appendix B — Rejected review notes

1. **"Fold `matrix_react` into `matrix_send` as a `kind`."** *Rejected (kept separate).* The shipped Telegram connector — this design's north star — ships `telegram_send` and `telegram_react` as separate FaF tools, and the two payloads are disjoint (reaction = target `event_id` + emoji only; message = text/format/reply/attachments), so a shared `kind` would add mutually-exclusive optional fields to one tool and *increase* illegal-state surface under the SDK's flat-kwargs schema. `kind` is correct for same-resource variation (`matrix_create_room`), wrong for two disjoint verbs. *(The sibling suggestion — cut `matrix_set_profile` from v1 — was **accepted**, §6.2.)*

2. **"Give an order-of-magnitude RAM/CPU budget and a ghosts-per-shard ceiling now."** *Rejected as a v1 deliverable (deferred, not dismissed).* Quoting a specific number without the load tests would be false precision. The ceiling is acknowledged qualitatively (single process, single `as_token`, single serial stream = a hard per-shard limit well below "millions active"); the actual number is a research-track output (Appendix A.3) that sets shard count K.

3. **"Widen the localpart character class to `[a-z0-9._=-]`."** *Rejected as unnecessary.* The underscores in `_aios_agent_` are literal in the fixed prefix, not matched by the variable-suffix class; the suffix is machine-generated `[a-z0-9]`, and MXID localparts must be lowercase per grammar. Widening the class would only matter if we derived localparts from free-form names, which we do not. The escaping/anchoring half of that same review point (`^…:your\.server$`) **was** accepted (§4.2).

4. **"Retirement's commit point must be a dedicated retire verb (or a verified-dead-session precondition), not the connection archive."** *Rejected (durability, medium).* aios already splits the semantics the finding wants: **detach is the reversible mute; archive is the permanent commit** (it scrubs secrets and has no unarchive path — §12.2), so a parallel retire endpoint would be a second commit-point primitive duplicating archive. The accident-blast-radius concern was absorbed elsewhere instead: the review-added dwell (a `retiring` row persists ≥1 full reconcile cycle before `deactivate`), the positive-archived-confirmation gate, and the per-pass blast-radius fuse give erroneous or bulk archives a loud abort window, and the reversible steps (leave+forget) precede the gated one-way door.

5. **"Drop `MATRIX_ROOMS_PER_GHOST_MAX` — the idle-TTL sweep plus the §14(c) quota already bound a ghost's room footprint."** *Rejected (simplicity, medium).* The three mechanisms bound different quantities: the quota bounds *rate* (creations/joins per hour), the TTL bounds *steady state* under bounded rate, the cap bounds *stock* deterministically. Under exactly the runaway/prompt-injected case, stock ≈ quota-rate × TTL ≈ tens of thousands of rooms per ghost — two orders above the cap — and both the membership-store sizing (A.3) and the per-ghost retirement cost R (§12.6) key on the deterministic bound. One axis each; not duplicate mechanisms.

6. **"Collapse §15 to a paragraph + deferral — it is implementation-grade design for research-gated M5 work."** *Rejected in the main (simplicity, medium).* The adversarial pass itself promoted §15's `connection_changes` ledger from optimization to **hard prerequisite for safe retirement** (its durable `removed` fact is the only sanctioned trigger for the one-way door), and §15's precision is what let review catch the absence-trigger and HALT-mechanism defects at design time rather than in production. The over-abstraction half **was folded**: `resume(after_connection_id, after_change_seq)` is demoted — ship `fresh` + `tail`, add `resume` only if a mid-first-backfill disconnect is measured to matter.

7. **"Compress §16 to prose — frozen MSC/version/date tables guarantee staleness; drop the `per_ghost` config arm."** *Rejected in the main, folded in part (simplicity, low).* The verified, dated map is what corrected §8's wrong fork framing and discharged its verification caveat; it stays as a **dated (2026-07) checklist re-verified at epic start** (E.5 already carries re-evaluation triggers). Folded: the speculative `per_ghost` arm is dropped — v1 ships `{kind:"none"}`, the epic adds `shared_device`, and `per_ghost` appears only if rung 2 is actually pursued.

8. **"§14(c)'s outbound quota inherits the per-agent index pathology via an `orig_channel` prefix" / "per-verb counting requires jsonb-unnesting `data->tool_calls`."** *Refuted on the facts (the rider of an otherwise-upheld high finding; recorded because the refutation reshaped §14(c)).* Outbound tool events carry **no** `orig_channel`, and `events` has no `external_account_id` column at all; every tool-result row carries `session_id` (NOT NULL) and the physical `tool_name` column migration 0022 promoted — so per-verb counting is a flat `count(*)` keyed `(session_id, tool_name)`: no prefix, no unnesting, no resolution needed at the dispatch boundary. The upheld substance — the originally-written quota key named a non-column — was folded by re-keying §14(c) session-side.

---

## Appendix C — v2 adversarial-review changelog

- **Retirement inverted from absence-driven to positive-fact-driven** (three upheld high findings folded): §12.4's discovery-absence diff is deleted; the sole trigger is §15's durable, sequenced `connection_changes` `removed` entry replayed via a persisted cursor — absence in the discovery view conflates archived with not-yet-backfilled / lossy-tail-dropped / reparented, and the wired action was irreversible. The §15 aios-core PR becomes a **hard prerequisite** for lifecycle and moves to the front of Milestone 5.
- **The one-way `deactivate` gained three guards** — positive archived confirmation, ≥1-cycle dwell, and a per-pass blast-radius fuse (halt+alarm above a threshold; deactivate-nothing after a cursor reset) — and was demoted to best-effort: leave+forget is the load-bearing cleanup, and the illusory admin-API fallback claim was struck (it needs the admin credential §12.5 rejects).
- **Snapshot-complete sentinel** added to the discovery protocol (v1 stream and §15 arms) as defense-in-depth: no consumer ever acts on a partial snapshot.
- **Ghost registry simplified** (partially-upheld simplicity finding): `live` derives from the SDK connection map; permanently-retained `retired` rows dropped (the §4.1 mint rule is the sole reuse guard); what persists is in-flight `retiring` state plus at most an append-only minted-localparts census.
- **Shard key rebuilt** (upheld): sharding on a bare ULID's first char is degenerate (timestamp bits — `0` until ~2248, verified against aios's own ULID generator) and *silent* (`M_EXCLUSIVE` cannot fire when every localpart legitimately matches one shard's regex). Localparts become `_aios_agent_<s><ulid>` with a mint-time-balanced, fixed-position base32 shard char, minted from day one; the `0*…f*` hex regex example deleted; §4.1 patched with the normative form.
- **Per-agent inbound budget re-keyed session-side** (two upheld findings): the `orig_channel`-prefix form cannot range-scan the 0128 index (empirically verified on the deployment's `en_US.utf8` collation; even explicit bounds/`text_pattern_ops` stay O(agent-lifetime)); it is now a session-keyed rolling count reusing `check_inbound_budget_session`'s shape plus one small session-led partial index, run post-resolver. The "no new index, no migration" claim was struck; both quota knobs remain M4 entry gates with the migration landing first.
- **Outbound quota re-keyed `(session_id, tool_name)`**: the originally-written key named a non-column (`external_account_id` is not on `events`); the rider claiming §14(c) inherits the prefix pathology was refuted (tool-result rows carry `session_id` + the 0022 `tool_name` column) and logged in Appendix B.
- **HALT mechanism fixed** (upheld): a raise inside the aiohttp transaction handler is swallowed into a 500 and restarts nothing; HALT is now non-200 for the current txn plus an out-of-band halt-supervisor TaskGroup child (raise → teardown → restart → env re-read). §3.3/§5.5/§5.7 patched to the corrected KIND table; the AS-ping wired into the M0 container healthcheck; the poison-envelope residual named honestly, with a dead-letter arm in the §15 queue and new open question #13.
- **§15 durability spec hardened** (medium): WAL + explicit `synchronous` level, commit off the event loop, volume-must-honor-fsync deployment invariant, ceiling re-derived to include commit latency.
- **`resume` subscribe arm demoted** to measured-need (`fresh` + `tail` ship); the broader "collapse §15 to a sketch" finding rejected — the review made §15 more load-bearing, not less (Appendix B #6).
- **`connector_inbound_acks`** added to the §12.7 prune family and reclassified in §12.8 as the fastest-growing residue (O(total-messages)), with retention explicitly ≥ §15's max park/redelivery horizon so dedup survives late redelivery (medium).
- **Members-per-room named** as the unbounded fan-out amplification axis (medium, folded minimally): routed into §10 #9 (the symmetric cap composes onto the §14(c) substrate) and into the A.1/A.3 load-test inputs as a first-class axis.
- **Bulk-retirement horizon honesty** (low): "hours" corrected to days-to-weeks for 10⁴ ghosts × 50 rooms under `rate_limited:true` at `MATRIX_RETIRE_CONCURRENCY=4`.
- **E2EE rung-1 receive path corrected** (medium): inbound Megolm keys are Olm-shared only to visible devices, so rung 1 needs per-ghost device entries over one shared store (the mautrix appservice-mode shape) or bot room membership — the E.3 spike now resolves inbound key receipt explicitly; "one Megolm session per shard" fixed to shared store/device; the speculative `per_ghost` config arm dropped (v1 = `{kind:"none"}`).
- **`forget_rooms_on_leave` flagged homeserver-wide** in the M0 patch (it changes leave semantics for human accounts on the fleet HS too); an M0 deactivation-test failure now resolves to "drop the deactivate step," never "provision the rejected admin credential" (two low findings).
- **`allow_senders` implementation moved to Milestone 4**, where §14(d) first arms it (YAGNI fold); v1 ships the existing three-arm admission union plus the structural membership gate.
- **Rejected instructively, logged in Appendix B #4–#8**: a dedicated retire verb (archive stays the commit point; dwell + fuse absorb the accident risk), dropping `MATRIX_ROOMS_PER_GHOST_MAX` (rate/steady-state/stock are different axes), collapsing §15, compressing §16's verified map, and the refuted §14(c) rider.
- **v1 body patched surgically (17 patches)**: header/§3.1 drop `mautrix.crypto` (deprecated libolm, two CVEs, never imported by v1); §8's E2EE closing paragraph → pointer to §16's verified map; §4.4 footnote narrowed to §16 E.3 (crypto store is connector-local Postgres, not secrets); §5.4 dedup rationale corrected (prefix = greppability; the ack-ledger PK is already per-agent); §4.5/§9 Fix 2 softened to the §14(b) structural-gate/`allow_senders` split; §9 Fix 3 annotated (per-verb counts in v1; invite-target caps a named follow-on); M0 gains the retention settings, deactivation tests, and healthcheck; M4 becomes the §14(d) Stage-2 gate list; the Status line now reads "§12–§16 designed, measurements research-gated."
- **Milestone 5 reordered by dependency**: §15 ledger PR first (retirement prerequisite), then spawn/retire automation + prune family, then lifecycle/GC as revised, then measure → queue → shard, then the demand-triggered E2EE epic.
