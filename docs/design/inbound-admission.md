# Inbound admission — the fail-closed "who may talk to the agent" gate

Design of record for epic **#1499**. Status: **v1 shipped and live** (spine
#1500, operator surface #1501, workflow-child bind guard #1502, per-counterparty
budget #1504; `RequireApproval` + grants ledger #1503; group grain #1505 decided
**room-grain**). This document is the durable copy of the settled design;
the code is authoritative where the two disagree, and
`tests/unit/test_inbound_admission_design_doc_drift.py` pins the claims below
that can drift (policy kinds, drop statuses, the NULL default).

## 1. Problem

The human connector inbound plane was implicitly **fail-open**: any message a
connector daemon posted for a bound connection was appended to a session and
woke the model, and a `per_chat` connection auto-spawned a fresh session for
any first-contact sender. Routing (`resolve_target_session`) keys purely on
`(connection, chat_id)`; nothing asked *"is this sender allowed to talk to this
agent?"*

## 2. Threat model

The model is **silent by default** — bare assistant text is private monologue
and reaches no channel; delivery happens only through an explicit connector
tool call (`aios/harness/channels.py`). "A stranger can make the bot spam a
room" is therefore not the failure mode, and **mention-gating is not an
admission concern** (the model already sees `chat_type` / `sender_uuid` /
`self_mentioned` metadata and decides whether to respond).

What an unadmitted sender *could* cause:

| Effect | Cost / harm |
|---|---|
| Session auto-spawn (`per_chat`, first contact) | session row + template materialization + ledger row — "anyone gets an agent" |
| Event-log append | their content persisted into a session's context |
| One model inference per message | token spend whether or not the model replies |
| Context injection | enters the model's context |

The injection effect bifurcates by binding mode:

- **`per_chat`** — the stranger gets their own isolated fresh session.
- **`single_session`** — every `chat_id` routes into the one bound session, so
  a stranger's message lands in the **operator's shared session**. This is the
  genuinely dangerous case.

Admission gates all four effects **before any of them happen**.

## 3. The seam

The single human-inbound chokepoint is `handle_inbound`
(`aios/services/inbound.py`), reached via `POST /v1/connectors/runtime/inbound`.
The lifecycle routes (`/runtime/lifecycle`, `/runtime/session-lifecycle`,
`/runtime/chat-lifecycle`) append `kind=lifecycle` control-plane events, not
human `role=user` messages; they are not an admission surface (their
wake-bearing variants are covered by the budget, §7).

`handle_inbound` order:

1. payload cap
2. `get_connection` (account-scoped)
3. archived-connection check → `detached`
4. **admission gate** → `denied_by_policy` / `pending_approval`
5. per-counterparty budget → `rate_limited`
6. `resolve_target_session` (may spawn a `per_chat` session)
7. per-agent budget (single-session bindings only)
8. stage attachments → append-with-dedup → `defer_wake`

The gate sits after the archived check and **before** resolution, so a denial
produces **zero side effects** — no session row, no event append, no wake. It
has `sender` and `connector_metadata` in scope (the resolver does not) and
already holds the loaded `connection`, so policy resolution adds no fetch.

## 4. The non-fatal status constraint (load-bearing)

The connector-http runner's `_is_fatal_inbound_status` treats **401/403** as
fatal: the connector container crash-restarts, killing every connection it
serves. A denial must therefore **never** be 401/403. Admission drops map to
routine, non-fatal statuses via the exhaustive `_inbound_drop_error` match
(`aios/api/routers/connectors.py`):

| `drop_reason` | HTTP | Meaning |
|---|---|---|
| `denied_by_policy` | 422 | sender not admitted by the connection's policy |
| `pending_approval` | 422 | `require_approval` connection; chat held for an operator's audited approval |
| `rate_limited` | 429 | admitted counterparty over its inbound budget |

Each carries `detail.drop_reason` so a caller can tell a denial, a held chat,
a throttle, and a delivery (200 with `appended_event_id`) apart. Regression
tests pin the mapping and `_is_fatal_inbound_status(422|429) is False`.

## 5. The policy model

A discriminated union over `kind` (`aios/models/inbound_policy.py`). Growth
rule, mirroring the triggers `source` / `action` unions: a new admission
behavior is always a **new kind, never a flag**. Illegal states (e.g. "allow
all, plus a list") are unrepresentable.

| kind | Payload | Admits |
|---|---|---|
| `allow_all` | — | everyone (explicit "open" acknowledgement, never the default) |
| `allow_list` | `chat_ids` (min length 1) | `chat_id` ∈ `chat_ids` |
| `allow_senders` | `sender_ids` (min length 1) | the connector-supplied canonical `sender.id` ∈ `sender_ids` (internal-only agents; see the Matrix connector design §14(b)) |
| `require_approval` | `approved` (may be empty) | `chat_id` ∈ `approved` **and** an active audited `inbound_grants` row exists |
| `deny_all` | — | no one (explicit fail-closed) |

Storage: an additive, nullable `connections.inbound_policy jsonb` column
(migration `0121`). **NULL resolves to the server default `deny_all`**
(`effective_inbound_policy`). The pydantic union is the write-path validator
(no DB CHECK on shape). The column is folded onto the `Connection` read model as
`inbound_policy` (stored, nullable) and `inbound_policy_effective` (derived,
read-only), so resolution adds zero round-trips.

An **empty `allow_list` is rejected at write time (422)** — to deny everyone,
pick `deny_all`; an empty list is never a silent deny-all.

A stored policy that no longer parses (hand-edited jsonb, a rolled-back deploy
writing an unknown kind) fails **closed** as a 422
(`malformed_inbound_policy`), not a 500.

The gate is a pure, exhaustive `match` (`_admits` in `aios/services/inbound.py`).

### Why `chat_id` (room-grain)

DM ⇒ `chat_id` *is* the person; group ⇒ `chat_id` is the room, and admitting it
trusts the room's membership (the model still behaves per-member via
`sender_uuid` / `self_mentioned`). `sender.display_name` is **never** consulted
— it is the untrusted, self-reported `from=` clause. #1505 settled room-grain
for v1; per-member admission exists only as the explicit `allow_senders` kind
for connectors that surface a strong canonical sender id.

**Confused-deputy residual.** The runtime token authorizes a *connection*, not
a `chat_id`: a compromised in-scope connector daemon can assert any admitted
`chat_id`. Containment is per-connection-scoped runtime tokens (the
`connection_ids` allowlist on the runtime token) plus rotation — the admission
policy defends against the unauthenticated stranger, not a compromised deputy.

## 6. Decisions locked

1. **Cutover = backfill to known chats.** Migration `0121` backfilled every
   existing connection to `allow_list(<known chats>)`, the **union** of (a) the
   connection's `chat_sessions` ledger rows (verbatim) and (b) the `chat_id`
   parsed from its historical `role=user` events' channel. A connection with
   no history stayed NULL → `deny_all`. The parse takes the **whole remainder**
   after the `"{connector}/{external_account_id}/"` prefix (never
   `split_part(channel,'/',3)`, which truncates a slash-bearing `chat_id` such
   as a Signal group id) and is **prefix-scoped** to this connection's own
   `connector` / `external_account_id` with LIKE metacharacters escaped (a
   session can carry events from several connections). Verified against live
   prod data before the flip: zero lockouts.
2. **v1 is connection-only.** NULL resolves straight to `deny_all` — no
   account-wide default and no `parent_account_id` inheritance walk (a parent
   must never be able to widen a child's fail-closed connection). An
   account-wide, narrowing-only, non-walking default is deferred.
3. **Kind-not-flag** policy union.
4. **Gate on `chat_id`** (room-grain), per §5.

## 7. Operator surface and guards

- `PUT /v1/connections/{id}/inbound-policy` (operator bearer, **not** the
  runtime token) — **Replace** semantics: the body is the bare union member;
  `{"kind":"allow_list"}` without `chat_ids` 422s rather than silently
  widening, and revocation is a Replace with the smaller list. Archived
  connections are refused (404). CLI: `aios connections set-inbound-policy
  <id> --kind allow_list --chat-id <id> …`.
- `inbound_policy_effective` is echoed on connection create / get / list, so
  the posture is visible without a second call. `connections recent-chats` /
  `bound-chats` surface the authoritative `chat_id`s to allowlist.
- **Default-closed at bind.** `attach` and `configure-per-chat` set
  `deny_all` when the column is NULL (never clobbering an operator-set
  policy), so a freshly bound connection admits nobody until opened.
- **Workflow-child bind guard (#1502).** Both operator bind sites —
  `attach_connection` and `bind_chat_to_session` — reject a target session with
  `parent_run_id IS NOT NULL` (409), so a connector can never route into a
  workflow child's run-attenuated, surface-frozen session.
- **`require_approval` grants (#1503).** A denied chat on a
  `require_approval` connection registers an audited *pending*
  `inbound_grants` row and drops with `pending_approval`. Operators use
  `POST …/inbound-grants/approve|revoke` and `GET …/inbound-grants/pending`
  (CLI `connections approve|revoke|list-pending`). Admission requires **both**
  the `approved` mirror entry and an active grant row, so neither a
  hand-rolled policy PUT nor a stale grant can admit on its own; a torn write
  fails closed. Stale pending rows are reaped by the worker
  (`inbound_grants_pending_ttl_seconds`).
- **Per-counterparty budget (#1504).** After admission, a rolling count per
  `(account, connector, external_account_id, chat_id)` over
  `inbound_rate_window_seconds` caps inference-bearing inbounds (and
  wake-bearing lifecycle calls) at `inbound_rate_max_per_window`; a per-agent
  session-keyed budget (`inbound_rate_agent_*`) bounds Sybil fan-in on
  single-session bindings. All knobs default to `0` (disabled, no query).

## 8. Non-goals

- Mention-gating as a core mechanism.
- Prompt-injection defense against *admitted* senders (the surface-lattice /
  sandbox containment layer).
- Connector `chat_id` canonicalization.
- Account-wide default policy (deferred; narrowing-only, non-walking).
- The triggers ingest plane (`POST /v1/triggers/ingest/{ingest_token}`) — a
  separate surface, fail-closed by its per-trigger secret.
- The agent-to-agent caller-edge reachability plane.
