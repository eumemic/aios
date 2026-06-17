# aios vs. NousResearch/hermes-agent — competitive recon

*Generated 2026-06-17 via `/competitive-recon` against `NousResearch/hermes-agent` at the then-current `main`. Two multi-agent workflows (compare: 35 agents / 4.19 M tokens; primitive-decomposition: 17 agents / 1.62 M tokens), every hermes-side claim adversarially re-verified against source. Self-contained — every adopt-item carries its primitive lineage and edit points inline.*

---

## Headline

> **0 new core primitives.** Every hermes capability worth taking decomposes into aios's existing ~13 primitives: **7 extends** (a new arm/kind on an existing discriminated union), **4 compositions**, **3 not-core**. The minimal-primitive principle held exactly — same verdict shape as the eve run, reached independently.

The sharper finding: **hermes leads on capability breadth and per-user ergonomics; aios leads on every structural axis** (durability, multi-tenancy, attenuation, correct-by-construction invariants). Almost nothing in hermes's *mechanisms* is worth adopting — its in-process hooks, plaintext credential files, LLM-summarization compactor, blocking sync loop, and one-process-many-platforms gateway each directly contradict an aios load-bearing stance and would be regressions. What's worth stealing is a small set of **capabilities** hermes proves are valuable, re-expressed as compositions/extensions of primitives aios already owns.

## Product models (the frame for every "who's better" call)

- **hermes-agent** — a maximalist, local-first, single-user **personal AI agent**: one synchronous agent core across CLI/TUI/Electron/messaging-gateway (~28 platforms), BYO-model across dozens of providers, "self-improving" (curates its own memory, writes its own skills). ~534 K LOC. State = single-file per-user SQLite. Survives long context via LLM compaction.
- **aios** — an event-driven, Postgres-backed, **multi-tenant hosted runtime** built from ~13 composable primitives, headline = the async "step model" (no loop). ~74 K LOC. State = append-only event log. Rejects LLM compaction for deterministic windowing.

7× LOC, opposite philosophies. **hermes optimizes one power-user's reach across everything; aios optimizes a correct-by-construction tenant-isolated kernel.** Judgments below are fitness-for-purpose, never feature-count.

---

## Scorecard (16 dimensions, adopt-value to aios)

| Dimension | Adopt value | Crux |
|---|---|---|
| Memory + user modeling | **HIGH** | hermes pushes a modeled user (Honcho, rented); aios owns a versioned-FS substrate with zero intelligence over it |
| Self-improvement / learning loop | **HIGH** | hermes forks a curator that writes its own skills; aios's agent literally cannot author a skill today |
| Multi-agent delegation | **HIGH** | hermes's one-line `delegate_task(tasks=[…])` ergonomics vs aios's durable+attenuated but DSL-heavy workflows |
| Context compaction *(philosophy clash)* | MEDIUM | hermes LLM-summarizes in place (lossy, non-deterministic); aios windows deterministically — the cheap *no-LLM* half is adoptable |
| Tools / dispatch | MEDIUM | hermes sync-batch blocks the loop; aios implicit-async. Adopt: AST auto-discovery + arg-aware gating |
| Extensions | MEDIUM | hermes 19 in-process hooks w/ host authority (reject the framework); adopt only the pre-tool guardrail *capability* |
| Scheduling | MEDIUM | hermes file-cron polled by the gateway; aios `triggers` union leads — but lacks direct-to-channel delivery |
| Credentials | MEDIUM | different problems: hermes multi-account *rotation*, aios encrypted *vaults*. Adopt rotation as a vault generalization |
| State persistence | MEDIUM | mutable SQLite+FTS5 vs append-only log. Adopt: native Postgres FTS over the log |
| Tenancy | MEDIUM | aios leads decisively; adopt-from-them only the pairing admission UX |
| Human-in-loop | MEDIUM | aios's unified no-park wake leads; adopt a hardline `deny` floor + arg-match escalation |
| Execution loop | LOW | aios async-step strictly stronger for a runtime; adopt only a turn-count circuit-breaker |
| Channels / serve | LOW | aios out-of-process per-connection isolation leads; adopt connector capability flags |
| Model abstraction | LOW | aios's LiteLLM delegation is correct; adopt only a declarative quirk descriptor |
| Sandboxing | LOW | aios isolation *depth* leads; the seam already exists — lift the TCB onto it |
| UI | LOW | product-posture context, not a defect; optional PTY-over-WS console terminal |

---

## What to steal — tiered adopt-list (14 items)

Each is annotated with its **primitive lineage** (the Phase-4 verdict). None adds a new core primitive.

### NOW (highest ROI)

1. **Attenuated agent-callable skill mutation** *(extends Tool)* — the single largest genuine gap. Two new `BuiltinToolType` arms (`skill_upsert`/`skill_archive`) as agent-tool-transport wrappers over the *existing* `services/skills.py`, mirroring the shipped `workflow_management` "strange-loop" builtins (harness supplies `session_id`/`account_id`, never the model). Plus a write-through fix to `provision_skill_files` so the DB stays single-source-of-truth. **This closes the self-improvement loop**: periodic curation becomes a pure composition (cron-trigger → workflow → `skill_upsert`), deterministic + attenuated + journaled. Steal hermes's two-tier *decomposition* (cheap per-turn capture vs. periodic consolidation), its active→stale→archived lifecycle, and its prompt content (umbrella-over-micro-skills; the anti-capture list — don't persist env failures or "hardened refusals"). Do **not** adopt the flat-file state, `max_iterations=9999` fork, or the ~470-line reconciliation layer (the journal *is* the authoritative record).
2. **Memory intelligence layer over the versioned-FS substrate** *(composition)* — three separable pieces, no new primitive: (a) a typed user-profile store whose distilled content the harness **auto-injects each wake** (one more `augment_*` arm in the `step_context.py` chain — push-recall, closing the "agent forgets to `rg`" silent-failure gap); (b) a Postgres **tsvector-GIN** (later pgvector) index over `memories.content` as a `memory_search` builtin; (c) a periodic **deterministic distillation workflow** that regenerates a compact "card" from *immutable* raw memory (NOT in-place LLM rewrite). Memory is the stated moat; aios owns a correct substrate but ships zero intelligence over it while hermes ships intelligence it rents and can't share.
3. **`deny` arm + arg-match escalation** *(extends Tool)* — widen `PermissionPolicy` from the 2-arm `Literal['always_allow','always_ask']` to add `deny` as the new ⊥ of the `_permission_meet` lattice — an unconditional catastrophic floor that composes through every #794 edge by associativity. Plus a small declarative per-tool risk-predicate set that bumps `always_allow→always_ask` on match. Buys hermes's hardline floor (`rm -rf /`, `mkfs`, fork-bomb — blocked below any YOLO) + selective escalation **without** its 61-regex graveyard or blocking threads. Closes aios's one real approval gap: today an `always_allow` bash runs `rm -rf /` ungated.

### SOON

4. **Lift the isolation TCB onto the SandboxBackend Protocol** *(extends Sandbox)* — the headline recon correction: aios **already has** the `@runtime_checkable` Protocol seam (13 verbs, not 6 — the docstring says "five" and is stale, fix it). The borrow is to hoist the one off-seam reach-around (`network.py::ensure_sandbox_network` calls `run_docker_cli` directly) onto the seam and make `worker.py`'s hardcoded `DockerBackend()` a settings-driven factory, so a second backend (Firecracker/cloud) inherits secrets-never-in-container. **Refactor onto the existing seam now** (correctness-bearing); add a second backend only when a real product need lands.
5. **No-model `deliver`/`notify` trigger action** *(extends the deliver kernel — `Tell{target=Channel}`)* — post a trigger's `sandbox_command` stdout (or a static message) **out to the owner's bound connector without a model wake**. Today `wake_owner` always costs a full model turn and `sandbox_command` can't reach a channel — so "poll an API, ping me if changed" costs a model wake every fire. The recon's sharpest insight: hermes's "direct delivery" and "script watchdog" are the **same** missing capability — a cheap model-free path from a fired trigger to a user-visible message — and it's one action arm. *(One open design fork — see below.)*
6. **IANA-timezone-aware cron** *(extends Triggers)* — build the `timezone` field the `CronSource` docstring **literally already reserves** (absent⇒UTC). Zero migration (`source_spec` is opaque jsonb). Cron is UTC-only today, so `0 9 * * *` fires at 9am UTC, not the user's 9am.
7. **Lightweight result-returning `invoke_agent` session tool** *(composition — IS the #1127 deliver-kernel porcelain)* — a model-callable fan-out-and-await-inline delegate for the common case, reusing the safety substrate: each child a `create_child_session` with the **same attenuation clamp** + #823 identity clamp, correlated via `request_id`, resolved via the kind-agnostic resolver. Durability tier as a discriminated KIND (ephemeral child-session vs. journaled run) = exactly the deliver-kernel `NewSession|NewRun` target union. Closes the single biggest *ergonomic* loss vs. hermes's one-line `delegate_task`.
8. **Postgres FTS/trigram over the event log** *(composition)* — a GIN-indexed generated `tsvector` column over `events.data->>'content'`, exposed through the existing `events_search` view as ranked `websearch_to_tsquery` + `ts_rank` + `ts_headline` snippets (pg_trgm for CJK). Purely additive. Today `events_search` is an admitted ILIKE-on-serialized-content seq-scan hack.

### SOMEDAY

9. **Deterministic in-window stale-tool-output reducer** *(composition — Context builder alone)* — the cheap *no-LLM* half of hermes's compaction as a pure `f(seq)`: md5-dedup repeated reads (keep newest), collapse stale output to a one-line synopsis, JSON-safe-truncate oversized args, strip historical base64 — byte-stable within a snap chunk exactly like the shipped `_quarantine_placeholder`/`_omission_marker`. Efficiency, not correctness (windowing drops these eventually).
10. **Pairing-style admission UX** *(extends Connection)* — a one-time-code owner-approves flow as an `admission_state` (pending|approved|denied) on the per-chat routing ledger, gating inbound *before* spawn. Nicer than editing env-var allowlists to admit a new chat principal.
11. **Postgres RLS as a DB-level backstop** *(not-core — ⚠ needs explicit sign-off)* — keyed on a per-tx `account_id` GUC; the `search_events` `set_config('app.session_id')`+`current_setting` view is the exact in-tree precedent. **Flagged per the no-belt-and-suspenders rule**: it sits behind already-correct query-layer enforcement and only helps when a hand-written query *forgets* the clause — a conditional backstop, not correct-by-construction. Chairman's call; if approved, the global-table exemptions + parent-mints-child management paths are the load-bearing policy DDL.
12. **Connector capability descriptor** *(extends Connection)* — typed `capabilities` field on the root-owned connectors catalog row (sibling of `tools_schema`), so shared code branches on declared capability (`supports_draft_streaming`, `native_buttons`) not `if connector == 'slack'`. The native-button half reuses the shipped `always_ask` dispatch KIND + the existing tool-confirmation endpoint verbatim.
13. **Declarative provider/model descriptor** *(not-core)* — fold the ~5 scattered `_supports_*` LiteLLM sniff sites into one `@cache`'d `ModelDescriptor`. A kaizen/broken-windows tidy, no authority/state/family. Keep it a flat lookup (no plugin registry).
14. **PTY-over-WebSocket console terminal** *(not-core)* — expose the existing `aios chat` REPL over a `/sessions/:id/pty` WS for the console, zero new client code; a transport re-wrap of the shipped SSE composition.

### Explicitly REJECTED (recorded as deliberate divergences, not oversights)

- **LLM-summarization compaction** — model call on the hot path, new failure mode, non-deterministic (breaks replay), prompt-cache-hostile, lossy-irreversible. Direct conflict with the monotonic-context invariant and the user's recorded preference. Its ~2,400 lines are mostly failure-handling aios's architecture makes unnecessary. *(The cheap no-LLM half is captured separately as #9.)*
- **In-process plugin framework + 19 lifecycle hooks** — in-process Python in module-global dicts running with full host authority is a confused-deputy/cross-tenant-blast-radius anti-pattern in a multi-tenant runtime. The one valuable capability inside (an enforced pre-tool guardrail) is captured correct-by-construction by the `deny` arm (#3).
- **Provider-subscription OAuth in core + native non-OpenAI transports** — ~11.2 K LOC of hand-rolled adapters + keychain OAuth would balloon the TCB and import ToS/endpoint-drift risk. Keep LiteLLM delegation + subscription-OAuth externalized to sidecars behind an api_base. *One positive action:* document that sidecar boundary as a stated decision.

---

## The keystone — primitive-gap map (0 new / 7 extends / 4 compositions / 3 not-core)

The Phase-4 skeptic **demoted the draft's single "new primitive"** (the no-model deliver action) to an extends. It verified the *capability* is irreducible — there is genuinely no worker→channel producer today (the outbound stream is sourced only from model-emitted tool_calls; `wake_owner` delivers a user-message-into-session, not a channel send) — but irreducibility-of-capability ≠ new-top-level-primitive: a new `Tell` *target* is an extends of the deliver kernel, the exact sibling of "new trigger action-kind = extends." That correction is what tightened the count to **0**.

| Item | Verdict | Rides on |
|---|---|---|
| skill-mutation | extends | **Tool** (new BuiltinToolType arms, strange-loop pattern) |
| deny-arm | extends | **Tool** (third arm on the permission discriminator = new ⊥) |
| isolation-tcb | extends | **Sandbox** (completes the existing 13-verb Protocol seam) |
| tz-cron | extends | **Triggers** (additive field the docstring reserves) |
| pairing-admission | extends | **Connection** (admission_state on the routing ledger) |
| connector-capability | extends | **Connection** (typed field on the catalog row) |
| no-model-deliver | extends | **deliver kernel** (`Tell{target=Channel}`) |
| memory-intelligence | composition | Context-builder ∘ Tool ∘ Triggers ∘ Workflow |
| invoke-agent | composition | Request-edge ∘ attenuation ∘ stimulate-spine (= #1127) |
| events-fts | composition | Event-log ∘ Tool (additive index) |
| stale-tool-reducer | composition | Context-builder (pure f(seq)) |
| rls-backstop | not-core | DB-policy sidecar (⚠ explicit sign-off) |
| provider-descriptor | not-core | local refactor inside completion.py |
| pty-console | not-core | transport re-wrap in api/ + console BFF |

**Three collapses** sharpen the map: the three connector items are the **Connection primitive growing along inbound/outbound/catalog axes** (only the outbound *direction* lacked a producer — which is *why* the "new thing" is one missing direction of an existing primitive); `invoke_agent` + `memory-loop` are **deliver-kernel/trigger policy surfaces**; `deny` + `skill-mutation` are **Tool kind-growth**.

---

## Where aios already leads (the honest counterweight, 13 areas)

All downstream of **one** decision repeated at every layer: *state lives in a durable, append-only, account-scoped Postgres substrate, and behavior is a correct-by-construction function of that log rather than a property maintained by in-memory vigilance.*

- **Execution** — async-step-over-event-log: every tool implicitly async, every crash recoverable. hermes's blocking in-memory loop forfeits both.
- **Long-context** — deterministic windowing + queryable full log + separate memory primitive. hermes's 2,400-line compactor (7 cited bug-branches) is the receipt for the trade aios refuses.
- **Tool dispatch** — one unified wake spine; ghost-repair across SIGKILL; typed evict-vs-refuse errors. hermes blocks the loop and loses in-flight results on exit.
- **Extension model** — closed runtime, data-or-protocol-only. hermes's 19 host-authority hooks are a confused-deputy surface in a multi-tenant runtime.
- **Memory substrate** — one versioned, account-shareable, auditable FS ("what did memory say last week and who changed it" — hermes can't answer).
- **Scheduling** — multi-worker `SKIP LOCKED` claim (hermes's node-local lock double-fires across hosts); sub-second event-driven latency vs. 60s poll; per-fire audit.
- **Connectivity** — out-of-process, per-connection-isolated, multi-tenant connectors. hermes runs ~28 platforms in one process where one reconnect storm degrades all.
- **State storage** — single-writer gapless log; derived state can't drift. hermes fights write-lock convoys on one shared SQLite file.
- **Tenancy** — `account_id` threaded correct-by-construction + a real delegation lattice. hermes has no tenant column at all.
- **Approval** — unified externally-executed-tool primitive, session never parks, durable+auditable. hermes blocks a thread-per-surface, state in process memory only.
- **Sandbox depth** — secrets-never-in-container + two-layer allowlisted egress + seccomp + durable full-rootfs snapshots + DNS-rebind defeat. hermes hands `-e KEY=VALUE` plaintext over an open network.
- **Orchestration** — durable replay-from-memo + associative attenuation meet + credential-free out-of-process host. hermes's `delegate_task` is in-memory state lost on parent crash.
- **UI factoring** — decoupled clients over a CI-enforced OpenAPI contract. hermes couples internal agent verbs to 3 hand-built frontends (85 RPC methods).

> The verdict is not "aios is better." It's the sharper claim the recon supports: **for a durable, multi-tenant, security-bearing cloud runtime, aios's substrate-first architecture is the correct foundation and hermes's would have to be rewritten to reach it — while for hermes's own single-operator personal-tool product, its loop-and-files design is fit-for-purpose and aios's infrastructure would be dead weight.** The places aios trails are almost all *ergonomics-and-reach* gaps, and the recon shows most are additive compositions over the substrate aios already owns.

## Greenfield — if aios were rebuilt today

**Keep all 13 foundational decisions** (step-over-log, implicit-async tools, correct-by-construction invariants, append-only log, deterministic windowing, closed two-primitive extension model, attenuation lattice, encrypted vaults + secret-egress, multi-tenancy-by-construction, credential-free workflow host, triggers union, out-of-process connectors, LiteLLM delegation). None is regretted under a from-scratch rebuild.

**Reconsider 8** — and notably every one is an *additive composition over the existing substrate*, not an architectural reversal: turn-progress circuit-breaker · in-window tool-result reduction · `deny`/risk arm · cheap `invoke_agent` · native FTS · RLS backstop (sign-off) · provider descriptor · per-call task-fan-out concurrency ceiling. (The last — a per-session semaphore over `_launch_tasks`, which today does one `create_task` per call unconditionally — surfaced in greenfield but wasn't in the adopt-list; worth a separate look as a latent blast-radius hazard.)

---

## Open design forks (for the owners, before any filing)

1. **`Tell{target=Channel}` legitimacy** *(lead fork)* — the kernel's three Tell targets are all *principals* that can be a caller/servicer; a connector channel is a non-principal *egress* with no awaited edge. Either (a) Channel is a 4th Tell-target arm, or (b) the worker-outbound producer is a sibling connector-outbound service helper the trigger action consumes directly, kernel untouched. **Both are extends-of-an-edge, neither is a new primitive** — but (a) vs (b) is a genuine fork the kernel owner should resolve first (per #1165 sequencing).
2. **`deny`-arm uniformity** — must be admitted in all three permission meets (builtin/MCP/http) or it's non-uniform; surface to the model as a tool-error (model-consciousness), not a silent drop.
3. **pairing-admission home** — admission must be decidable *before* a session is spawned, so `admission_state` likely needs its own `(connection,chat_id)`-keyed table, not a `chat_sessions` column.
4. **events-FTS scope** — confirm per-session (the existing `app.session_id` GUC) vs. cross-account; a wider read surface is the one variant that touches the attenuation edge.
5. **RLS sign-off** — explicit chairman approval required; global-table exemptions + parent-mints-child admission are the load-bearing policy DDL.

---

## Filing recommendation

The 11 in-core items (7 extends + 4 compositions) are **shovel-ready-shaped** — each has a verified primitive lineage, named edit points, and a deterministic acceptance story — *except* where an open fork above gates it (#5 deliver, #10 pairing → `needs-design` until the fork resolves). The 3 not-core items: #11 RLS needs sign-off; #13 provider-descriptor is a kaizen tidy; #14 PTY is console-BFF work. Several map onto in-flight epics (the deliver kernel #1122/#1165/#1127, triggers #818/#819, durable sandboxes #916) and should be filed as **issues that depend-on / are sub-issues-of** those, not standalone — to avoid the eve-run's relink churn.

**This report is the durability anchor**: issues link a committed `docs/design/` copy of it (absolute `blob/master` URL), never an ephemeral scratch path.

## Filed as

Filed 2026-06-17 (each issue links back here):

| Adopt | Issue | Ratification |
|---|---|---|
| #2 memory intelligence (epic) | **#1370** → subs #1371 (profile auto-inject, needs-design), #1372 (`memory_search`, shovel-ready), #1373 (distillation, needs-design) | epic |
| #1 agent-callable skill mutation | **#1374** | shovel-ready |
| #3 `deny` arm + arg-match escalation | **#1375** | shovel-ready |
| #4 lift isolation TCB onto the backend seam | **#1376** (depends-on #1347) | shovel-ready |
| #5 no-model channel deliver | **#1377** (relates #1197/#1165/#1335/#1342) | needs-design |
| #6 IANA-timezone cron | **#1378** | shovel-ready |
| #8 Postgres FTS over the event log | **#1379** | shovel-ready |
| #10 connector inbound admission (pairing) | **#1380** (depends-on #462) | needs-design |
| #12 connector capability descriptor | **#1381** | shovel-ready |
| #11 RLS DB backstop | **#1382** | needs-design + needs-decision (sign-off) |
| #13 declarative provider/model descriptor | **#1383** | shovel-ready |
| #7 result-returning `invoke_agent` | *already filed* — **#1127** (in-progress) | — |
| #9 in-window stale-tool-output reducer | *broadens existing* — **#1359** (commented) | — |
| #14 PTY-over-WS console terminal | *not filed* — aios-console repo work | — |

Rejects (#15 LLM compaction, #16 in-process hook framework, #17 provider-OAuth-in-core) are recorded above as deliberate divergences, not filed as work.
