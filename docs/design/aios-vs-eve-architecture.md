# aios vs. vercel/eve — an architectural comparison

*2026-06-17. Source-grounded across 14 dimensions by a multi-agent research workflow (14 analysts → 14 adversarial verifiers → 3 synthesizers); every eve-side claim re-checked against the upstream `vercel/eve` source, every aios claim against this repo + the `aios-console` repo. All 14 dimensions returned at high confidence with only minor corrections.*

> **Companion:** [`eve-adoption-primitive-decomposition.md`](eve-adoption-primitive-decomposition.md) decomposes the §2 adopt-list into aios primitives (verdict: **0 new core primitives**). The adopt-items are filed as issues **#1335–#1362** plus a review-finding on PR #1314.

---

## 0. The one thing to internalize first

**These are not two implementations of the same thing. They are two different products that happen to share a problem domain.**

- **eve** is a *deploy-your-own-app framework*. You author an agent as TypeScript files in your own repo, `vercel deploy` it, and it becomes **your** single-tenant app. Durability, sandboxing, cron, model credentials, and the trace store are all **delegated to the Vercel platform** (or a self-host substitute). eve's genius is *authoring ergonomics* and *leverage*: it writes as little infrastructure as possible and leans on a platform for the hard parts.

- **aios** is a *multi-tenant hosted runtime*. An agent is a row in **your** Postgres, minted over an API by any client, serving many accounts with cryptographic isolation. aios **owns** its durability, sandbox hardening, scheduler, credential brokering, and audit log. aios's genius is *correctness-by-construction* and *operator control*: every invariant is auditable in-repo and nothing load-bearing is hostage to a third party.

Almost every pro/con below is a *consequence* of this split. eve gets a beautiful authoring loop and tiny infra surface **because** it serves one tenant and rents the hard parts. aios gets real multi-tenancy, a lossless queryable substrate, and an always-responsive agent **because** it pays to own everything. Reading any single dimension as "X is better" without that frame is a category error — so each verdict below is "better *for its product model*."

The interesting question your prompt asks isn't "who wins" — it's **"what has eve discovered that aios could adopt *without* giving up the multi-tenant, owned-substrate spine?"** That's where the real signal is.

---

## 1. Dimension scorecard

| # | Dimension | Who leads (for its purpose) | eve's edge | aios's edge | Adopt value |
|---|-----------|------------------------------|------------|-------------|-------------|
| 1 | Authoring & dev experience | **eve** (by a lot) | filesystem-as-spec, `defineTool` end-to-end types, compile diagnostics, `npx eve init` + live TUI | definition-is-data → any client mints/mutates; immutable `agent_versions`; rich policy/attenuation surface | **Medium** |
| 2 | Execution & durability | **tie / different bets** | durability is a *platform primitive* (Workflow SDK), tiny recovery code, versioned cross-deploy session migration | owns the substrate; ghost-repair side-effect safety; single-source wake predicates with a drift-guard test | Low |
| 3 | Async tool & concurrency | **aios** | atomic replayable step → trivial recovery; no dedup/ghost machinery needed | **model stays responsive while tools run; mid-tool user interleaving** (eve blocks the run for the tool step) | Low |
| 4 | Context management | **aios** | LLM compaction → unbounded effective length; **reactive tool-result pruning** | lossless immutable-log windowing; prefix-cache stable; `search_events` recall; **zero hot-path model calls** | **Medium** |
| 5 | State & persistence | **aios** | zero-infra, no DB to run; small read-time **version-chain migrator** | SQL-queryable lossless source of truth; correct-by-construction derived status | **Medium** |
| 6 | Sandboxing | **aios** (depth) / **eve** (breadth) | 4 backends incl. real microVM isolation; **template prewarm**; declarative mid-turn cred brokering | deep hardening on its one backend (netns-sidecar, seccomp, egress CA); durable FS + crash-salvage | **Medium** |
| 7 | Tools, skills & subagents | **eve** (ergonomics) / **aios** (safety) | `defineTool` typed `execute`; **subagent = model-callable delegation tool**; per-tool auth/approval | transport frontier; **#794 attenuation lattice**; MCP-as-extension; durable workflow runs | **High** |
| 8 | Channels & integrations | **split** | bigger catalog; **interactive OAuth**; event-stream outbound (typing/streaming) | persisted multi-tenant connection records; **focal-channel** abstraction; process-isolated connectors | **High** |
| 9 | Triggers & scheduling | **aios** | correct-by-delegation (platform cron); schedules are version-controlled files; random-path ingress | runtime-mutable + **agent-authored**; `trigger_runs` audit; reactive sources; safe multi-worker firing | **Medium** |
| 10 | Human-in-the-loop | **aios** | durable park; unified approval+question protocol; per-input `needsApproval`; `once()` | **no-park → agent stays live**; operator-scoped approval (segregation of duties); queryable `awaiting` | **Medium** |
| 11 | Model abstraction | **split** | **error-class-aware retry**; runtime-owned routing/fallback; provider-constrained structured output | vendor-neutral (LiteLLM); open `litellm_extra` escape hatch; hard per-call deadlines | **High** |
| 12 | Observability & evals | **split** | **OTel export**; **built-in behavioral eval + LLM-judge framework** | event-log = durable SQL audit/trace store, zero infra; Procrastinate-aware profiler; per-call cost | **High** |
| 13 | Multi-tenancy & ops | **aios** (it's the whole point) | `vercel deploy`, nothing to operate | real account isolation; per-account HKDF crypto; owned migrations; worktree dev isolation | Low |
| 14 | Client/UI surface | **split** | **headless embeddable client** (`useEveAgent`, cross-framework) | first-party **operator console** (full CRUD, 223 files); open REST/SSE contract | **Medium** |

---

## 2. Where eve is genuinely ahead — the adopt-list (prioritized)

These are ideas aios could take **without** surrendering its load-bearing philosophy. Ordered by ROI.

### Tier 1 — do these

1. **Error-class-aware model retry** *(model abstraction; small effort, low risk, high ROI — the single clearest place aios is just *worse*)*.
   Today aios's retry is error-class-**blind**: a terminal `400 invalid-prompt` or `ContextWindowExceeded` hits the same generic `except Exception` and the same fixed `[2,8,30,120]s` backoff as a transient `429`, only failing after burning the whole budget — **~160s + up to 4 doomed prompt-token round-trips** on an error that will never succeed (`harness/loop.py:592`). Port the *shape* of eve's `classifyModelCallError` (`model-call-error.ts:291`): branch on LiteLLM's already-typed exceptions — `RateLimitError/APIConnectionError/InternalServerError/Timeout` → retry; `AuthenticationError/BadRequestError/ContextWindowExceededError/ContentPolicyViolationError` → fail fast. No new dependency. This is the highest-ROI single change in the whole report.

2. **A behavioral eval + LLM-judge framework** *(observability; medium effort, high ROI — a category aios lacks entirely)*.
   aios can assert harness *mechanics* (e2e) but cannot assert agent *behavior* ("called search before read", "never called the refund tool", "the answer is factually correct") as a scored CI gate — so prompt/model changes ship on manual smoke. This is a near-perfect fit because aios already has (a) a wire-driven session driver (the CLI sends a turn, streams events) and (b) a **durable event log far richer than eve's in-memory capture**. eve's `evals/assertions/run.ts` + `runner/derive-run-facts.ts` + `runner/verdict.ts` are a directly transcribable blueprint; assert over the *persisted log* instead of an ephemeral stream, route the judge through the existing LiteLLM seam.

3. **Schema-first tool authoring** *(tools; small-medium effort, high ROI, zero architectural cost)*.
   Every built-in hand-writes a raw JSON-Schema dict **and** a separate handler doing `arguments.get(...)` + manual `isinstance` checks (`tools/bash.py:84-135`); schema and code can silently drift. Adopt a `defineTool`-style decorator that derives `parameters_schema` from a Pydantic input model and types the handler — **the pattern already exists inside aios's own `workflow_management` builtins** (`model_json_schema()`, `workflow_management.py:473`); it's just not generalized to the filesystem/network built-ins. Pure DX/correctness win.

4. **Interactive OAuth / connection-authorization for connectors** *(channels; medium effort, high ROI — aios's biggest connector gap)*.
   aios connector credentials are an operator-set encrypted blob (`ConnectionSetSecrets`, replace-wholesale) — there is **no** user-delegated "sign in to your Google" consent flow. eve's `startAuthorization`/`completeAuthorization` (`runtime/connections/types.ts`) suspends on a framework-owned callback, runs consent, resumes with a journaled token, and caches per-principal. aios already has the durable wake/park mechanism and per-connection encrypted secrets — this closes the clearest functional hole, and pairs naturally with the existing `aios-mcp` orchestration direction.

5. **Model-initiated subagent delegation as a first-class session tool** *(tools; medium effort, high ROI — closes the biggest functional gap)*.
   In eve a subagent is an ordinary model-callable delegation tool (`{message, outputSchema}`, dispatched through the normal tool loop). In aios the **session-level model has no subagent tool at all** — delegation only exists as operator-mediated durable workflow runs; the clean `agent()`/`parallel()`/`pipeline()` fan-out is walled inside out-of-process workflow scripts. Expose a built-in `delegate`/`spawn_agent` that spawns a child session **clamped through the existing #794 attenuation meet** (`workflows/step.py:954`) — eve's ergonomics, made multi-tenant-safe by a property eve lacks.

### Tier 2 — strong, do soon

6. **In-window deterministic tool-result pruning** *(context; medium effort)*. eve's single highest-value context idea (`harness/tool-result-pruning.ts`): collapse old large tool-result events to a recall-able placeholder when it saves meaningful tokens, instead of only dropping whole events at the window boundary. The proof eve hands you: **the tool-result sink, not the conversation, is what actually overflows real sessions.** In aios this is *strictly less lossy* — the placeholder points at `search_events` so the model can recover the exact output (eve has no recall). **Critical constraint:** make the prune decision a pure function of the immutable log (`f(seq boundary)`, *not* live token estimates) to preserve monotonicity and prefix-cache stability — exactly the property eve's version sacrifices and aios is positioned to keep.

7. **Explicit serialized-payload version contract** *(state; medium effort)*. aios evolves event-payload shapes by *implicit forward-tolerance scattered through `harness/context.py`* — there is no `schema_version` inside a stored event's `data` and no single migrator. Adopt eve's `runMigrationChain` (~80-line pure runner: one-step migrators, hard-fail on gaps/future-versions): stamp `schema_version` into event `data` and walk old rows forward **at read time** in one auditable place. Composes cleanly with the immutable log (migrate-on-read, never rewrite). Addresses aios's single weakest spot on persistence.

8. **OTel export adapter** *(observability; small-medium effort)*. A thin exporter mapping aios's existing `span`-kind events to OTLP unlocks every off-the-shelf trace UI + cross-service correlation, while the event log stays the source of truth and the bespoke profiler stays a fast-path consumer. Design the `span` schema as OTLP-projectable (stable `trace_id/span_id/parent_span_id`) so it's an adapter, not a rewrite.

9. **Per-input approval predicate + approve-once** *(HITL; small effort)*. Generalize the existing `classify_permission` hook (today only wired for `http_request`) into a first-class `needs_confirm if predicate(args, agent)` on any tool spec, so an owner can express eve's `amount>1000 → always_ask` without a bespoke classifier — the plumbing already evaluates per-call (`loop.py:921-933`); only the authoring surface is missing. Add an optional session-scoped approve-once memo to kill approval fatigue (eve's `once()`).

10. **Event-stream-driven outbound for connectors** *(channels; medium effort)*. eve's channels react to the harness event stream, so typing indicators / streaming partials / in-place edits fall out for free. aios connectors already SSE-subscribe to `/runtime/calls`; extend that to a filtered session-event stream so `slack_send`-style finality can coexist with live typing without new tools.

### Tier 3 — worthwhile, someday

11. **Typed authoring SDK + `aios init` + watch-mode dev loop** *(authoring)*. A `defineAgent({ tools: [defineTool(...)] })` builder that *lowers to* `AgentCreate` JSON with compile-time checking over the policy union (aios already generates a TS SDK from `openapi.json`), plus an `aios init` scaffold and a watch-mode `aios chat` that re-POSTs a new agent version on save. **Ergonomics layer in front of the existing API — does not touch the DB-row/versioning model.**
12. **Declarative "standing triggers"** *(triggers)*. Let an environment ship a reviewed set of file-authored triggers reconciled into rows at deploy (build-time duplicate-name + cron-occurrence validation), coexisting with runtime-mutable ad-hoc triggers. Moves model-authored-trigger errors left.
13. **Unguessable random-path ingress** *(triggers)*. Mint a high-entropy path segment as the `external_event` ingress URL so the path *is* the capability (eve's `cron-handler-route.ts`), complementing the current hashed-bearer token.
14. **Publish a headless TS client** *(UI)*. Extract `AiosSessionStore` (an `EventSource` wrapper + the `buildTranscript` reducer + optimistic compose) — logic that **already exists privately inside aios-console** — into a framework-agnostic `@aios/client` + `@aios/react useAiosSession`, generated like the Python SDK. The console becomes its first customer; aios gains the embed story it structurally lacks.
15. **Template-prewarm + a microVM/gVisor backend** *(sandbox)*. Bake a reusable committed base image (CA + packages + base setup) so cold starts don't re-apply the netns lockdown from scratch; and make the sandbox Protocol a *real* multi-impl seam by shipping a hardware-isolation backend (today only opt-in `--runtime runsc`).

### Explicitly **reject** (these conflict with load-bearing aios stances)

- **LLM compaction** (`harness/compaction.ts` — "You are a conversation summarizer"). Lossy, non-deterministic, non-auditable, rewrites the prompt-cache head at the *most expensive* moment, and **has no recall mechanism**. Directly contradicts aios's immutable-log / monotonic-context / no-compaction stance, which the user holds as load-bearing. aios's deterministic windowing + `search_events` is the better answer.
- **Outsourcing durability to a beta SDK.** eve's entire durability guarantee rides `@workflow/*@5.0.0-beta.*`. Adopting it would surrender ghost-repair side-effect-safety and single-source wake predicates — the guarantees a multi-tenant backend cannot delegate.
- **Path-derived identity for tools/agents.** Re-introduces the rename-breaks-the-model-contract hazard aios deliberately avoids with stable ids + versioning.
- **Durable park for HITL.** Contradicts the no-park property that is aios's headline HITL differentiator.
- **eve's no-DB / no-tenancy model.** It is right for eve's product and disqualifying for aios's.

---

## 3. Where aios is genuinely ahead

The counterweight — real architectural advantages, not consolation prizes:

- **Responsiveness & interleaving (async model).** aios's headline property is *real*: `run_session_step` returns immediately after launching fire-and-forget tool tasks, and a user message short-circuits inference eligibility **regardless of in-flight tools** (`sweep.py:704`, verified not gated by the in-flight set). A user can redirect a busy agent mid-tool. **eve blocks the run for the duration of a tool's step** — a 90s web-fetch freezes the session, and there is no AbortSignal path (verified). For a conversational/connector product this is the single biggest gap in eve.
- **Lossless, queryable, prefix-stable context.** Immutable append-only log → the window is a *pure function* of it; re-rendering is byte-identical; nothing is ever destroyed (recoverable via `search_events`); zero hot-path model calls. eve's compaction is lossy and rewrites the cache head.
- **State you can actually operate.** Full SQL over sessions/events; derived status is a SQL predicate (never a stored column) sharing the *same* single-source fragments as the scheduler, with a structural drift-guard test — a class of "worker wakes with no work / skips a session" bugs designed out. eve has no `SELECT`, no point-in-time query, no cross-session analytics.
- **No-park HITL + segregation of duties.** The agent never stops being responsive while an approval is pending; approval is a first-class operator-scoped API resource (a distinct principal from the conversant — the segregation eve structurally lacks); `Session.awaiting` is a queryable derived view (eve's pending state is opaque snapshot keys).
- **Real multi-tenancy.** sha256-hashed account keys, a DB-enforced single-active-root invariant, account_id scoping, and **per-account HKDF-derived encryption subkeys** so a tenant's vault secrets are cryptographically isolated. eve has *zero* tenancy — "that's your job," per its own docs.
- **Sandbox hardening + durable FS.** Deep defense-in-depth on its one backend (netns-sidecar holding the firewall while the sandbox has no `NET_ADMIN`, always-on seccomp, cgroups, egress CA) + commit-on-teardown + crash-corpse salvage + GC reconciler. eve's *realistic self-host backends* — Docker and just-bash — ship **unhardened** (no cap-drop, no seccomp; just-bash is an in-process interpreter with full network).
- **Triggers as a live, audited, agent-authored primitive.** Runtime-mutable, self-schedulable, `trigger_runs` audit, reactive sources (`run_completion`, `external_event`), safe multi-worker firing (FOR-UPDATE-SKIP-LOCKED + lease). eve's schedules are static build-time cron files with no audit and no self-scheduling.
- **The focal-channel abstraction.** One session bound to many channels with one focal channel + unread markers + `switch_channel` recap — the natural primitive for one always-on assistant spanning Signal+Slack+WhatsApp. eve's per-channel-session model **cannot express it**.
- **Durability you can audit.** Every invariant (gapless seq, monotonic context, tool-always-appends-one-result, NOTIFY-after-commit) is enforced in-repo; **ghost-repair distinguishes `did_not_run` (safe retry) from `may_have_completed` (side-effect risk)** — a double-fire guard eve appears to lack. Not hostage to a beta SDK.
- **Event-log-as-trace-store.** Forensically complete at event granularity with *zero* external infra; per-call cost/usage is a queryable dimension. eve's granular event stream is in-memory/ephemeral; full value needs Braintrust + the Vercel dashboard.

---

## 4. The deliberate philosophical forks

Four places where the systems diverge *by design*, and aios's choice is the right one for its product — worth stating so they're not mistaken for gaps:

| Fork | eve | aios | Verdict |
|------|-----|------|---------|
| Context bound | LLM summarization (lossy, rewrites head) | deterministic windowing + recall (lossless) | aios — and **don't** adopt eve's compaction |
| Durability | rent it (Workflow SDK) | own it (event log + sweep) | each right for its product; aios must not outsource |
| HITL | durable park (run blocks) | no-park (agent stays live) | aios for a responsive multi-tenant backend |
| Authoring | code (files in your repo) | data (rows behind an API) | aios for multi-tenancy — *but the ergonomics gap is real and fixable* |

The last row is the subtle one: **"definition is data" and "authoring surface is typed and local" are orthogonal concerns aios conflated.** aios chose *data* and got *no ergonomics* for free — when it could have both (Tier 1/3 adopt items).

---

## 5. If aios were rebuilt from scratch today

**Keep the spine — it's right and eve doesn't beat it for this purpose:** immutable append-only Postgres event log as lossless source of truth; no-loop step model (the source of async-responsiveness, monotonic context, prompt-cache stability, free mid-turn injection); deterministic windowing (no LLM compaction); no-park HITL; durable hardened sandboxes; account-scoped multi-tenancy with owned migrations; MCP-as-extensibility; fail-hard simplicity.

**Six structural changes eve's design argues for:**

1. **Unify the two durable runtimes into one primitive with two policies.** aios maintains *two* near-identical durable step engines — the **session step** (procrastinate + event log) and the **workflow step** (replay-from-memo journal) — with duplicated idioms (`lock=id`, `queueing_lock=id`, NOTIFY-after-commit, single-writer journal). **eve's biggest structural win is having exactly ONE workflow engine under both conversations and scripts.** A greenfield aios should have one durable-step primitive parameterized by policy: `model-call` (today's session) vs `deterministic-replay` (today's workflow). This is the highest-altitude simplification on the table.

2. **Make per-tool durability correct-by-construction, not correct-by-vigilance.** Today async tools are fire-and-forget asyncio tasks backstopped by a *polling sweep*; ghost-repair + `SELECT FOR UPDATE` dedup + `open_tool_call_count` compensation + `TaskRegistry` cancellation are all hand-rolled because the tasks aren't durable. **Keep the interleaving/responsiveness *semantics*, but make each tool call a durable, journaled, replayable unit with its own deadline** — which would collapse that entire recovery-machinery surface. This is precisely "unify toward minimal primitives, correct-by-construction" (CLAUDE.md) applied to the subsystem that most resists it. eve proves a real durable-execution engine gives this for free; the lesson isn't "rent eve's engine," it's "this complexity is essential *only because* we chose non-durable tasks."

3. **Make cross-tenant access unrepresentable.** `account_id` scoping is enforced *by convention* (thread it into every WHERE clause) and leaks are a **recurring audit finding** (the "account_id kwarg accepted but unused in SQL" sweep, #426). Turn eve's accidental safety ("we have no shared rows to leak") into aios's structural safety ("our shared rows *cannot* be leaked"): Postgres **Row-Level Security** with a per-request `SET LOCAL app.account_id`, or a query layer that physically cannot emit an unscoped statement.

4. **Name the serialized-payload contract.** Stamp `schema_version` into every event `data` blob and workflow memo frame; centralize one read-time `runMigrationChain`. Replaces implicit forward-tolerance accreting silently in the context builder with a named, tested home for "the conversation wire shape changed."

5. **Build the authoring-ergonomics, evals, and embeddable-client layers in from day one.** These are the three things aios *conflated away* by choosing data-not-code: a typed builder that lowers to `AgentCreate` JSON with compile-time checking; a behavioral eval harness as a peer to the unit/e2e suites (the durable log makes this *easier* than eve's in-memory capture); and a published headless TS client that the operator console consumes as its first customer.

6. **Treat sandbox + model-routing as real multi-impl seams.** Ship a gVisor/microVM backend so untrusted-agent execution has a hardware-isolation tier (not just opt-in `runsc`), with template-prewarm amortizing cold starts. Model the "mind" as a *typed* field with optional ordered fallbacks + error-class-aware retry, instead of burying `api_base`/`provider.order` in an open `litellm_extra` dict (which is also a weaker security posture — an un-validated redirect target on an open dict).

**The through-line:** aios's macro-architecture is the correct choice for a multi-tenant, auditable, always-responsive runtime — and materially stronger than eve on every dimension that *is* its product (tenancy, responsiveness, lossless state, owned durability, sandbox hardening, triggers). eve's contribution to a rebuild is almost entirely at the **micro-pattern and ergonomics** altitude: typed authoring, error-class retry, in-window pruning, versioned payloads, model-callable delegation, interactive OAuth, evals, OTel, a headless client — *plus* one genuinely high-altitude idea (one durable primitive, not two). None of those require giving up the spine; all of them are additive.

---

## 6. Concrete next steps (filing suggestions)

1. **`shovel-ready`-able now:** error-class-aware retry (#model); schema-first tool decorator generalized from `workflow_management` (#tools); per-input approval predicate + approve-once (#hitl); random-path `external_event` ingress (#triggers).
2. **`needs-design` (worth a design issue):** behavioral eval + LLM-judge harness; in-window deterministic tool-result pruning (with the pure-function-of-log constraint); `schema_version` + read-time migration chain; interactive-OAuth connection-authorization subsystem; model-initiated `delegate` tool clamped through #794.
3. **Architectural / `kaikaku` altitude:** unify the session-step and workflow-step durable runtimes onto one primitive; per-tool durable-unit refactor to retire ghost-repair/dedup-by-vigilance; RLS-based cross-tenant-by-construction.
4. **Product-strategy calls (not eng):** publish a headless `@aios/client`; gVisor/microVM sandbox backend; typed authoring SDK + `aios init` dev loop.
