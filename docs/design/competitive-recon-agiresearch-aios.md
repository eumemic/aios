# Competitive recon — agiresearch/AIOS vs our aios

**Date:** 2026-06-27
**Their target:** [`agiresearch/AIOS`](https://github.com/agiresearch/AIOS) @ `4171a8ea` — the academic "AIOS: LLM Agent Operating System" (COLM 2025), ~28k LoC Python + a stub Rust scaffold.
**Our repo:** `aios` (eumemic agent runtime), ~84k LoC Python.
**Method:** cloned their source; 14-dimension comparison, each *analyzed against actual source then adversarially fact-checked*; synthesis → adopt-list / where-we-lead / greenfield; then the keystone **primitive-gap filter** (classify every adopt-item against our own primitive catalog, adversarially minimized). 41 agents, ~4.5M tokens.

---

## Headline

> **0 new core primitives.** Every one of the 7 genuinely-stealable ideas is an **extends** (5) or **composition** (2) of a primitive we already have. Their signature move — cross-session LLM-call batching — is an explicit **reject**. The single high-value steal is **semantic memory recall**: we are keyword-only (Postgres FTS) and that is the weakest link in our strongest differentiator ("memory is the moat"). Net: the comparison *validates our architecture by counterexample* — they are strictly dominated on every load-bearing axis (durability, tenancy, capability security, async tools, orchestration), and their three marketed centerpieces are vaporware or broken on the live path.

A name collision, not a competitor: same three letters, opposite product model. Theirs is a **single-tenant, local-first research kernel** where the OS multiplexes "syscalls" and the *agent logic lives outside* in an external SDK (Cerebrum). Ours is a **multi-tenant, durable, hosted runtime** where the *harness IS the loop* and an agent is just config (a "mind" = LiteLLM string). "Who's better" is only meaningful per-dimension, per-product-model.

---

## The adversarial-verify payoff (marketing vs source)

The single highest-value step. Their README sells a COLM-2025 operating system; their source tells a different story. Verified against code, not docs:

| Marketed centerpiece | Reality in source |
|---|---|
| **A-MEM agentic memory** (auto-evolving, semantic) | `execute_memory_evolve` is **unreachable** on the live `address_request` scheduler path *and crashes if reached* (a `bool.lower()`, a `range()` over a list). `evolution_history` is populated in `__init__` and never mutated. Default `in_house` store is a **process-local dict + in-memory Chroma** — a restart loses everything. |
| **Smart (embedding) model routing** | ChromaDB/gdown corpus, weak last-user-message signal; not on the default live path. Defeats prompt caching by design. |
| **Round-robin generation preemption** (`gen_snapshot`/`gen_recover`) | Bare `pass` bodies, **zero callers** anywhere in the tree. Real only for local HF models (saved KV-cache); for API models the partial is saved but **never re-read** — the loop regenerates from scratch. |
| **Multi-tool dispatch** | **Literally broken**: `address_request` returns inside the first loop iteration, so a response with N tool calls executes **exactly one and silently drops the rest**. |
| **Security** | Zero auth on any route (`CORS allow_origins=['*']`); an unauthenticated `/core/config/update` can overwrite the operator's billed provider keys; the filesystem syscall joins `root_dir + caller-supplied path` with **no traversal guard** → arbitrary host read/write. |

Every dimension's verdict came back `minor-corrections / high-confidence` — and almost every correction *downgraded their side*.

---

## Scorecard (14 dimensions)

`Edge` = better **for its own product's domain**. "What they have we don't" is the honest gap.

| Dimension | Edge | Crux / what they have that we don't |
|---|---|---|
| Durability / crash recovery | **US** (categorical) | Their run-state is a `Future` in a RAM dict keyed by a `randint` PID; a restart silently drops every in-flight agent. Ours IS the append-only log. |
| Tenancy / auth / capability security | **US** (categorical) | They have *none*. Ours: account scoping + attenuation lattice + per-account-encrypted secrets. |
| Async & concurrency | **US** | Thread-per-agent + thread-per-syscall (a `Syscall(Thread)` whose `run()` is just `event.wait()`); a slow tool head-of-lines every agent. We're fire-and-forget async. |
| Multi-agent orchestration | **US** | They cannot call agent→agent at all (the one line that would is commented out). We have durable call_* + workflows. |
| Tools & dispatch | **US** | Theirs is synchronous, unguarded, and the N-tool loop is broken. Ours is async + permission-gated + crash-recoverable. |
| Channels / triggers / HITL | **US** | They assume a human present at a REPL; no connectors, no cron, no approval gate. |
| Execution model / the loop | **US** (our domain) | They put the loop in agent code (bring-your-own-loop expressiveness) — at the cost of every reliability property. |
| Memory — **storage** | **US** | Versioned, immutable, tenant-isolated, deterministic vs their non-durable global dict. |
| Memory — **retrieval** | **THEM** ⭐ | **Semantic/vector recall.** We are keyword-FTS-only; "lost account" won't surface "the customer churned". *The one real capability gap.* |
| Sandboxing | **SPLIT** | US: multi-tenant isolation + credential mediation + durable FS. THEM: **GUI computer-use** (a11y-tree grounding) over a full-OS VM — fidelity for a benchmark, no isolation. |
| Scheduling / fairness | **THEM** (concept) | A pluggable `BaseScheduler` policy seam (FIFO/RR) + cross-request batching. We have only a binary fg/bg priority. (Their *impl* is weak: dead priority field, 1s latency tax.) |
| Model abstraction / routing | **THEM** (capability) | Multi-core router + in-process local serving (vLLM/HF/ollama). We bind one mind per agent — deliberately, for prompt-cache stability. |
| Context management | **US** (our domain) | They preempt the token stream (mostly broken); we re-derive a monotonic, cache-stable context from the log. |
| Authoring / DX | **SPLIT** | US: zero-code versioned config + durable. THEM: framework-agnostic bring-your-own-loop + a pip-for-agents hub (`app.aios.foundation`). |

---

## Adopt-list (ROI-ranked) — with primitive lineage

Each item is tagged with its **phase-4 classification** (0 new primitives across the board).

### Tier 1 — soon
1. **Hybrid semantic + keyword memory recall** · *extends `Memory store`* · effort M / risk M
   On each immutable `MemoryVersion` write, also compute+store a `pgvector` embedding alongside the existing `content_tsv`; extend `memory_search` from pure `ts_rank` into a hybrid query fusing FTS + vector cosine via reciprocal-rank fusion, through the *same* per-session view. **Reject** their auto-inject / auto-extract / LLM-evolution — memory stays model-driven, immutable, deterministic, cache-stable. *This is the identical "maintained-denormalization of an append-only log + read-only-SQL tool" shape the FTS column already established (migration 0119) — the second ranking term, not a new mechanism.*
   ⚠️ **Gating sign-off:** this is the **first pgvector and first model-call on the memory write path** in the tree. Fail-hard makes the embedding provider a hard dependency of every memory write. And in-DB vector/FTS indexing both require plaintext → in tension with encryption-at-rest. Settle the write-path failure mode before building.

2. **Per-account fairness weight in wake priority** · *extends `Step/wake`* · effort M / risk L
   Generalize the binary `_FOREGROUND_PRIORITY=0` / `_BACKGROUND_PRIORITY=-10` derivation in `get_wake_priority_context` to fold a per-account fair-share weight (or token-bucket demotion) into the *same* `priority` column procrastinate already sorts by. No new mechanism. ⚠️ The fair-share term must demote **only within the background tier** — a heavy account's foreground (user) message must never sink below a light account's. Defer the exact weighting until contention is observed (lock framing now).

### Tier 2 — someday (genuine but unforced; each a clean extends/composition)
3. **Per-agent model fallback chain** · *extends `Agent`* · effort S / risk L
   `model: str | ModelPolicy` where `ModelPolicy` is a KIND union (`single | fallback_chain | escalation`), resolved to **one** concrete string at step entry (cache-stable). On a *terminal* provider error that isn't auth (5xx / region-outage / model-not-found) retry the next string before parking. **Sharpest concrete deliverable:** `loop.py:125` currently latches litellm `NotFoundError` (404) as *unconditionally terminal* — directly contradicting a fallback chain; move it into the chain-advancing set, latch only after exhaustion. Ship `single`+`fallback_chain`; **defer `escalation`** (escalate-on-success risks the rejected mid-step model swap, and is arguably just `call_agent` composition). **Reject** their embedding SmartRouting.

4. **Agent export/import as a `name@version` artifact** · *composition (Agent + Skill + ToolSpec), core_change: none* · effort M / risk L
   A serialized `AgentVersion` + referenced `SkillVersion` bundles — all already pure Pydantic. Export = `get` → JSON; import = deserialize → `create` (through the same surface-attenuation clamp). Pure CLI/lib porcelain (`aios agents export/import`). **No agent-as-code, no hub.** Fold the "single-bundle first-run scaffold" half into the existing `aios init` scaffold (#1353) at filing time.

5. **GUI/browser computer-use as a sandbox capability layer** · *composition (Environment+Sandbox + Tool + ToolSpec + invocation spine), core_change: none* · effort L / risk M
   A headless Chromium/Playwright as an optional **image layer** of the existing per-session sandbox (`EnvironmentConfig.image` already accepts any ref), surfaced as in-tree builtins `screenshot/click/type/scroll` with `executes='sandbox'` (the axis bash already uses), behind the existing fail-closed egress. The steal-worthy part is **pure observation logic**: their accessibility-tree / Set-of-Mark grounding (numbered bounding-box overlays → model targets element *indices*, not pixels), returned through the existing multimodal `ToolResult.content` shape (`read.py` already returns `image_url` parts). **Reject** their nested-KVM/shared-VM stack. *Crucially these are in-tree builtins → they enter `principal.tools` → lattice-visible — no attenuation gap (the load-bearing disanalogy with item 6).*

6. **External/third-party tool registration seam** · *extends `ToolSpec` union (NEW arm)* · effort L / risk M · **needs-design**
   A `ToolProvider` arm letting operators register sandbox-executed tools per-agent at runtime without a core deploy. **This is the only item with a genuine new union arm AND a security gate** (see below). ⚠️ **Two gating questions first:** (a) is the in-tree ~49-builtin ceiling actually binding? `mcp_toolset` already gives unbounded *external* runtime tools, `custom` gives client-executed, `http_servers` gives endpoint passthrough — the only uncovered slice is a *runtime-registered sandbox-executed* handler; demand a concrete driver. (b) the attenuation gap below.

### Tier 3 — defer (shape worth copying, but already bounded)
7. **Cost-aware inbound intent triage** · *extends `Connection/Channel`* · effort M / risk M
   Their two-stage intent-router shape (cheap keyword/score gate, model only on genuine ambiguity) as a pre-inference filter in `services/inbound.py`. **Drop their LLM-fallback** (the API process never calls a model — that *is* waking the session) and **stay fail-closed** (their drop-on-low-confidence is fail-OPEN, loses a real message). **Defer** entirely until inbound spend is measured — #1504's rolling-count budget already bounds it.

### Reject
- **Cross-session LLM-call batching** — their signature OS move (recurs 4×). Conflicts head-on with prompt-cache stability + per-session isolation, imposes an interactive-latency floor, and only pays off in a self-hosted/shared-inference regime we don't run. *(Greenfield keeps an opt-in, adaptive, zero-when-shallow coalescing seam behind `completion.py` as a someday-maybe at high concurrent-session counts — but not the fixed-interval drain they ship.)*

---

## The keystone — primitive-gap map

**0 new primitives. 5 extends, 2 compositions, 0 not-core.** Our 16-primitive core absorbs every adopt-item. No item is a new *spine* element (no new durable log, no new re-entrancy mechanism, no new provenance carrier) — the only bar that clears a new-primitive claim.

```
EXTENDS (small generalization of one primitive):
  semantic memory recall      → Memory store        (add a 2nd ranking column)
  per-account fairness         → Step/wake           (richer value into existing priority col)
  model fallback chain         → Agent               (model: str → discriminated KIND union)
  external tool seam           → ToolSpec union      (new arm — NEEDS-DESIGN, security-gated)
  cost-aware inbound triage    → Connection/Channel  (new InboundDrop predicate arm)

COMPOSITIONS (built from existing primitives, core_change: none):
  GUI computer-use             ← Environment+Sandbox + Tool + ToolSpec + invocation spine
  agent export/import          ← Agent + Skill + ToolSpec  (client-side serialize round-trip)
```

**Two collapse findings the skeptic made explicit:**
- *external-tool seam* + *GUI computer-use* are the **same** Tool-axes + ToolSpec-arm primitive (both `executes='sandbox'`) — but they **diverge load-bearingly**: browser tools are in-tree builtins (lattice-visible, no new arm); the external seam is runtime-declared (lattice-blind unless explicitly fixed). The collapse must not launder the external item's gap behind the browser item's cleanliness.
- *per-account fairness* + *inbound triage* share a **verdict (defer)** and a **discipline (KIND not flag)**, but are **distinct mechanisms at distinct code sites in distinct processes** (worker `defer_wake` priority scalar vs API-process `handle_inbound` predicate chain). Do **not** merge into one code path.

### ⚠️ The one security finding (load-bearing, surfaced by the skeptic)
`surface_of(principal)` (`models/attenuation.py:73`) projects **only** `principal.tools` — the agent's *declared* ToolSpec list. Tools injected via the existing **`ToolProvider` prelude seam** (`tools/providers.py`, connection-scoped) are **NOT in `principal.tools` and therefore NOT clamped by the attenuation meet today.** Routing a *sandbox-executed* handler (the most powerful execute-class) through that seam would extend an attenuation-*bypassing* path to sandbox execution — an ambient-authority hole (#794-class defect). **The external-tool arm must enter the declared, `surface_of`-visible tool list and resolve by registry name — never via the prelude seam.** This is the gate on item 6, and it's a finding worth acting on *independent* of whether we ever build the external seam.

---

## Where we lead (the honest counterweight)

Eight dimensions where we're categorically ahead **for the hosted, multi-tenant, asynchronously-reachable-human domain** — each validated by their counterexample:

1. **Durability** — run-state IS the durable log; any worker rebuilds exact context after a crash; `find_and_repair_ghosts` is real recovery. Theirs dies with the process.
2. **Tenancy + capability security** — account-scoping everywhere + a single algebraic lattice meet (non-amplification across arbitrary spawn depth, free by associativity) + secrets swapped only at the TLS-MITM egress boundary. Theirs: zero.
3. **Async / responsive-while-working** — the no-loop step model makes "every tool implicitly async" + mid-turn user injection fall out for free. Theirs blocks a thread per syscall.
4. **Multi-agent orchestration** — agent→agent + DAG fan-out from one durable trusted-request-edge; survives restart. They can't call agent→agent at all.
5. **Tool dispatch** — exactly-one-result + ghost-repair + a real permission ladder. Theirs is broken + ungated.
6. **Sandbox envelope** — credential mediation, netns-sidecar fail-closed egress, gVisor-selectable, durable full-FS persistence; runs anywhere Docker runs. Theirs needs nested KVM, one shared VM, arbitrary host read/write.
7. **Memory storage model** — versioned, immutable, actor-stamped, tenant-isolated, deterministic. Theirs non-durable + unreachable evolution.
8. **Channels / triggers / HITL** — three compositions of one event-log + durable-wake spine, fail-closed inbound. Theirs structurally absent.

---

## Greenfield — what the mirror showed about *us*

**Keep verbatim (right even greenfield):** agent-as-config-not-code; the no-loop step model; run-state-IS-the-log; gapless single-writer `append_event` + ghost-repair; durable job queue as scheduling substrate; tenancy in the data-access layer; capability authority as one lattice meet; secrets-never-in-sandbox; the per-session sandbox envelope; channels/triggers/HITL as compositions; deterministic model-driven memory; one model bound per agent (cache-stable).

**Reconsider — two internal items the teardown surfaced (not steals, reflections):**
- **RLS-backed tenancy** — `account_id` scoping is currently held by **convention** (`_get_scoped`/`_list_scoped`); there is **no Postgres RLS** (zero `CREATE POLICY` in any migration). One forgotten `WHERE` clause is a latent cross-tenant leak. Our own stance is "make invalid states unrepresentable" — back the convention with RLS as a fail-closed DB backstop (the `unscoped_*` paths get an explicit bypass role). *The highest-stakes invariant in the system is the one held by vigilance rather than structure.*
- **Fold `api_base_trusted` into the attenuation lattice** — capability authority is 90% one beautiful primitive (the meet) + 10% a special-cased model-endpoint allowlist sitting beside it. Folding the inference endpoint into the lattice gives one meet to audit and covers it by the same non-amplification proof. (Lowest-stakes; validate the lattice-fit first — it may genuinely be a single-value trust boundary, not a set to intersect.)

**Self-improvement broken window (act now, unrelated to them):** the trigger invariant `enabled ⟺ next_fire IS NOT NULL` is currently upheld by the #957 heal-on-update *pipeline* — a corrective mechanism contra correct-by-construction. Make it a CHECK/generated column so the illegal state is unrepresentable and the heal becomes deletable dead code.

---

## Issue filing map (post dedup against the live tracker)

Checking our own tracker first reshaped the plan — two top items already exist:

| Adopt item | Lineage | Action | Readiness |
|---|---|---|---|
| 1 · Semantic/hybrid memory recall | extends `Memory store` | **New issue**, pairs-with epic #1370, extends the shipped FTS slice #1372 | needs-design |
| 2 · Per-account fairness weight | extends `Step/wake` | **Comment on open #418** — contributes a new "Option D" (fold a fair-share weight into the existing `priority` column procrastinate already sorts by; demote *within the background tier only*; the Postgres-vs-process-local meter fork is #418's own Option-A cons) | (on #418) |
| 3 · Model fallback chain | extends `Agent` | **New issue** — ship `single`+`fallback_chain`, the `loop.py:125` NotFoundError reclassification is the concrete deliverable; defer `escalation` | needs-design |
| 4 · Agent export/import | composition | **New issue** — client-side serialize round-trip; fold scaffold-half into shipped `aios init` (#1353) | needs-design |
| 5 · GUI computer-use layer | composition | **New issue** — a11y-tree grounding is the steal; reject nested-KVM | needs-design |
| 6 · External tool-provider seam | extends `ToolSpec` | **New issue** — carries the lattice-visibility security gate (below) | needs-design |
| 7 · Cost-aware inbound triage | extends `Connection` | **Defer, don't file** — #1504's rolling-count budget already bounds inbound spend; revisit only when spend is measured and shown triage-dominated | — |

Two of our own findings the mirror surfaced, fileable independently of adopting anything:
- **Lattice-visibility gap** (security; #794-class) — `ToolProvider`-prelude-injected tools bypass `surface_of`/the attenuation meet today. Folded into item 6's body; merits a standalone security issue.
- **Trigger invariant broken window** — make `enabled ⟺ next_fire IS NOT NULL` a CHECK/generated column, retiring the #957 heal.

## Bottom line

Steal **one capability now-ish** (semantic recall — close the gap in the moat), **one fairness fix** the multi-tenant model exposes, and a handful of composable extensions later — **adding zero new core primitives**. Reject their signature batching. Fix two of our own things the mirror exposed (the lattice-visibility gap for runtime tools; RLS as a tenancy backstop). Everything load-bearing in our design is validated by their counterexample.
