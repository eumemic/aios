# Primitive decomposition of the 15 eve-adoption items

*2026-06-17. Multi-agent workflow: 1 primitive cartographer → 15 classifiers → draft minimizer → adversarial skeptic. Verdict: **0 new core primitives.** Every item is a composition, a small extends of one existing primitive, or non-core (a lib/tests/sidecar consumer of shipped surfaces).*

> **Companion:** [`aios-vs-eve-architecture.md`](aios-vs-eve-architecture.md) is the full 14-dimension comparison this adopt-list came from. The items below are filed as issues **#1335–#1362** plus a review-finding on PR #1314; each issue carries its own self-contained implementation design and links back here for the lineage.

## The catalog (what the core already is)

12 shipped primitives — everything else is a projection of these:
1. **Event-log append** (gapless seq, single writer) — the one source of truth; status/window/awaiting are pure derivations.
2. **Session step + wake/sweep** (the no-loop loop).
3. **Externally-executed tool / dispatch-KIND classifier** — built-in/MCP/custom/always_ask are 4 arms of one parked-call mechanism.
4. **Deterministic windowing + monotonic context replay** (no LLM compaction; `search_events` is the recall hatch).
5. **The invocation edge** (`request_opened`/`request_response`) — the **Ask** arm; *already shipped*, caller-kind-agnostic, one resolver.
6. **Durable workflow replay engine** (journal + replay-from-memo).
7. **#794 capability-surface attenuation** (the lattice meet).
8. **Triggers** (source × action discriminated unions).
9. **Sandbox backend Protocol** + provisioning spec.
10. **Vaults / per-account crypto + egress secret substitution** (incl. the `vault_oauth` engine).
11. **Connections / bindings / routing-key / focal-channel.**
12. **The mind / LiteLLM provider boundary.**

**Emerging unification (in flight, not new work I'd file):** the **`deliver(Ask|Tell)` kernel over a private `stimulate` spine** — #1122 epic, #1197 (factor the spine + Tell arm), #1165 (trigger=Tell), #1127 (session-caller parking `invoke()`), #1130/#1131/#1132. Ask = the shipped invocation edge; Tell = fire-and-forget. **Several of the 15 are consumers of this edge, not new primitives.**

## The decomposition (0 new primitives)

| # | Item | Verdict | Rides which primitive | Lands as | Label |
|---|------|---------|----------------------|----------|-------|
| 1 | Error-class retry | **composition** (thin) | mind boundary + step | named predicate at the model-call `except` seam (REFUSAL_FINISH_REASON precedent) | **shovel-ready** |
| 3 | Schema-first tool authoring | **composition** (thin) | dispatch-KIND classifier | lift `workflow_management._parse` → shared `tools/input.py` util | **shovel-ready** |
| 11 | Provider-native structured output | **composition** (thin) | mind boundary + Ask edge | `_supports_json_schema_response_format` family-gate + `output_schema` field; keep post-hoc validator | **shovel-ready** |
| 13 | Declarative standing-triggers | **composition** (core_change=**none**) | triggers + event-log | validation+random-path ingress **already shipped**; reconcile = deploy tooling on #1226 | **shovel-ready** (reconcile slice) |
| 10 | Connector session-event stream | **composition** (thin) | event-log NOTIFY + connections SSE | 2nd reader of the shipped `_notify_delta` channel; rendering = SDK (non-core) | needs-design (verify) |
| 4 | Model-initiated delegation | **composition** (extends-existing) | **Ask edge** + #794 + dispatch | `agent_tool` builtin over the edge — **depends on in-flight #1127/#1122** (session caller) | needs-design |
| 6 | In-window tool-result pruning | **extends** | windowing | relocate the deterministic-collapse operator in-window, behind #167 `WindowingStrategy` KIND | needs-design |
| 7 | Versioned event-data migration | **extends** | event-log | `schema_version` codec at the single decode chokepoint; sibling of `host_semantics_epoch` | needs-design |
| 9 | Per-input approval + approve-once | **extends** | dispatch-KIND classifier | generalize `classify_permission` to declarable data; approve-once = log derivation | needs-design* |
| 5 | Interactive OAuth (connection) | **extends** | vaults/`vault_oauth` | generalize to target-agnostic `(target_kind, target_id)` discriminated flow (real schema work) | needs-design |
| 15 | Template-prewarm + microVM | **extends** | sandbox Protocol | microVM = 2nd Protocol impl + config-driven backend; prewarm = compose snapshot; **gVisor already ships** | needs-design (microVM) / shovel-ready (prewarm) |
| 2 | Behavioral eval + judge | **not-core** | consumes Ask+`output_schema`, event log | `tests/` harness + published eval lib | needs-design (lib) |
| 8 | OTel export adapter | **not-core** | consumes the span stream | published lib / sidecar exporter (don't put OTLP in the hot path) | needs-design (lib) |
| 12 | Typed authoring SDK + init + watch | **not-core** | consumes openapi contract + `agent_versions` | published lib + CLI | needs-design (lib) |
| 14 | Published headless TS client | **not-core** | consumes openapi + SSE projection | published TS lib (twin of the Python SDK) | needs-design (lib) |

\* approvals: the predicate generalization is shovel-ready-shaped, but the **value-keyed idempotency** (approve `amount=1001` must NOT auto-approve `amount=1,000,000`) is a real design point → needs-design until that's nailed.

**Counts:** 0 new primitives · 5 extends · 7 compositions · 4 not-core.

## The cross-item collapses (the high-value insights)

1. **OAuth does NOT collapse onto the Ask edge** (collapse-*negative*, the skeptic's headline guard). A naive unifier groups everything that "parks and resumes," but interactive-OAuth is resumed by a **human browser callback** — no runtime servicer awaits it, no totality/quiescence invariant keys on it. Modeling it on the deliver spine would force totality/depth machinery onto a consent flow for zero gain. It's a **`vault_oauth` extension**, not a pending-request. *This is the kind of spurious primitive the whole exercise exists to prevent.*
2. **Delegation (#4) is a consumer of the already-shipped Ask edge**, generalized to session→session by the in-flight #1127. Filing it as a new primitive would have duplicated live design work.
3. **Standing-triggers reconcile + authoring-SDK reconcile = ONE deploy-time reconcile loop** (the IaC-sync reregister pattern, generalized by #1226 to a target registry; the concrete reconcile script now lives in the company application, not in this substrate). Not two loops; not a core primitive. At most one additive `managed_by` provenance marker — provenance, never behavior.
4. **retry + structured-output** are two independent thin-affordances at the *same* mind/LiteLLM boundary (the `except` path and the kwargs path) — same generalize-over-enumerate family-rule shape as the existing `_supports_*` cache helpers.
5. **pruning + event-version** are two distinct *extends* on the "deterministic transform of the immutable log" family (render-side vs decode-side) — neither is a 14th peer primitive.
6. **evals + otel + authoring-sdk + headless-client** are all non-core consumers of two shipped surfaces — the `openapi.json` wire contract and the event-log SSE projection. They belong in `packages/`, `tests/`, or a sidecar — keeping the core minimal.

## What this means

The core stays clean: **nothing new gets added to the primitive set.** The work splits into three lanes — thin **compositions** (some immediately shovel-ready), careful **extends** of a single primitive each (invariant-preserving, needs-design), and **non-core** lib/tests/sidecar deliverables. Several items **cross-reference in-flight epics** (#1122/#1127, #1226, #167, #1014/#1022) rather than minting anything. That reshapes the filing plan away from "primitive epics" entirely.
