# WaM fusion-eval — the FOUNDATION (R0 ≈ native, by TOST)

The gating deliverable of the **workflows-as-models (WaM)** fusion eval. It proves
the `model:"workflow:<id>"` binding plumbing is *transparent* before any fusion
recipe is measured against it. If the identity recipe **R0** (a one-node workflow
`resp = await call_llm(req); return resp`) is not equivalent to the **same model
bound natively**, every downstream fusion delta is uninterpretable — the binding,
not the recipe, would be the variable. So this is the scoreboard's zero-point.

It extends the spirit of the #1221/#1282 blind-bakeoff *pattern* (isolated
dispatch, paired runs, an isolated account) but is purpose-built for a
**statistically-grounded equivalence verdict** (TOST) rather than a subjective
A/B/C ranking.

## What it checks

1. **R0 round-trips end-to-end.** A real session whose agent's `model` is
   `workflow:<id>` produces an assistant turn via the park→inner-run→`call_llm`→
   harvest path (the `model_workflow_park` / `model_workflow_harvest` /
   `model_workflow_harvest_end` spans are asserted present).
2. **Behavioral equivalence** to the same model bound natively, both at
   `temperature=0`, by **TOST** (Two One-Sided Tests) — equivalence is *positively
   established*, never inferred from "no significant difference."
3. **No double-charge.** The bound workflow meters its own `call_llm` spend on the
   inner run (charged once at the inference site); the harvested turn is **not**
   re-billed to the outer session. Asserted empirically: every R0 inner run
   charged `>0` µUSD, and every R0 *session* token meter stayed `0`.

## Pre-declared equivalence margins (stated before the data)

| metric | margin | alpha | rationale |
|---|---|---|---|
| output-length diff (chars), R0−native | ±8 | 0.05 | inside one short word; temp=0 re-emission noise floor |
| output-token diff, R0−native | ±2 | 0.05 | apples-to-apples usage (both arms expose `output_tokens`) |
| exact-match rate (normalized text) | ≥ 0.90 | — | transparent binding reproduces native token-for-token at temp=0 |

The **foundation passes** iff both TOSTs conclude *equivalent* (each `p_TOST < alpha`,
CI within ±margin) **and** exact-match rate ≥ 0.90 **and** no-double-charge holds.

## Result of record (2026-06-30, prod `api.aios.eumemic.ai`)

Run: `BASE_MODEL=anthropic/claude-sonnet-4-5`, temp=0, n=40, isolated account
`acc_01KWBY0PS67QQRXY6RA52A1FPV` (budget cap $8). R0 = `wf_01KWBY8YA06RASHZBV1T2DMSNT`.

| check | result |
|---|---|
| usable pairs | 40 / 40 |
| exact-match rate (normalized text) | **1.000** (bar ≥ 0.90) |
| no-double-charge | **HOLDS** — 40/40 inner runs charged once (>0 µUSD); 0 sessions re-billed |
| TOST output-length (±8 chars) | **EQUIVALENT** — n=40, mean_diff=0, CI [0, 0], p_TOST=0 |
| TOST output-tokens (±2) | **EQUIVALENT** — n=40, mean_diff=0, CI [0, 0], p_TOST=0 |
| **FOUNDATION VERDICT** | **PASS — R0 ≈ native (equivalence positively established)** |

At temp=0 every R0 output was byte-identical to native, so the paired diffs are a
constant 0 (sd=0). The TOST zero-variance branch concludes equivalence iff that
constant is strictly inside the margin — which it is. (A non-degenerate sample
would yield a tight non-zero CI; the all-zero result is the strongest possible
transparency outcome and is honestly the ceiling, not a statistical artifact.)

## How temperature=0 is made symmetric

A session turn builds its `LlmRequest` with `params = agent.litellm_extra`
(`src/aios/harness/loop.py`), and the WaM park forwards that **same** `request.params`
into R0's `call_llm` input (`src/aios/harness/model_workflow.py`). Setting
`litellm_extra={"temperature": 0}` on **both** agents therefore pins temp=0
identically on both arms — native passes it straight to the provider, R0 forwards
it through the binding. Fully symmetric, near-deterministic.

### temperature note (the opus-4-8 caveat)

`anthropic/claude-opus-4-8` **rejects `temperature`** (the provider returns
`temperature is deprecated for this model`). For a clean, low-noise equivalence
*foundation* we therefore baseline on **`anthropic/claude-sonnet-4-5`**, which
respects `temperature=0` and so makes any R0-vs-native delta attributable to the
binding rather than sampling. The binding is model-agnostic, so this proves the
plumbing; the harness `BASE_MODEL` is a one-line change to target any other model.
To run the foundation against opus-4-8, drop the temperature param and switch the
primary metric to a determinism-robust one (e.g. multi-sample agreement) — out of
scope for the foundation, which only needs *one* transparent baseline.

## Run it

```bash
# Creds: an OPERATOR key over an ISOLATED, budget-capped test account.
# The workflow: model-binding privilege is operator-only (enforce_workflow_binding_privilege),
# and creating the agent via the account's own API key IS the operator/HTTP path — so it is
# permitted with no allowlist entry. NEVER use a live-seat or Ultron account.
export AIOS_URL=https://api.aios.eumemic.ai
export AIOS_API_KEY=<isolated-test-account-key>   # never logged, never a CLI flag

python3 run_eval.py --n 40 --timeout-s 120 --throttle-s 2.0
# -> prints the per-task table, the no-double-charge check, both TOSTs, the verdict,
#    and writes results.json

# Reuse already-provisioned resources (idempotent re-run):
python3 run_eval.py --n 40 \
  --r0-workflow-id wf_xxx --env-id env_xxx \
  --native-agent-id agent_aaa --r0-agent-id agent_bbb
```

Throttling: pairs run **sequentially** with `--throttle-s` between them (default
2s) — throttle-before-saturation, so the inference pool is never flooded. 40 tasks
× 2 arms ≈ 80 short inferences (~$0.025 total at the subsidized rate).

## Files

- `run_eval.py` — the harness (provision → paired sample → no-double-charge check → TOST).
- `aios_client.py` — minimal stdlib AIOS API client (urllib only; no SDK coupling).
- `tost.py` — pure-Python paired TOST (Student-t CDF via incomplete-beta; no scipy).
- `test_tost.py` — validates the TOST math against known critical values.
- `tasks/r0_tasks.json` — 40 short single-turn factual prompts (deterministic-friendly).
- `results.json` — last run's full per-pair record + verdict (git-ignored if large).

## Extension point (for the fusion recipes)

A fusion recipe is just another bound workflow (debate, best-of-N, judge-and-revise…).
To score recipe `R` against native: register `R`, point the `r0_agent` at
`workflow:<R_id>`, and the **same** paired-sample + cost machinery yields its delta
vs native — now interpretable, because R0 has established the binding is transparent.
For a recipe you expect to *differ* (better answers), swap the TOST for a
**superiority** test on a quality metric; keep the no-double-charge assertion as-is.

---

# Phase A — the MEASURING harness (does heterogeneous R1 beat the matched-fan-in baseline?)

The foundation proved the binding is transparent. Phase A asks the first REAL fusion
question and answers it with the red-team's non-negotiable controls. **Tier:**
REASONING/CORRECTNESS with programmatically-checkable answers (no LLM-judge — a judge
would itself be a fusion node and confound the question).

## The recipe — R1, a verifier-gated chain

`recipes.py` builds R1 as ONE bound workflow that does, inside a single run:
**Thinker(A) → Worker(B) → Verifier(C)**, with the Worker retrying on a Verifier
REJECT up to `RETRY_CAP=2`. The Verifier sits on a **different substrate than the
Worker** (the substrate-different-verdict invariant — a checker sharing the worker's
blind spots adds correlated noise, not independent detection). All chain calls
accumulate into the run's single `call_llm_cost_microusd`, so reading it after the run
gives the true **$/task for the whole chain** (retries included).

## The three conditions (each bound as `model:"workflow:<id>"`, measured identically)

1. **best-single** — one call, the best single model (R0-shaped). No fan-in.
2. **self-fusion** — the R1 chain with the **best single model in all three roles**.
   THE MATCHED-FAN-IN BASELINE: identical structure/compute/retries, homogeneous substrate.
3. **heterogeneous-R1** — the same chain with heterogeneous models, Verifier ≠ Worker substrate.

The **best single model is chosen by a calibration pass** over the pool (not assumed).

## The honest question (red-team-enforced)

Does **heterogeneous-R1 beat self-fusion** (condition 3 vs 2) at **equal fan-in**, past
a pre-declared MDE, with FWER controlled? A "3 beats 1" win is confounded by fan-in and
is **not** the verdict. A polished "fusion wins" without the self-fusion control is the
exact fraud the red-team killed — this harness refuses to produce it (the primary
comparison IS 3-vs-2).

## Pre-declared (before the data)

| knob | value |
|---|---|
| primary comparison | heterogeneous-R1 vs self-fusion (paired McNemar on per-item correctness) |
| secondaries (Holm-corrected together) | R1 vs best-single; self-fusion vs best-single |
| MDE | +8 percentage points accuracy (the smallest effect worth fusion's Nx cost) |
| alpha | 0.05, Holm-Bonferroni across the 3 comparisons |
| per-item deltas + discordant cells | reported (an aggregate win hiding regressions = a routing bug) |
| $/task | true, from the run-level `call_llm_cost_microusd` meter |

## Stats (`stats_fusion.py`, pure stdlib — validated in `test_stats_fusion.py`)

- **McNemar exact** (binomial tail on the discordant pair) — paired, removes item-difficulty variance.
- **Paired bootstrap CI** on Δaccuracy (resamples item indices, preserving pairing).
- **Holm-Bonferroni** step-down FWER control (no independence assumption).
- **MDE/power note** — leads with the observed CI vs the MDE, not a bare "not significant".

## Model pool (verified reachable on prod 2026-06-30)

Only **Anthropic (ant-proxy)** and **OpenRouter** keys resolve on the worker. Reachable:
`anthropic/claude-opus-4-8` (Anthropic; rejects `temperature` — handled per-role),
`openrouter/openai/gpt-5.5` (OpenAI), `openrouter/moonshotai/kimi-k2.6` (Moonshot),
`openrouter/z-ai/glm-5.2` (Z-AI; **panel-member only, never the verdict**). Gemini/DeepSeek
slugs were not valid on the deployed OpenRouter account — substrate diversity comes from
the three reachable labs. Default heterogeneous roles: A=Opus, B=GPT-5.5 (worker/openai),
C=Kimi (verifier/moonshot) — three distinct substrates, Verifier ≠ Worker.

> **OPS NOTE (separate from the eval — flag for the fleet).** The worker resolves provider
> creds only for **Anthropic** (ant-proxy) and **OpenRouter**. Native-provider model strings
> for OpenAI (`openai/gpt-5.5`), Google (`gemini/...`), DeepSeek, and Moonshot returned
> `AuthenticationError` (no key) — including **GPT-5.5 via the native `openai/` path / oai-proxy,
> which did NOT resolve**; GPT-5.5 only worked routed through `openrouter/openai/gpt-5.5`. If
> the constellation expects first-class multi-provider inference on the aios worker (not just
> via OpenRouter's markup), the oai-proxy / native-provider key wiring on the worker is a gap
> worth a look. This is an infra observation, not an eval result.

## Run it

```bash
export AIOS_URL=https://api.aios.eumemic.ai
export AIOS_API_KEY=<isolated-test-account-key>   # operator over its own account
python3 run_fusion.py --n 30        # calibrate + 3 conditions + Holm-corrected verdict
# pin the best single (skip calibration):
python3 run_fusion.py --n 30 --best-single gpt --skip-calibration
```

Sequential, `--throttle-s` between items (throttle-before-saturation). ~30 items × (4
calibration + best-single + 2 chains) ≈ a few hundred short inferences; budget the test
account accordingly (Phase A ran under a $50 cap).

## Phase A files

- `run_fusion.py` — the measuring harness (calibrate → 3 conditions → McNemar/Holm/bootstrap → honest verdict).
- `recipes.py` — R1 verifier-gated chain + best-single passthrough workflow bodies (model-parameterized).
- `scoring.py` — programmatic answer extraction + checkable-answer scoring (no LLM-judge).
- `stats_fusion.py` — paired-binary stats (McNemar exact, Holm, bootstrap CI, MDE note).
- `test_stats_fusion.py` — validates the stats against known values (incl. a win-hiding-losses case).
- `tasks/reasoning_checkable.json` — 30 checkable-answer reasoning items (easy corpus).
- `tasks/reasoning_hard.json` — 30 harder checkable-answer items (traps/combinatorics/modular/clock-drift).
- `fusion_results.json` — last run's full per-condition record + comparisons + verdict (git-ignored).

## Phase A finding: the REASONING TIER CEILINGS (a tier property, not a fusion result)

The harness + R1 are built and validated end-to-end on prod (R1 chains run, cost
accumulates across the chain into one `call_llm_cost_microusd` = true $/task, no
double-charge — the bound session's own token meter stays 0). But the *substantive
measurement* on the reasoning tier is **null by construction**:

* On the **easy** checkable-reasoning corpus the frontier single models ceiling — opus
  and gpt-5.5 both score **30/30** in calibration.
* On a deliberately **harder** corpus (combinatorics, modular arithmetic, repeating
  decimals, clock-drift, classic traps — all canonical answers programmatically verified)
  the best single models **still ceiling**: opus **30/30**, gpt-5.5 **30/30** in calibration.

When the best single model is already at ~100%, fusion **cannot** beat it — there is no
headroom for the verifier-gate to recover an error the worker didn't make. This is a
**property of the tier**, NOT a failure of fusion or of the harness. With a strong
worker the verifier almost never rejects, so R1 mostly equals best-single plus the
chain's extra cost. Confirming the null further would only burn budget. (Phase A spend
on the test account: ~$1-2 total, well under the $50 cap — the ceiling was found cheaply.)

**Where fusion's value, if any, must live:** tiers where the best single model genuinely
**fails some fraction** — coding **pass@1** on real backlog issues with hidden tests,
and open-ended research/judgement. That is the headroom the matched-fan-in test needs to
have discriminating power. The reasoning tier was the right *first* tier (it proves the
matched-fan-in machinery on an objective, judge-free metric) but it is not where the
fusion question gets answered.

## What Phase B adds (not in this PR)

Coding (pass@1 on real aios/aios-console issues with hidden-test scoring) — the REAL
headroom test — plus the full recipe x tier matrix + the uncorrelated LLM judge (on a
substrate different from every panel member). The matched-fan-in machinery, R1 recipe,
stats, and provisioning proven here **carry forward unchanged**; only the corpus + scorer
swap (the harness already takes `--tasks <file>`; a coding scorer replaces `scoring.py`'s
checkable-answer compare with a hidden-test runner).

---

# Phase B — CODING tier (pass@1 on real aios PRs): scorer + corpus built & validated; headroom CONFIRMED; full 3-condition run BLOCKED on shared OpenRouter credit

Phase B moves the fusion question to a tier with real headroom — coding, where the best
single model genuinely fails a meaningful fraction (unlike reasoning, which ceiling'd).

## What's built & VALIDATED

- **Hidden-test pass@1 scorer** (`coding_scorer.py`): clone aios at the item's
  `base_parent_sha` via `git archive` into a temp dir, overwrite the candidate's
  `src_files`, drop in the held-out `test_files` from the merge commit, run ONLY those
  tests with **`PYTHONPATH=<tmp>/src` first** (so the patched tree shadows the editable
  aios install — without this the test silently scores the canonical checkout, the
  subtlest possible fake-pass), under a per-item timeout. pass@1 = green; non-applying
  patch = fail; env that can't stand up = **SKIP** (logged, excluded — one bad item never
  wedges or biases the run). **Built-in ground-truth self-check**: scoring each item's
  merge-commit source against its held-out test MUST pass — **all 14 corpus items pass**,
  certifying provenance (base/merge SHAs, file paths, test isolatability) before any model
  is scored.
- **Coding corpus** (`tasks/coding_aios.json`): 14 real merged aios PRs (12 HIGH-headroom
  subtle correctness bugs + 2 LOW) whose held-out test is ISOLATABLE (pure-logic pytest,
  no DB/network/docker), each with full provenance (pr#, issue#, base/merge SHA, src/test
  files). 9 are single-file (the clean pass@1 set — a multi-file item can fail merely by
  the model omitting a file, a confound).
- **Coding harness** (`run_coding.py`): the SAME matched-fan-in 3-condition design,
  R1 recipe, McNemar/Holm/bootstrap stats, calibration, and no-double-charge assertion as
  Phase A — only the prompt (`coding_prompt.py`: ask for full corrected file content in a
  path-tagged fence) and scorer swap.

## Finding 1 — the CODING TIER HAS HEADROOM (the headroom Phase A's reasoning tier lacked)

Calibration that completed cleanly (Opus, on Anthropic — no gateway-credit issue):
**pass@1 = 0.375** (3 pass / 5 fail / 1 env-skip on the 9 single-file items). The best
single model genuinely FAILS ~60% of these subtle bugs — exactly the room fusion needs to
have any measurable effect. Per-item discrimination confirmed (c01/c12/c13 pass; c05/c07/
c08/c16/c20 fail; c06 env-skip). **This is the qualitative result Phase A was missing**:
a tier where the matched-fan-in test could, in principle, discriminate.

## Finding 2 (BLOCKER) — the full 3-condition verdict is blocked on shared OpenRouter credit

The substrate-different-verdict invariant REQUIRES a non-Anthropic model for the Verifier
(Verifier ≠ Worker substrate). The only non-Anthropic gateway reachable on the worker is
**OpenRouter** (Phase A finding) — and **OpenRouter's shared key is credit-limited**:
mid-run it began returning HTTP 402 ("requires more credits, or fewer max_tokens"), and the
remaining affordance **shrank 47864 → 18525 → 9749 tokens** as the coding run (large ~20KB
contexts) consumed it. A harness fix (cap `max_tokens=24000` for OpenRouter models —
shipped in `recipes.py`) bought headroom but did not stop the depletion. Continuing would
(a) risk exhausting a SHARED FLEET resource other constellation agents depend on, and
(b) produce a credit-corrupted result (silently-zeroed conditions). So per the
"don't overspend a shared resource / don't fake a result" rule, the het-R1 and self-fusion
conditions (both route through OpenRouter for the heterogeneous Verifier/Worker) were NOT
completed. Test-account spend was ~$6.76 (under the $50 cap) — the blocker is the OpenRouter
credit, not the test-account budget.

**To unblock (ops, not eval):** top up / raise the monthly limit on the worker's OpenRouter
key, OR wire a second non-Anthropic provider with its own credit (the oai-proxy/native
GPT-5.5 path that didn't resolve in Phase A — see that ops note). With either, the SAME
harness + corpus + scorer runs the full 3-condition coding measurement unchanged.

## Secondary note — Opus run-to-run variance

Opus-4-8 runs temperature-free (it rejects `temperature`), so its calibration swings
(pass@1 0.714 on one partial run, 0.375 on another). A clean coding verdict wants either a
temperature-respecting best-single OR multiple seeds per item to average out the variance —
worth pre-declaring when the full run is unblocked.

## Phase B files

- `coding_scorer.py` — hidden-test pass@1 scorer (git-archive sandbox, PYTHONPATH-forced patched tree, SKIP-on-env-break, ground-truth self-check).
- `coding_prompt.py` — coding prompt + path-tagged-fence source extraction.
- `run_coding.py` — the coding measuring harness (same matched-fan-in design as Phase A).
- `tasks/coding_aios.json` — 14 provenance-tagged real-PR coding items (9 single-file).
- `coding_results.json` — last run's record (git-ignored).
