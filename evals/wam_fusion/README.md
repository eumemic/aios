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
