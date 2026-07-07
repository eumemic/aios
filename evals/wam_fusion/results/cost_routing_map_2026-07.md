# $0 Cost-Routing Map — 2026-07

_Generated 2026-07-07 09:52 from the cost-map matrix (chairman-approved pre-work for the parked cost-routing option, D-Q24 pivot)._

**Design**: 266 items (46 hidden-test coding + 140 reasoning-headroom + 80 reasoning-extrahard) × 6 core models + 1 deferred reference column × k=3 provider-default-decoded samples, all via the company's subsidized gateways (ant-proxy / oai-proxy; $0 cash, zero OpenRouter calls). Coding is scored by local execution of each item's held-out golden tests; reasoning by exact-match with a numeric normalizer.

**Column status**: the 6 core columns (opus-4-8, sonnet-5, haiku-4-5, gpt-5.5, gpt-5.4-mini, gpt-5.4) are the mandated map. `claude-fable-5` (the most-premium frontier model, shared with production) was DEFERRED per the run's capacity guidance: its weekly pool window was exhausted when this run began. It carries 168 PENDING-WINDOW cells (never dispatched) — the reasoning corpora were collected opportunistically before the deferral and are reported as a partial reference; its coding column is entirely PENDING-WINDOW and fills on a later run once the window resets. Read fable-5 rows as reference-only.

**Column substitution**: `gpt-5.3-codex` was requested but the codex backend rejects it for ChatGPT accounts (probed 2026-07-06, both API shapes); `gpt-5.4` is the substitute third OpenAI column.

## Matrix completion

- expected cells: 5586 (266 items × 7 models × k=3)
- scored: 5399, holes: 19, pending (never dispatched): 168, excluded by golden self-check: 0
- coding items excluded by golden self-check (oracle can't stand up in this env today — excluded for ALL models): none

| model | calls | ok | natural-stop | truncation holes | other holes | pending | identity mismatches | repaired |
|---|---|---|---|---|---|---|---|---|
| opus-4-8 | 798 | 798 | 100.00% | 0 | 0 | 0 | 0 | 0 |
| sonnet-5 | 798 | 797 | 100.00% | 1 | 0 | 0 | 0 | 0 |
| haiku-4-5 | 798 | 798 | 100.00% | 0 | 0 | 0 | 0 | 0 |
| fable-5 | 630 | 612 | 99.35% | 0 | 18 | 168 | 0 | 4 |
| gpt-5.5 | 798 | 798 | 100.00% | 0 | 0 | 0 | 0 | 0 |
| gpt-5.4-mini | 798 | 798 | 100.00% | 0 | 0 | 0 | 0 | 0 |
| gpt-5.4 | 798 | 798 | 100.00% | 0 | 0 | 0 | 0 | 0 |

## 1. Accuracy table (k-sample means, item-bootstrap 95% CIs, B=1000)

| model | tier | headroom (n=140) | extrahard (n=80) | coding (n=46) |
|---|---|---|---|---|
| opus-4-8 | frontier | 100.0% [100.0%, 100.0%] (maj 100.0%) | 97.5% [94.6%, 99.6%] (maj 97.5%) | 38.4% [25.4%, 52.2%] (maj 39.1%) |
| sonnet-5 | cheap | 100.0% [100.0%, 100.0%] (maj 100.0%) | 97.9% [94.6%, 100.0%] (maj 98.8%) | 46.4% [32.6%, 60.1%] (maj 47.8%) |
| haiku-4-5 | cheap | 99.0% [98.1%, 99.8%] (maj 100.0%) | 87.1% [80.4%, 93.3%] (maj 88.8%) | 28.3% [17.4%, 39.9%] (maj 28.3%) |
| fable-5 | frontier | 100.0% [100.0%, 100.0%] (maj 100.0%) | 97.6% [94.3%, 100.0%] (maj 98.6%) | — |
| gpt-5.5 | frontier | 100.0% [100.0%, 100.0%] (maj 100.0%) | 99.2% [97.9%, 100.0%] (maj 100.0%) | 39.9% [26.8%, 52.9%] (maj 37.0%) |
| gpt-5.4-mini | cheap | 98.1% [96.4%, 99.5%] (maj 98.6%) | 92.1% [87.5%, 96.2%] (maj 93.8%) | 34.8% [21.7%, 47.1%] (maj 37.0%) |
| gpt-5.4 | cheap-sub | 100.0% [100.0%, 100.0%] (maj 100.0%) | 98.8% [97.1%, 100.0%] (maj 100.0%) | 41.3% [28.3%, 55.1%] (maj 39.1%) |

Per coding stratum (majority-of-k accuracy):

| model | multi-attempt (n=4) | standard (n=39) | thin-spec (n=3) |
|---|---|---|---|
| opus-4-8 | 25.0% | 35.9% | 100.0% |
| sonnet-5 | 25.0% | 46.2% | 100.0% |
| haiku-4-5 | 25.0% | 25.6% | 66.7% |
| fable-5 | — | — | — |
| gpt-5.5 | 25.0% | 33.3% | 100.0% |
| gpt-5.4-mini | 25.0% | 33.3% | 100.0% |
| gpt-5.4 | 25.0% | 35.9% | 100.0% |

## 2. Cheap-coverage table (the headline)

For each cheap model vs the per-corpus frontier best: % of items where the cheap model matches-or-beats it (majority-of-k), the accuracy delta (cheap − frontier), and the safe-lane flag (Δ ≥ −2pp; CI-supported when the bootstrap CI lower bound also clears −2pp).

| corpus | cheap model | vs frontier best | coverage % | Δacc (maj) | 95% CI | flag |
|---|---|---|---|---|---|---|
| headroom | haiku-4-5 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE (CI-supported) |
| headroom | sonnet-5 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE (CI-supported) |
| headroom | gpt-5.4-mini | opus-4-8 | 98.6% | -1.4% | [-3.6%, 0.0%] | PROVISIONAL (CI crosses) |
| headroom | gpt-5.4 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE (CI-supported) |
| extrahard | haiku-4-5 | gpt-5.5 | 88.8% | -11.2% | [-17.5%, -5.0%] | NOT SAFE |
| extrahard | sonnet-5 | gpt-5.5 | 98.8% | -1.2% | [-3.8%, 0.0%] | PROVISIONAL (CI crosses) |
| extrahard | gpt-5.4-mini | gpt-5.5 | 93.8% | -6.2% | [-11.2%, -1.2%] | NOT SAFE |
| extrahard | gpt-5.4 | gpt-5.5 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE (CI-supported) |
| coding | haiku-4-5 | opus-4-8 | 89.1% | -10.9% | [-21.7%, -4.3%] | NOT SAFE |
| coding | sonnet-5 | opus-4-8 | 100.0% | 8.7% | [2.2%, 17.4%] | SAFE (CI-supported) |
| coding | gpt-5.4-mini | opus-4-8 | 93.5% | -2.2% | [-13.0%, 6.5%] | NOT SAFE |
| coding | gpt-5.4 | opus-4-8 | 97.8% | 0.0% | [-6.5%, 6.5%] | PROVISIONAL (CI crosses) |

Per coding stratum:

| stratum | cheap model | vs | coverage % | Δacc | 95% CI | flag |
|---|---|---|---|---|---|---|
| multi-attempt | haiku-4-5 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| multi-attempt | sonnet-5 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| multi-attempt | gpt-5.4-mini | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| multi-attempt | gpt-5.4 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| standard | haiku-4-5 | opus-4-8 | 89.7% | -10.3% | [-20.5%, -2.6%] | NOT SAFE |
| standard | sonnet-5 | opus-4-8 | 100.0% | 10.3% | [2.6%, 20.5%] | SAFE |
| standard | gpt-5.4-mini | opus-4-8 | 92.3% | -2.6% | [-12.8%, 7.7%] | NOT SAFE |
| standard | gpt-5.4 | opus-4-8 | 97.4% | 0.0% | [-7.7%, 7.7%] | PROVISIONAL |
| thin-spec | haiku-4-5 | opus-4-8 | 66.7% | -33.3% | [-100.0%, 0.0%] | NOT SAFE |
| thin-spec | sonnet-5 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| thin-spec | gpt-5.4-mini | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |
| thin-spec | gpt-5.4 | opus-4-8 | 100.0% | 0.0% | [0.0%, 0.0%] | SAFE |

## 3. Frontier comparison (per-lane default evidence)

| corpus | frontier best (majority) | pairwise Δ (majority, 95% CI) |
|---|---|---|
| headroom | opus-4-8 (100.0%) | opus-4-8−gpt-5.5: 0.0% [0.0%, 0.0%]; opus-4-8−fable-5: 0.0% [0.0%, 0.0%]; gpt-5.5−fable-5: 0.0% [0.0%, 0.0%] |
| extrahard | gpt-5.5 (100.0%) | opus-4-8−gpt-5.5: -2.5% [-6.2%, 0.0%]; opus-4-8−fable-5: -1.4% [-5.7%, 2.9%]; gpt-5.5−fable-5: 1.4% [0.0%, 4.3%] |
| coding | opus-4-8 (39.1%) | opus-4-8−gpt-5.5: 2.2% [0.0%, 6.5%] |

## 4. Metered-era savings simulation (LIST-price estimates)

Prices ($/MTok in, out — 2026-07 list, estimates; we currently pay $0 via subsidized gateways): opus-4-8 (5.0, 25.0), sonnet-5 (3.0, 15.0), haiku-4-5 (1.0, 5.0), fable-5 (10.0, 50.0), gpt-5.5 (5.0, 30.0), gpt-5.4-mini (0.75, 4.5), gpt-5.4 (2.5, 15.0).

Mean measured tokens per call and cost per call by corpus:

| model | headroom $/call (in/out tok) | extrahard $/call (in/out tok) | coding $/call (in/out tok) |
|---|---|---|---|
| opus-4-8 | $0.0046 (213/142) | $0.0118 (240/423) | $0.3453 (11746/11464) |
| sonnet-5 | $0.0043 (213/244) | $0.0196 (240/1258) | $0.2403 (11554/13711) |
| haiku-4-5 | $0.0025 (155/467) | $0.0041 (180/780) | $0.0526 (8902/8734) |
| fable-5 | $0.0113 (214/183) | $0.0303 (239/558) | — |
| gpt-5.5 | $0.0065 (137/193) | $0.0156 (159/493) | $0.2285 (7288/6403) |
| gpt-5.4-mini | $0.0012 (137/253) | $0.0041 (159/876) | $0.0496 (7288/9803) |
| gpt-5.4 | $0.0036 (137/216) | $0.0094 (159/598) | $0.1254 (7288/7144) |

Routing each stratum to its **cheapest adequate model** (majority accuracy ≥ frontier best − 2pp) vs all-frontier:

| corpus/stratum | n | frontier (acc, $/call) | routed to (acc, $/call) | per-call saving |
|---|---|---|---|---|
| coding/multi-attempt | 4 | opus-4-8 (25.0%, $0.3453) | gpt-5.4-mini (25.0%, $0.0496) | 85.6% |
| coding/standard | 39 | opus-4-8 (35.9%, $0.3453) | gpt-5.4 (35.9%, $0.1254) | 63.7% |
| coding/thin-spec | 3 | opus-4-8 (100.0%, $0.3453) | gpt-5.4-mini (100.0%, $0.0496) | 85.6% |
| extrahard/all | 80 | gpt-5.5 (100.0%, $0.0156) | gpt-5.4 (100.0%, $0.0094) | 39.8% |
| headroom/all | 140 | opus-4-8 (100.0%, $0.0046) | gpt-5.4-mini (98.6%, $0.0012) | 73.1% |

**Overall simulated cost reduction at quality parity: 65.3%** (item-weighted; all-frontier baseline = each corpus's frontier-best model).

## 5. Honest caveats

- **k=3 sampling noise**: per-item majority verdicts flip easily near 50%; the bootstrap CIs quantify item-sampling error, not draw-sampling error.
- **46-item coding corpus is small** — CIs are wide; stratum slices (4 multi-attempt, 3 thin-spec items) are anecdotal, not statistical.
- **Training-data contamination**: `eumemic/aios` is a PUBLIC repo; coding items are real merged PRs and may be in model training data. This inflates absolute coding scores for ALL models; the differential risk across vendors/cutoffs is noted but unmeasured.
- **Corpus vintage + model list are dated 2026-07**; the map decays with model-lineup churn (historically ~2–4 pool events/month on our gateways).
- **gpt-5.4 substitutes for gpt-5.3-codex** (upstream-rejected for ChatGPT accounts); no cheap OpenAI coding-tuned column was reachable at $0.
- **List prices are estimates**, present-day published rates; subscription-window economics (what we actually ride) are not per-token and could diverge arbitrarily.
- **Provider-default decoding** means k=3 draws are not independent for models that decode near-deterministically (see the entropy ledger below) — for those, majority-of-k ≈ single-draw accuracy.
- **Reasoning corpora are own-authored** (140+80 items, verified answers); they proxy 'careful multi-step reasoning', not any production lane directly.
- **Coding oracle tests are white-box**: several held-out tests import private helpers/constants from the rewritten module by exact name, so a candidate must reproduce the merged PR's internal interface, not just its behavior (drop-in-replacement fidelity). Collection errors on candidate runs are scored INTERFACE-FAIL (a real failure, not a hole) because a same-day golden self-check proved every item's oracle stands up with the true sources on this machine.
- **Truncation-holed cells were re-administered once at a doubled max_tokens cap** (--retry-truncations; the cap is administration, not construct). Any cell still truncating at the raised cap remains a HOLE.

## 6. Administration-ledger appendix (Stage 0 + matrix-wide)

Stage-0 verdict: **PASS** (2026-07-06T23:20:39)

### stage0 / identity: PASS
```json
{
 "mismatches": {}
}
```

### stage0 / natural_stop: PASS
```json
{
 "rates": {
  "opus-4-8": 1.0,
  "sonnet-5": 1.0,
  "haiku-4-5": 1.0,
  "fable-5": 1.0,
  "gpt-5.5": 1.0,
  "gpt-5.4-mini": 1.0,
  "gpt-5.4": 1.0
 },
 "note": "matrix-wide rate re-checked at analysis; this is the stage0 sample"
}
```

### stage0 / extraction_calibration: PASS
```json
{
 "parse_rates": {
  "opus-4-8": 1.0,
  "sonnet-5": 1.0,
  "haiku-4-5": 1.0,
  "fable-5": 1.0,
  "gpt-5.5": 1.0,
  "gpt-5.4-mini": 1.0,
  "gpt-5.4": 1.0
 },
 "cross_model_spread": 0.0,
 "contract": "corpus ANSWER: <value> final line + one deterministic repair re-ask"
}
```

### stage0 / holes: PASS
```json
{
 "per_model": {
  "opus-4-8": 0,
  "sonnet-5": 0,
  "haiku-4-5": 0,
  "fable-5": 0,
  "gpt-5.5": 0,
  "gpt-5.4-mini": 0,
  "gpt-5.4": 0
 },
 "truncations": {
  "opus-4-8": 0,
  "sonnet-5": 0,
  "haiku-4-5": 0,
  "fable-5": 0,
  "gpt-5.5": 0,
  "gpt-5.4-mini": 0,
  "gpt-5.4": 0
 }
}
```

### stage0 / entropy_probe: PASS
```json
{
 "per_model": {
  "opus-4-8": {
   "r010": {
    "draws": 8,
    "unique_texts": 7,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  },
  "sonnet-5": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 3,
    "unique_answers": 2,
    "answer_agreement": 0.875
   }
  },
  "haiku-4-5": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  },
  "fable-5": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  },
  "gpt-5.5": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 7,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  },
  "gpt-5.4-mini": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  },
  "gpt-5.4": {
   "r010": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "r070": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   },
   "x040": {
    "draws": 8,
    "unique_texts": 8,
    "unique_answers": 1,
    "answer_agreement": 1.0
   }
  }
 },
 "note": "ledger only: determinism is valid, just recorded"
}
```

### Matrix-wide administration (per model)

```json
{
 "opus-4-8": {
  "calls": 798,
  "ok": 798,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 },
 "sonnet-5": {
  "calls": 798,
  "ok": 797,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 1,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 },
 "haiku-4-5": {
  "calls": 798,
  "ok": 798,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 },
 "fable-5": {
  "calls": 630,
  "ok": 612,
  "natural_stop_rate": 0.9934640522875817,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 18,
  "repaired": 4,
  "quarantined": 0
 },
 "gpt-5.5": {
  "calls": 798,
  "ok": 798,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 },
 "gpt-5.4-mini": {
  "calls": 798,
  "ok": 798,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 },
 "gpt-5.4": {
  "calls": 798,
  "ok": 798,
  "natural_stop_rate": 1.0,
  "identity_mismatches": 0,
  "truncation_holes": 0,
  "other_holes": 0,
  "repaired": 0,
  "quarantined": 0
 }
}
```

Raw per-cell matrix: `cost_routing_matrix_2026-07.json` (texts stripped; full transcripts remain in the local state dir, not committed).