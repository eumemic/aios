# Gate acceptance

These three gate runs show the WaM deploy gate gives the verdict it should in three
cases where the right verdict is known in advance. They are what retired the external
`wam_fusion` harness.

| case | candidate | bar | expected |
|---|---|---|---|
| R0 ≈ native | R0 on the agent's own model | `wam-gate` | PASS |
| known-worse | R0 on a much weaker model | `wam-gate` | FAIL |
| blind judge | R0 on the agent's own model | `wam-gate-tail3` (judge sees 3 messages) | INVALID |

## Record (prod, 2026-10-12)

Account `acc_9f36880d…` (Eumemic). Agent `eval-probe` (`agent_01M4Q2WZTZ0X9MG3SD5T0D2F3K`,
`anthropic/claude-opus-5-5`). Judge `openrouter/openai/gpt-6-luna`. Seed 1. Budget $400
each. Gate and eval workflows at version 1.

| case | run | verdict | n / clusters | W [lower, upper] | control W [upper] | cost ratio | p95 latency ratio |
|---|---|---|---|---|---|---|---|
| R0 ≈ native | `wfr_01M4Q7SDDHT4GQB60D2E5NA1BV` | **PASS** | 208 / 138 | 0.500 [0.475, 0.525] | 0.059 [0.093] | 1.01 | 1.07 |
| known-worse (Haiku 4.5) | `wfr_01M4Q97DNV9P421ZXJ7SRTP4FE` | **FAIL** (win rate) | 208 / 138 | 0.154 [0.123, 0.185] | 0.026 [0.048] | 0.20 | 0.93 |
| blind judge | `wfr_01M4QAKK8CXXPQ537VYE4XX7TK` | **INVALID** (control not worse) | 207 / 137 | 0.510 [0.487, 0.532] | 0.915 [0.945] | 1.00 | 0.99 |

- **PASS.** No degenerate turns or invalid tool calls in any run (exact upper bound 1.4%).
  The control's tie rate was 9%.
- **FAIL.** The weak arm answered every item. Its `max_tokens` is clamped, so it lost
  on quality, not on errors. It is also a fifth of the cost: the gate fails it on the
  win rate alone.
- **INVALID.** The tail-3 judge doesn't see the rule. A reply that followed it looks
  inconsistent (its working gives one value, its ANSWER line another), so the judge
  favours the control: its win rate against the baseline is 0.915. The gate refuses to trust a judge that can't
  show the control worse, even though the candidate pair looks fine (W 0.51). One
  item was excluded for a judge failure on the candidate's reply; it counts as a loss
  in the worst-case bound (0.507).

Each run took about 25 minutes.

## The probe corpus

The gate replays an agent's real requests. In prod, no agent had enough of them since
request capture began (two per (session, day), at least 208 items), so the runs sample a
synthetic agent, `eval-probe` (`probe_corpus.py`): 150 sessions of six turns each, drawn
from the reasoning problems in `../suites/`.

From the third turn on, every problem is governed by a rule given two turns earlier
("report the final answer plus 100"), which the agent applies silently. That rule is
what makes the three verdicts known in advance:

- The negative control sees only the system prompt and the last user turn. It misses
  the rule and answers plausibly but wrongly.
- The default judge sees 8 messages, so it sees the rule and finds the control worse:
  the gate is valid.
- The tail-3 judge sees 3 messages, so it can't see the rule: the gate is INVALID.

A control item needs the judge's tail to be strictly longer than the control's view and
strictly shorter than the whole request. In a 6-turn session that holds only for later
turns, so about a third of the items carry a control arm (76 and 135 here). That was
enough, because the control loses nearly every one.

## Re-running

A gate samples only requests sent after its candidate version was created. Register
the candidates first, then build the corpus, then launch.

```bash
export AIOS_URL=https://api.aios.eumemic.ai AIOS_API_KEY=<operator key>
uv run python -m evals.register evals/bars/*.json evals/acceptance/wam_gate_tail3.json
uv run python -m evals.acceptance.candidates
uv run python -m evals.acceptance.probe_corpus --environment-id ENV_ID --sessions 150
# after the corpus is built, and after the next full hour (the window ends on the hour):
uv run python -m evals.gate launch --agent PROBE_AGENT_ID --baseline-model anthropic/claude-opus-5-5 \
    --candidate R0_OPUS@V --environment-id ENV_ID --seed 1 --budget-usd 400
uv run python -m evals.gate launch ... --candidate R0_HAIKU@V
uv run python -m evals.gate launch ... --candidate R0_OPUS@V --gate wam-gate-tail3
```

Run the gates one at a time; each holds about 37 runs at its peak. Avoid the nightly
database backup (03:00 UTC).
