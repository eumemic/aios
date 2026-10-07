# evals

Evals run in aios as workflows. Core never reads a score. Every number and verdict
is computed by the eval workflows here and stored in their runs' outputs.

- `workflows/` holds the eval workflow templates (see `workflows/__init__.py`).
- `bars/` holds the bars the gate is registered with.
- `register.py` registers the workflows in an account.
- `gate.py` launches a gate, reports its verdict and re-runs its analysis.
- `auto_review/` is the corpus for the MCP auto-review checker.
- `wam_fusion/` is the external harness the gate replaces.

Run the scripts from the repo root, with an operator key for the account that holds
the agent:

```bash
export AIOS_URL=https://api.aios.eumemic.ai AIOS_API_KEY=<operator key>
uv run python -m evals.register                      # prints each workflow's id@version
uv run python -m evals.gate launch --agent AGENT_ID --baseline-model MODEL \
    --candidate WF_ID@VERSION --environment-id ENV_ID --seed 1 --budget-usd 300 --dry-run
uv run python -m evals.gate report RUN_ID
```

## The WaM deploy gate (`wam-gate`)

The gate asks one question. Is the candidate workflow, bound as the agent's model,
non-inferior to the baseline model on the agent's real requests? It replays requests
the agent sent after the candidate version was created, so recipe authors can't tune
on gate data.

It runs three arms on every sampled request, each a sub-run acting for the agent
(`as_agent`) and holding only the request's ref:

- **baseline:** `eval-r0` sends the request to the baseline model;
- **candidate:** the pinned candidate workflow starts with the request, the way a
  workflow bound as the agent's model does;
- **negative control:** `eval-r0` sends the baseline model only the system prompt and
  the last user turn.

A judge from a different model family compares the candidate and the control with
the baseline, in both orders. It sees the system prompt, the last `judge.tail`
messages and both replies.

**The verdict** is PASS, FAIL, INVALID or INCONCLUSIVE, with the statistics behind
it:

- **PASS** needs all of these:
  - the cluster-robust lower bound of the candidate's win rate (a tie counts half) is
    at least `0.5 - delta`;
  - every limit holds:
    - degenerate turns and invalid tool calls, each an exact one-sided bound on the
      items where only the candidate fails;
    - the uncached cost ratio;
    - the p95 latency ratio.
- **INVALID** means the judge can't be trusted, for any of these reasons:
  - it shares a family with a model an arm used;
  - the negative control isn't shown worse than the baseline (the control's upper
    bound must be under 0.5);
  - it ties too many control pairs.
- **INCONCLUSIVE** means the run can't decide, for any of these reasons:
  - too few clusters, overall or for the control;
  - too many excluded items;
  - a budget stop or a full run cap;
  - too few eligible items for the bar's power.

The run refuses before spending anything when the corpus is too small. A verdict is
advisory. After a PASS, the deploy is one `PUT` of the agent's `model` to the
`workflow:<id>@<version>` the verdict names.

### The bar (`bars/wam_gate.json`)

The bar is baked into the registered `wam-gate` version, so changing it takes a new
version.

| field | meaning |
|---|---|
| `delta`, `alpha` | non-inferiority margin on the win rate, one-sided level of every bound |
| `power` | joint power the sample must reach under `planning`, checked before anything is spent |
| `sample_size`, `cluster_cap` | requests drawn (at most 500), at most this many per (session, UTC day) |
| `min_clusters`, `max_exclusion_rate` | below or above these, the verdict is INCONCLUSIVE |
| `wave_size`, `max_attempts` | items in flight at once (keep `wave_size × 6` well under the account's run cap); retries of an item the eval's own load hit |
| `bootstrap_rounds` | resamples for the p95 latency bound |
| `limits` | `degenerate` and `tool_calls` (rates), `cost` and `latency` (allowed ratio minus one) |
| `control` | `min_clusters` and `tie_ceiling` for the negative control |
| `judge` | `model` (its family must differ from every arm's), `params`, `tail` (messages shown), `max_chars` per message |
| `budget` | per-item spend ceilings: `arm_usd` (baseline, control), `candidate_usd`, `judge_usd` |
| `planning` | the rates the power check assumes, and the launcher's cost and run-slot estimates |

### Scope

- The candidate may use `call_llm`, `invoke_workflow` and `agent()`. An arm acts
  within the gate's surface, which holds only the replay tools. So the launcher
  refuses a candidate that declares tools or servers, and an `agent()` child of the
  candidate runs with no tools.
- The latency limit compares candidate and baseline arm runs. It doesn't include the
  park and harvest overhead a bound model adds in production.
- Uncached pricing hides a recipe whose only cost is breaking the prompt cache.
