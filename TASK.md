# Follow-up: make trigger-swap e2e actually green (beyond #2421 `-4`)

PR #2421 (https://github.com/eumemic/aios/pull/2421, tip `81563e78`) only adds curl `-4` hygiene. Uncorrelated review **rejected** the IPv6-as-MASTER-RED root cause and documented the #2042 A-record subset / re-resolve miss. `-4` alone is expected **not** to clear MASTER RED.

## Your job
Land the durable fix so both legs go green on Code Validation:
- `tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_unrestricted_dnat_only`
- `tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_limited`

Base: **origin/master** (fetch tip). Optionally cherry-pick / rebase onto `#2421`’s `-4` if still useful as hygiene — or supersede `#2421` with one PR that actually fixes RED.

## Investigate carefully (do not rubber-stamp either story)
1. `#2042` already shipped **name-based** credential-host interception (`credential_dns` + sentinel `169.254.53.53`). If that path is live for Unrestricted+Limited session provision used by `run_trigger_step`, A-subset re-resolve should already be impossible for `api.github.com`. Prove whether credential DNS / name DNAT is actually installed on the trigger-fire sandbox path.
2. If name-based path is broken/missing for trigger fires (or Unrestricted skips DNS redirect), **fix the product path** — that is the durable fix. Prefer that over weakening the e2e with `--resolve` (the test deliberately avoids `--resolve` to exercise the real chokepoint; see comments in the file and `test_run_env_var_placeholder.py`).
3. If the product path is correct and the only gap is IPv6 AAAA bypass of IPv4-only DNAT, keep/strengthen `-4` (or equivalent) and prove it; update comments to match evidence.
4. Compare: did `test_run_env_var_placeholder.py` Unrestricted/Limited legs still pass on the failing Code Validation run? If run-origin passes and trigger-origin fails, the bug is in the trigger fire / provision seam, not generic DNAT.

## Constraints
- No Track G / Coolify / merge / push
- PR-only; Shepherd pushes
- Do not weaken secret-egress / MitM / DNAT security
- Docker may be absent — unit/integration you can run; CI is oracle for docker e2e
- DONE.md with evidence-backed root cause (not the discarded false IPv6-only story unless you re-prove it)

## Success
Open PR (or update) whose tip is expected to make those two e2e tests green.
