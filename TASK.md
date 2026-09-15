# Fixround: #2422 name-based interception breaks lockdown/DNAT verify on CI

PR: https://github.com/eumemic/aios/pull/2422
Branch: `trigswap2` tip `45a054aa`
Failing run: https://github.com/eumemic/aios/actions/runs/34699377660 (e2e docker)

## Root errors (empty recorder / KeyError stdout are symptoms of provision abort)
1. Limited: `SandboxBackendError: network lockdown verification failed … OUTPUT policy is not DROP after apply`
2. Unrestricted: `SandboxBackendError: secret-egress DNAT verification failed … nat OUTPUT carries no DNAT rule after apply`

FAILED tests include:
- test_trigger_swap_fires_under_limited / unrestricted_dnat_only
- test_run_swap_fires_under_limited / unrestricted_dnat_only
- test_run_bash_env_var_placeholder_round_trip
- test_placeholder_visible_in_container_secret_absent

Leave #2421 closed. Do not reopen IPv6/`-4` as the master clear.

## Goal
Make name-based credential interception (#2042 rebase) actually apply + verify on CI so Limited DROP and Unrestricted/Limited DNAT chokepoint rules land. Both trigger-swap legs AND the run-origin placeholder/swap e2e must go green.

## Investigate
- Why does `apply_network_lockdown` / `apply_secret_egress_dnat` leave filter OUTPUT not DROP or nat without DNAT? (apply script abort early — proxy alias resolve miss, `dns_port` missing, iptables backend, `-I` DNS DNAT vs Docker 127.0.0.11, sentinel `169.254.53.53` conflicting with link-local/metadata rules, verify grep mismatch vs `iptables -S` output, credential_dns not bound so provision refuses incorrectly, etc.)
- Prefer fixing the product apply/verify path so the chokepoint is real; do not weaken fail-closed verification.
- Rebase onto latest origin/master if behind; push is Shepherd’s job.

## Constraints
- No Track G / Coolify / merge / push
- PR-only; continue on `trigswap2` (update #2422)
- Docker may be absent locally — unit tests for script generation + any integration you can run; CI is oracle for docker e2e
- DONE.md with evidence-backed root cause of the verify failure

## Success
New tip on #2422 expected to clear the lockdown/DNAT verify failures and the listed e2e tests.
