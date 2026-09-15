# aios#2410 fixround (botpost2410j) — DNAT family RED + persist-credentials

PR: https://github.com/eumemic/aios/pull/2410
Branch: `gvisorgrn` tip `2e0225cc`
Fail: https://github.com/eumemic/aios/actions/runs/34822598014
Empty-resolv e2e NOT in FAIL list anymore (DNS-by-arg worked for that). e2e RED is now DNAT family:
- same four swap legs (HTTP_STATUS=000 / empty recorder)
- placeholder KeyError stdout (provision fail symptom)
nslookup-by-arg path may have broken credential swap.

Also bot finding: restore `persist-credentials: false` on checkout (GITHUB_TOKEN in .git/config during agent phase) in eumemic-bot-review workflow.

## Do
1. Fix credential-swap / DNAT functional path so the four swap + placeholder e2e go green without re-breaking the resolv/image-contract win.
2. Restore `persist-credentials: false` on the agent-job checkout.
3. Rebase onto origin/master if behind.

## Constraints
pr_only; do not merge; do not push; no Track G. DONE.md. Docker may be absent — CI oracle.

## Success
e2e(docker) green + posted ### Code review with no blockers.
