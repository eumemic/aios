# aios#2410 fixround (botpost2410k) — REAL credential-swap/DNAT fix

PR: https://github.com/eumemic/aios/pull/2410
Branch: `gvisorgrn` tip ~`d27c242d` / code `f3c847d6`
Prior round botpost2410j: Opus FAIL — only landed `persist-credentials: false`. Zero Python change. Do not claim DONE for item 1 again without a real fix.

## Root cause to fix (from review / CI)
After busybox nslookup-by-arg (`9b246ab7`), CI shows `nat OUTPUT carries no DNAT rule after apply` and the six credential-swap / placeholder e2e fail (same family as #2422: HTTP_STATUS=000 / empty recorder / KeyError stdout). Bisect: `e10f4e07` had swap family green; `2e0225cc` red.

## Do
1. Actually fix credential-swap / DNAT resolution so rules land after apply and the four swap + placeholder e2e can go green — without re-breaking the empty-resolv / image-contract win if still green.
2. Move DNS oracles into the sidecar-after-flush / netns-joining context (not plain `--network` container).
3. Replace DONE.md with a true report of THIS round.
4. Keep `persist-credentials: false` from j.
5. Rebase onto origin/master if behind.

## Constraints
pr_only; do not merge; do not push (Shepherd pushes); no Track G. Docker may be absent — CI is oracle.

## Success
Real Python/product change that addresses DNAT verify failure; tip ready for e2e green + clean ### Code review.
