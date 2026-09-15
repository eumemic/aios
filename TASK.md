# aios#2410 — botpost2410l: real credential-swap / DNAT fix (hosts vs DNS)

## Context
Prior tip `e07df125` on `gvisorgrn` / https://github.com/eumemic/aios/pull/2410 is a FAIL review docs-only tip. Prior j-round `d27c242d` only landed `persist-credentials: false`. Zero Python changed for the DNAT/swap regression.

CI e2e(docker) still RED: `nat OUTPUT carries no DNAT rule after apply` + six credential-swap / placeholder failures (run `34919782033`).

## Root cause (Opus, pinned)
`9b246ab7` replaced `getent` with `busybox nslookup`, which is DNS-only and never reads `/etc/hosts`. In the e2e host-gateway shape, `aios-worker` exists only via `--add-host`, so `PROXY_IP` is empty, the DNAT block is skipped, apply still exits 0, and verify fails.
Bisect: `e10f4e07` swap family green; `2e0225cc` red.

## Chosen fix shape (Shepherd: smallest / restore green)
**Shape 1:** Restore NSS/hosts-file lookup in `resolve_ipv4` (hosts first, then busybox nslookup). Return to e10f4e07-green behaviour. Document tenant-writable hosts caveats in comments if relevant.
Do **not** do shape 2 (IP injection redesign) unless shape 1 is clearly wrong after inspection.

## Also
- Move DNS oracles into sidecar-after-flush / netns-joining context if still asserting the wrong context.
- Rebase onto `origin/gvisorgrn` tip (or master if needed) before work; keep #2410 branch `gvisorgrn`.
- Write a **true** DONE.md for this round (not stale botpost2410i).
- Do not push; Shepherd pushes.
- Uncorrelated review required before PR update.

## Success
- Python/product change that makes proxy alias resolve when only in `/etc/hosts` (or `--add-host`).
- DNAT rules land after apply (no empty PROXY_IP skip).
- Local unit coverage for hosts-first resolve if feasible.
- DONE.md accurate.
