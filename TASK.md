# aios#2410 merge-path fixround (botpost2410i)

PR: https://github.com/eumemic/aios/pull/2410
Branch: `gvisorgrn` tip `e10f4e07`
Requester: AIOS Bot

## Do
1. **Fix e2e that botpost2410h did not clear**: layer `/etc/resolv.conf` present but **empty**.
   Fail: https://github.com/eumemic/aios/actions/runs/34807834078/job/103863091603
   `test_image_layer_carries_the_embedded_dns_resolver` — content `''`.
   `COPY --link` failed the CI oracle. Tip docs already warn same-path bake may be dead.
   Make `docker cp` from a never-started container read `nameserver 127.0.0.11`, **or** fix the operator/chroot read path if baking to `/etc/resolv.conf` is impossible under BuildKit (non-special bake path + ensure getent in runsc chroot sees it).

2. **Fix fresh bot finding**: derive runsc operator image mount from the **same image as the tenant container** (`spec.image`), or reject mismatch — not `get_settings().docker_image` alone (`docker.py` ~306).

## Success
- e2e(docker) green on the image-contract resolv test
- Fresh ### Code review with no blocking findings
- Mergeable
- pr_only — do not merge; do not push (Shepherd pushes); no Track G
- DONE.md with evidence-backed resolv approach (not another unverified --link bet)

Rebase onto origin/master if behind. Keep prior #2410 security/harness work.
