# aios#2410 merge-path fixround (botpost2410h)

PR: https://github.com/eumemic/aios/pull/2410
Branch: `gvisorgrn` tip `96b7369f` — DIRTY/CONFLICTING vs master (~5 commits behind).
Requester: AIOS Bot

## Do
1. **Rebase onto current origin/master** so the PR is mergeable/clean.
2. **Fix real e2e fail**: layer `/etc/resolv.conf` present but **empty** after `COPY --link`.
   Signature: https://github.com/eumemic/aios/actions/runs/34666903810/job/103480557877
   `tests/e2e/test_sandbox_image_contract.py::test_image_layer_carries_the_embedded_dns_resolver`
   — `docker cp` from never-started container finds no `nameserver 127.0.0.11`.
   Prior `COPY --link` (tip `96b7369f`) did **not** clear it. Either make the bake survive docker-cp from a never-started container naming `127.0.0.11`, **or** if same-path bake to `/etc/resolv.conf` is dead under BuildKit, bake to a non-special path and fix the runsc operator/chroot read path so `getent` still sees the embedded DNS (Dockerfile comments already note this escape hatch).
3. **Resolve or fail-closed** eumemic-bot finding: `src/aios/sandbox/backends/docker.py` hardcodes
   `_RUNSC_OPERATOR_LOADER = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"` and
   `_RUNSC_OPERATOR_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu"` — select loader/lib paths per image arch, **or refuse runsc on arm64**.

## Success
- PR mergeable/clean
- e2e (docker) green on the image-contract resolv test (and no regress of gVisor / review-harness work)
- Fresh ### Code review with no blocking findings (CI/bot after push)
- pr_only — do not merge; do not push (Shepherd pushes); no Track G
- DONE.md with root cause of empty resolv + arch fix choice

## Notes
Keep the two-job eumemic-bot publish isolation and other #2410 security work intact through the rebase.
