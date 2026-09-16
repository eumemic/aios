# REVIEW — round `gvisred3`, tip `5d94b476909f19d13358009d30f20a8ec7ec25bb`

Checker: claude-opus-5 (uncorrelated; implementer was grok-4.6 on `gvisred3`).
Branch under review: `gvisred3rev` (forked from the implement tip). Base:
`origin/master` @ `f5c22254` (#2434). Evidence:
https://github.com/eumemic/aios/actions/runs/35161851660 — 4 failed / 372 passed.
Not pushed, not merged, no PR opened.

## Verdict: **FAIL** — both mechanisms are right, two shipped claims are not.

The tip fixes the RED legs. It also asserts, in four places, a security
invariant that its own change breaks, and it fixes leg (b) only for images that
never flatten. Both are High/Medium and both are **fixed in this worktree** as
`1a588c41` (product + tests + docs). Re-review of that commit should be short;
the tip's own code is otherwise sound and stays as-is.

## What was verified

**(a) Seccomp / threads — mechanism correct.** Both upstream premises check out
against gVisor `master`:

* `runsc/specutils/seccomp/seccomp.go` pins every `SCMP_ACT_ERRNO` to a
  package-level `errnoAction = seccomp.ReturnError.Code(uint16(unix.EPERM))`
  and never reads `ErrnoRet` — so the vendored `clone3` ENOSYS(38) rule arrives
  in the Sentry as EPERM, which is not a glibc/libuv fallback trigger. That is
  the `uv_thread_create` exit-134 signature exactly.
* An ALLOW is the *only* available repair: ENOSYS cannot be expressed through
  OCI seccomp under runsc, and clone3's flags live in a `struct clone_args` in
  user memory that seccomp cannot filter.

`_seccomp_opt` is correctly runsc-only and passes `unconfined` through
untouched; the derived profile puts the ALLOW ahead of the vendored ENOSYS rule,
which is what gVisor's in-order ruleset evaluation needs. The three RED tests
(`tests/e2e/test_sandbox_seccomp.py`) build their spec with
`runtime=get_settings().sandbox_runtime`, so the derivation does reach them in
the gVisor job. `test_unshare_user_namespace_denied` stays green:
`unshare -U` still falls through the masked-eq ALLOW into the #807 deny.

**(b) Snapshot residual — mechanism correct.** `runsc/boot/vfs.go` `mountTmp`
skips its internal tmpfs on `ENOTEMPTY` (and on an explicit `/tmp` spec mount,
which is why production's #2280 bind mount was never affected and only the
bind-mount-free e2e spec went red). The sentinel therefore does keep `/tmp` on
the rootfs. The image is built in-job (`docker build -t aios-sandbox:ci`), so
the Dockerfile change lands in the same CI run — no registry-rebuild dependency.
`Dockerfile.sandbox` is single-stage with no `VOLUME` and no later `/tmp` purge.

**(3) #2434 kept.** The tip is a single commit touching six files; nothing in
the SizeRw commit/flatten path, the `skipped_empty` identity, or the worker
`/etc/hosts` DNS is touched. No regression by construction.

**(4)/(5) Coverage and message.** Focused unit coverage exists and passes; the
commit body matches the diff. `uv run mypy src tests` clean, `ruff check` /
`ruff format --check` clean. Focused runs only — no full suite, no `-n`:
`tests/unit/sandbox` + `tests/unit/test_tar_filter.py` → **724 passed**
(683 + 41 after the fixes).

## Findings

### F1 — High. The clone3 ALLOW re-opens CLONE_NEWUSER under runsc; the tip says it does not.

`_runsc_seccomp_profile`'s docstring: *"`CLONE_NEWUSER` stays denied: the
authored unshare EPERM block and the arg-filtered clone ALLOW are untouched,
and those are what `test_unshare_user_namespace_denied` exercises."* The same
claim is in the commit message, `config.py`'s `sandbox_runtime` description and
the design doc. It does not follow, and it is false under runsc:

* gVisor implements clone3 — `linux64.go`:
  `435: syscalls.PartiallySupported("clone3", Clone3, "Options CLONE_NEWTIME,
  CLONE_SYSVSEM and SetTid are not supported.", nil)`. `Clone3` copies
  `clone_args` and calls the same `t.Clone(&cloneArgs)` as legacy clone, passing
  the flags through untouched apart from `CLONE_DETACHED`/exit-signal checks.
* `task_clone.go` gates `CLONE_NEWUSER` on nothing but `t.IsChrooted()` — no
  capability check (that is standard unprivileged-userns behaviour).
* The inserted ALLOW is unfiltered, necessarily so.

So a tenant in a runsc sandbox can obtain a user namespace via
`clone3(CLONE_NEWUSER)` while the guard test, which drives only `unshare`, stays
green. That is precisely the shape a checker exists to catch: the test that is
supposed to prove the property is insensitive to the change that breaks it.

Residual risk is bounded, and that is *why* the ALLOW is still the right call —
the #807 deny block is unconditional and first-match, so
`mount/umount/setns/unshare/keyctl/bpf` remain EPERM inside any namespace
obtained this way, and a fresh netns has no routable interface (wiring one in
needs CAP_NET_ADMIN in the **parent** userns). runc is untouched: it honours
`ErrnoRet`, keeps ENOSYS, and never sees the derived profile.

**Fixed in `1a588c41`:** the ALLOW is kept; `docker.py`, `config.py` and the
design doc now state the hole as an accepted risk with the bounding argument,
and `test_runsc_profile_is_the_authored_one_plus_exactly_the_clone3_allow` pins
the derivation to *authored + exactly one rule* (plus
`test_runsc_profile_keeps_the_unconditional_namespace_deny`) so a second hole
cannot be added silently.

### F2 — Medium. The `/tmp` sentinel does not survive flatten, so leg (b) regresses on the next cycle.

`EPHEMERAL_PREFIXES` (`src/aios/sandbox/_tar_filter.py`) drops everything under
`tmp/` from the flatten export — keeping the directory, dropping its contents,
including `/tmp/.aios-keep`. A flattened image therefore resumes with an
**empty** `/tmp`, `mountTmp` overlays tmpfs again, and the hidden-writes bug is
back. The design-doc sentence the tip added ("`/tmp` stays on the rootfs and
snapshot/resume keeps `/tmp/marker`") is true only until the first flatten. The
RED test passes because it pins `flatten_if_unique_bytes_over=None`, so CI would
not have caught the gap.

**Fixed in `1a588c41`:** `KEPT_PATHS = {"tmp/.aios-keep"}` carves the sentinel
out of `_is_ephemeral`, with `TestGvisorSentinel` asserting it survives while
its siblings are dropped and that it matches the Dockerfile that plants it. The
sentinel is zero bytes, so the `_ephemeral_bytes` flatten-gate estimate is
unaffected in any meaningful way.

### F3 — Low (note only). Derived-profile temp file is never cleaned up.

`_runsc_seccomp_profile` writes a `NamedTemporaryFile(delete=False)` and is
`functools.cache`d on the source path: one leaked file per profile path per
worker process (bounded, but never removed), and an edit to the authored profile
inside a live process is not picked up. A missing/unreadable profile now raises a
bare `OSError` out of `create()` rather than the `SandboxBackendError` its
siblings raise two lines below — it still fails hard, just with a less
recognisable error. Left as-is.

### F4 — Low (note only). The ALLOW is inserted at index 0, ahead of the authored #807 deny block.

Harmless today — the deny block deliberately excludes `clone`/`clone3` — but if
`clone3` were ever added there, the runsc copy would silently override it rather
than failing loudly. Inserting immediately after the authored deny block instead
of at the head would make that a loud CI failure. The new
authored-plus-exactly-one-rule test narrows the blast radius; the insertion point
is unchanged.

### F5 — Low (note only). A tenant can delete the sentinel.

`/tmp/.aios-keep` is root-owned `644` in a sandbox whose agent runs as root. A
session that removes it *and* empties `/tmp` gets the tmpfs overlay back on the
next resume. Self-inflicted and not worth a guard; noted for the record.

## Reproduction commands

```
uv run pytest tests/unit/sandbox tests/unit/test_tar_filter.py -q   # 724 passed
uv run mypy src tests                                               # clean
uv run ruff check src tests && uv run ruff format --check src tests # clean
```

No docker/gVisor e2e was run here (per brief). The e2e verdict still rests on
the next gVisor Validation run.
