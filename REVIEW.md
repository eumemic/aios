# Uncorrelated review — `botpost2410h` tip `c24d25bf` (aios#2410)

**Verdict: changes requested — 1 High, 2 Medium, 1 Low. All four are fixed on
this review branch (`botpost2410hrev`). The resolver bake is unchanged and
still unverifiable here: no Docker daemon, so the one test that can decide it
was NOT run.**

Scope this round: (1) rebase/mergeability, (2) the embedded-resolver bake,
(3) the arm64 fail-closed for the hardcoded x86_64 runsc operator paths.

---

## Item 1 — rebase / mergeability: **verified clean**

`origin/master` (`63337f26`, fetched this session) IS an ancestor of `HEAD`;
`git rev-list --left-right --count origin/master...HEAD` = `0 30`. Nothing to
rebase, no conflicts possible, the PR is fast-forwardable. The implementer's
claim holds.

Everything the round was told to keep is present in the tree, not just in the
log: `64e9b37f` / `6c6f750c` / `2e121847` (runsc egress in the target Sentry,
the operator-binary claim, chroot before the privileged loader), `bce1dfc2`
(the resolver bake) and the whole review-harness series, including the two-job
`agent` → `publish`-on-a-fresh-runner isolation in
`.github/workflows/eumemic-bot-review.yml`. No Track G.

## Item 2 — resolver bake: **kept, with the root-cause reasoning tightened**

`COPY --link docker/sandbox-resolv.conf /etc/resolv.conf` stays. Same-path is
still the only shape that can work — glibc reads `_PATH_RESCONF` and nothing
else, and the runsc operator root is a read-only image mount, so
`setup._RESOLV_PREAMBLE` has nowhere to write.

What I could add without a daemon is evidence that *rules out the two
alternative explanations* for the observed empty read, which the branch had
left open:

- **Not a stale image.** The lane that saw it (`code-validation.yml`, the
  `e2e` docker shard with `sandbox_changed=true`) runs
  `docker build -t aios-sandbox:ci -f docker/Dockerfile.sandbox .` and points
  `AIOS_DOCKER_IMAGE` at that local tag. It really was reading a fresh build,
  not a registry pull predating the bake.
- **Not a blind probe.** `docker cp` reads the container's *layers*, underneath
  the daemon's resolv.conf bind mount — [moby/moby#9998](https://github.com/moby/moby/issues/9998)
  is a bug report about exactly that ("returns the layer's empty stub, not the
  live file"). So an empty read is the layer's own content. This one mattered:
  if `docker cp` had been the artefact, no Dockerfile change could ever turn
  that e2e green and the fix would belong in the test.

Both are now written into `docker/Dockerfile.sandbox` and the e2e docstring.
The `--link` mergeop mechanism itself remains labelled HYPOTHESIS, because it
is one: `--link` is not a documented remedy for this path, and
`test_image_layer_carries_the_embedded_dns_resolver` is the only oracle.
**It was not run** — `docker` is absent from this environment.

## Item 3 — arm64 fail-closed: right call, wrong shape

Refusing is the correct arm of TASK.md's choice: per-arch loader paths would
be an untested second code path for a runtime nobody runs on arm64, and the
image really is published multi-arch (`build-sandbox.yml` builds
`linux/amd64,linux/arm64`), so an Apple Silicon pull genuinely carries a loader
at another triple. The *placement* is also right — both entry points that
consume `_RUNSC_OPERATOR_LOADER` are guarded, and `create` + `run_netns_sidecar`
are the only two (`registry.py` reaches the backend through exactly these,
browser containers included). But:

### F1 (High) — the branch is red on mypy

`uv run mypy src tests` on `c24d25bf`:

```
tests/unit/sandbox/test_docker_runtime_argv.py:88: error: Module "aios.sandbox.backends.docker"
  does not explicitly export attribute "platform"  [attr-defined]
tests/unit/sandbox/test_docker_runtime_argv.py:89: error: Module "aios.sandbox.backends.docker"
  does not explicitly export attribute "SandboxBackendError"  [attr-defined]
```

The new test reaches through `docker_backend.platform` / `.SandboxBackendError`,
both implicit re-exports, which this repo forbids. CI fails on the type shard.
This is the *same miss as last round in a different shard*: DONE.md's
verification was again a narrow `pytest` on one file, and CLAUDE.md's
pre-commit trio (`mypy` **and** `ruff` **and** `pytest tests/unit`) was not run.

**Fixed** — the test imports `SandboxBackendError` from `backends.base` and
patches the stdlib `platform` module directly.

### F2 (Medium) — the guard was a denylist, so most machines fell through

```python
platform.machine().lower() in {"aarch64", "arm64", "armv8l"}
```

names three spellings of one architecture. On `armv7l`, `ppc64le`, `riscv64`
or `i686` the check passes and execution proceeds into the x86_64 loader
path — i.e. the silent fallthrough the round was asked to close. "Fail hard,
no fallbacks" and correct-by-construction both point the other way: the
property is *"this machine is x86_64"*, not *"this machine is not one of three
arm strings"*.

**Fixed** — one `_RUNSC_SUPPORTED_MACHINES = frozenset({"x86_64", "amd64"})`
allow-list and a `_require_runsc_supported_machine(what)` helper, called from
both sites (the two copies had already drifted in wording). The message now
names the machine it actually saw and the loader path that does not exist for
it, instead of asserting "arm64" at a reader who is on ppc64le. The comment
records that `platform.machine()` is the *worker's* arch — a proxy for the
daemon's, which a remote `DOCKER_HOST` can break — and that both ways out of
that mismatch are safe: refuse a runsc sandbox that would have worked, or fall
through to the preamble's `operator tool root incomplete` (exit 90). Neither
silently blackholes egress.

### F3 (Medium) — the unit suite became arch-dependent, and the sidecar was untested

The commit message claims "Covers create and the netns sidecar path". The code
does; the tests did not — `test_create_refuses_runsc_on_arm64` was the only new
test, and nothing exercised `run_netns_sidecar`'s guard. Worse, the three
existing runsc argv tests assume an x86_64 host implicitly. Simulating an
aarch64 machine on `c24d25bf`:

```
FAILED tests/unit/sandbox/test_docker_runtime_argv.py::test_create_emits_configured_runtime
FAILED tests/unit/sandbox/test_docker_runtime_argv.py::test_netns_sidecar_runsc_execs_into_the_target_sentry
FAILED tests/unit/sandbox/test_docker_runtime_argv.py::test_netns_sidecar_runsc_runs_only_operator_image_binaries
```

`uv run pytest tests/unit` — the command CLAUDE.md requires before every
commit — is now red on any Apple Silicon machine, which is what this repo's
own docs show the maintainer developing on (`/Users/tom/.docker/...`).

**Fixed** — an autouse fixture pins `x86_64` for the module, and the arch tests
override it. Coverage is now: refusal on `aarch64, arm64, armv7l, ppc64le,
riscv64, i686` for **both** `create` and `run_netns_sidecar`, each asserting
the daemon was never touched; plus a test that arm64 still gets a working
*default-runtime* sandbox and sidecar, so a future widening of the guard cannot
take the platform out entirely.

### F4 (Low) — the e2e operator-chain contract contradicts the new decision

`tests/e2e/test_sandbox_image_contract.py` pins
`/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2` unconditionally, while
`TestArchitecture::test_image_architecture_matches_runner` in the same file
expects an arm64 image on an arm64 host. After this change those two tests fail
on arm64 *by design* — they assert a path the product now says is
x86_64-only — which reads as a finding instead of a restatement.

**Fixed** — the two operator-chain tests carry an `_x86_64_only` skip marker
naming `_RUNSC_SUPPORTED_MACHINES` as the reason. CI runs on amd64 runners, so
no coverage is lost.

---

## Verification run here

```
uv run mypy src tests                                   # Success: no issues in 1098 source files
uv run ruff check src tests && ruff format --check      # All checks passed / 1098 files formatted
uv run pytest tests/unit -q -n 4                        # 5315 passed pre-fix, 17 xdist worker
                                                        #   crashes; all 17 pass serially and are
                                                        #   unrelated to this branch (memory
                                                        #   pressure in this container)
uv run pytest -q tests/unit/sandbox tests/unit/test_networking.py \
              tests/unit/test_sandbox_registry.py tests/unit/test_detect_filter_sync.py
                                                        # green on x86_64 …
PYTHONPATH=<machine()->aarch64> uv run pytest -q …      # … and green with an aarch64 host: 758 passed
```

**Not run: the e2e image contract.** `docker` does not exist in this
environment. Whether `COPY --link` actually lands the resolver bytes in the
layer is still open, and only
`tests/e2e/test_sandbox_image_contract.py::test_image_layer_carries_the_embedded_dns_resolver`
on a daemon can close it. Not pushed, not merged, no PR opened.

## Residual risk for the shepherd

The one thing that can still fail after this branch merges is the resolver
e2e, for the reason the Dockerfile now spells out. If it is red again, the
next round should stop iterating on the Dockerfile: with a stale image and a
blind probe both ruled out, a second empty read means same-path bake is dead
under BuildKit, and the resolver must reach the chrooted operator on the read
path instead.
