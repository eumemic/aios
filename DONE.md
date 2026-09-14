# Done

- Implementer commits reviewed: `119ed983` (fix) and `0b0effc9` (DONE.md), on
  branch `botpost2410g`. This review branch is `botpost2410grev`.
- `origin/master` (`abe20173`) is an ancestor of HEAD; 28 commits ahead. The
  prior gVisor work (`17ef3de9` chroot-before-loader, `538ab985` resolver bake)
  and the whole review-harness series are retained.

## Root cause — what is observed and what is hypothesised

OBSERVED: the e2e image contract
(`tests/e2e/test_sandbox_image_contract.py::test_image_layer_carries_the_embedded_dns_resolver`)
built the image from this Dockerfile and read `/etc/resolv.conf` back out of
the committed layer with `docker create` + `docker cp`. It came back with no
`nameserver` line, although `docker/sandbox-resolv.conf` is correct and the
plain `COPY … /etc/resolv.conf` already sat after every `RUN`. The CI
`detect` filter puts `docker/sandbox-resolv.conf` and `docker/Dockerfile.sandbox`
in `sandbox_changed`, and on `pull_request` the base is `PR_BASE_SHA`, so that
run really did build a fresh image — it is not a stale `:smoked` pull.

HYPOTHESISED: that BuildKit's `/etc/resolv.conf` special-casing
(moby/buildkit#1267) leaves its placeholder in the committed snapshot and that
`COPY --link` dodges it by building the file over `scratch` and merging it on
top rather than writing through the parent snapshot. This is a bet, not a
verified mechanism — `--link` is nowhere documented as a remedy for this path,
and BuildKit's documented masking is a RUN-time bind mount, which this COPY is
already past. The implementer's DONE.md stated it as settled fact; it is not.

## Bake path

Kept `COPY --link docker/sandbox-resolv.conf /etc/resolv.conf`. Same-path bake
is the only shape that can work: glibc reads `_PATH_RESCONF` = `/etc/resolv.conf`
and nothing else (no env override), the runsc operator root is mounted READ-ONLY
so `setup._RESOLV_PREAMBLE` cannot write it, and a gVisor-internal bind mount is
not dependable. TASK.md's advisory "non-special path + symlink/bind" direction
cannot deliver the runtime property on its own.

## Fixes made on this review branch

- `tests/unit/test_detect_filter_sync.py`: `test_build_sandbox_triggers_on_every_copied_file`
  was RED on `0b0effc9`. Its `^COPY <path>\s` pattern is flag-sensitive, so
  adding `--link` read as "the file is no longer COPYed". Widened to
  `^COPY (?:--\S+ )*<path>\s`. The implementer ran only the narrow resolver
  test file and could not see it; CI would have gone red on the unit shard
  before ever reaching the e2e.
- `docker/Dockerfile.sandbox`: rewrote the resolver comment to separate the
  observed failure from the `--link` hypothesis, name the e2e as the only
  oracle, record that a non-special path alone does not rescue it, and note
  that `--link` raises the floor to BuildKit >= 0.10 / Docker >= 23 with no
  `# syntax=` pin (an older daemon fails loudly instead of shipping an empty
  resolver).
- `tests/unit/sandbox/test_sandbox_resolv_conf.py`: added a SCOPE paragraph
  saying these are source-level pins that cannot distinguish a surviving layer
  from a stripped one, and re-keyed the matcher onto the DESTINATION so a plain
  `COPY … /etc/resolv.conf` is found and rejected by name rather than silently
  missed by a flag-sensitive pattern.
- `tests/e2e/test_sandbox_image_contract.py`: gave the nameserver assertion an
  actionable message and a comment saying what a repeat failure means (the
  `--link` layer was stripped too; the fix then has to move to the operator
  read path).

## Verification

- `uv run pytest -q tests/unit/sandbox/test_sandbox_resolv_conf.py tests/unit/test_detect_filter_sync.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/test_gvisor_validation_workflow.py` — 24 passed.
- `uv run ruff check` / `ruff format --check` / `uv run mypy` on the four
  touched files — clean.

  (Corrected on the review branch: this verification was too narrow. Repo-wide
  `uv run mypy src tests` — the command CLAUDE.md requires — was RED at
  `c24d25bf`, on `docker_backend.platform` / `docker_backend.SandboxBackendError`
  implicit re-exports in the new test. Repo-wide `uv run pytest tests/unit` was
  also red on any non-x86_64 host, because three pre-existing runsc argv tests
  assume the machine passes the new guard.)
- Docker is NOT available in this environment (`docker: command not found`), so
  the e2e image contract — the only test that can decide whether `--link`
  actually works — was NOT run. The fix remains unverified where it matters.
- Not pushed, not merged, no PR.

## Follow-up fixes after rebase

- The resolver bake remains a same-path `COPY --link` because glibc and the
  read-only runsc chroot both require `/etc/resolv.conf`; a non-special path
  cannot be consumed by `getent`. The Docker e2e contract is the authoritative
  check, and Docker is unavailable in this environment.
- Runsc is fail-closed on every machine that is not x86_64/amd64. The operator
  image and explicit ELF loader/library paths are x86_64-only; both sandbox
  creation and the sidecar path refuse before invoking Docker.

  (Corrected on the review branch: as shipped at `c24d25bf` this was a DENY-list
  of three arm spellings — `aarch64`/`arm64`/`armv8l` — so `armv7l`, `ppc64le`,
  `riscv64` and `i686` fell through to the x86_64 loader path. It is now an
  ALLOW-list, `_RUNSC_SUPPORTED_MACHINES = {"x86_64", "amd64"}`, applied through
  a single `_require_runsc_supported_machine()` helper instead of two inline
  copies.)

## Review-branch round (`botpost2410hrev`, uncorrelated review of `c24d25bf`)

Verdict and full findings: `REVIEW.md`. Items 1 and 2 verified clean; item 3 was
the right decision in the wrong shape. Fixed here:

- **mypy red** — the new test imports `SandboxBackendError` from
  `aios.sandbox.backends.base` and patches the stdlib `platform` module
  directly, instead of reaching through the `docker` module's implicit
  re-exports. `uv run mypy src tests` now: no issues in 1098 source files.
- **Deny-list → allow-list** — see the correction above.
- **Arch-dependent unit suite** — an autouse fixture pins `platform.machine()`
  to `x86_64` for `test_docker_runtime_argv.py`, so the module's runsc argv
  tests state the host they always assumed. Confirmed green under a simulated
  aarch64 host (758 passed).
- **Sidecar coverage** — the commit message claimed create + sidecar; only
  `create` had a test. Both paths now refuse `aarch64, arm64, armv7l, ppc64le,
  riscv64, i686` and assert the daemon was never called, plus a regression test
  that arm64 still gets a working default-runtime sandbox and sidecar.
- **e2e consistency** — `test_operator_chain_binary_at_absolute_path` and
  `test_operator_chain_executes_end_to_end` pin x86_64 loader paths against a
  multi-arch image, so they are now skipped off amd64 for the same reason the
  backend refuses. CI runners are amd64; no coverage lost.
- **Resolver root cause hardened, code unchanged** — the two competing
  explanations for the observed empty `/etc/resolv.conf` are now ruled out in
  `docker/Dockerfile.sandbox` and the e2e docstring: the lane that saw it builds
  the image locally (`docker build -t aios-sandbox:ci` in `code-validation.yml`,
  with `AIOS_DOCKER_IMAGE` pointed at that tag), so it was not a stale pull; and
  `docker cp` reads the container's layers beneath the daemon's resolv.conf bind
  mount (moby/moby#9998), so the empty read is the layer's own content, not a
  blind probe. The `--link` mergeop mechanism itself stays labelled HYPOTHESIS.

Still NOT run: the e2e image contract. `docker` does not exist in this
environment. Not pushed, not merged, no PR opened.
