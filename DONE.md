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
- Docker is NOT available in this environment (`docker: command not found`), so
  the e2e image contract — the only test that can decide whether `--link`
  actually works — was NOT run. The fix remains unverified where it matters.
- Not pushed, not merged, no PR.
