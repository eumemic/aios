# Done

Both High findings from the eumemic-bot review of aios#2410 are fixed. The
existing gVisor product, coding-agent harness, operator-root / `bash -p` /
library-path design, and mint-after-agent token isolation remain in history.

## Commits

- `363b78b7` — publish reviews with `scripts/eumemic_bot_review.py` extracted
  from `github.event.pull_request.base.sha`
- `4c359ef0` — enter the read-only operator image with a static BusyBox
  `chroot` before starting the privileged runsc dynamic loader

The prior #2410 tip was `8bfa17f3`; all of its commits remain ancestors of the
new head. `origin/master` is also already an ancestor, so no rebase was needed.

## Security properties

### Trusted publisher

After the agent exits and before the App token is minted, the workflow extracts
the publisher from the trusted base commit into `runner.temp`. It disables Git
replacement-object processing while doing so. The `GH_TOKEN` step invokes only
that staged path; it no longer executes Python from the PR-head checkout. A
staging failure prevents token minting and is reported in the run summary.

### Tenant preload isolation

The sandbox image now installs `busybox-static`. Privileged runsc execution
starts that static binary directly, uses its `chroot` applet to enter the
read-only operator-image mount, and only then starts the operator dynamic loader
with its existing trusted library path and `bash -p`. Consequently neither a
tenant loader nor tenant `/etc/ld.so.preload` is consulted before the trusted
shell. The operator command-binding preamble continues to pin every external
egress command to operator-image paths.

## Verification

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/sandbox/test_runsc_operator_shadow.py
48 passed

$ uv run pytest tests/unit -q -n 4
6157 passed, 7 warnings in 115.30s

$ uv run ruff check src/aios/sandbox/backends/docker.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/sandbox/test_runsc_operator_shadow.py tests/unit/test_eumemic_bot_review.py tests/e2e/test_sandbox_image_contract.py
All checks passed!

$ uv run ruff format --check src/aios/sandbox/backends/docker.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/sandbox/test_runsc_operator_shadow.py tests/unit/test_eumemic_bot_review.py tests/e2e/test_sandbox_image_contract.py
5 files already formatted

$ git diff --check
(no output)
```

Docker is not installed in this workspace, so the live sandbox-image contract
test could not be run locally. CI will build the changed Dockerfile and verify
that `busybox-static` is installed. No push was performed.
