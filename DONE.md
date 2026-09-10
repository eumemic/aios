# Done

Both High findings from the eumemic-bot review of aios#2410 are fixed, and the
leftovers an uncorrelated review found in the first pass at them are fixed too.
The existing gVisor product, coding-agent harness, operator-root / `bash -p` /
library-path design, and mint-after-agent token isolation remain in history.

## Commits

- `363b78b7` — publish reviews with `scripts/eumemic_bot_review.py` extracted
  from `github.event.pull_request.base.sha`
- `4c359ef0` — enter the read-only operator image with a static BusyBox
  `chroot` before starting the privileged runsc dynamic loader
- `e0ba994a` — stage that publisher into a `mktemp -d` directory and run it
  under `python3 -I`, so the PR head cannot supply its *imports* either
- `b2f0ce4f` — bake the embedded resolver into the operator image, because the
  chroot moved `/etc/resolv.conf` onto a read-only mount and took DNS with it

The prior #2410 tip was `8bfa17f3`; all of its commits remain ancestors of the
new head. `origin/master` is also already an ancestor, so no rebase was needed.

## Security properties

### Trusted publisher

After the agent exits and before the App token is minted, the workflow extracts
the publisher from the trusted base commit, with Git replacement-object
processing disabled.

It stages into a directory `mktemp -d` creates under `RUNNER_TEMP`, not a fixed
path: the agent step runs earlier, unsandboxed, with write access there, so a
fixed destination could be pre-created as a symlink or FIFO for the redirect to
open, and `mktemp` fails rather than reuse an existing path. The path is handed
to the publish step as a step output.

The `GH_TOKEN` step runs that staged path under `python3 -I`. Isolated mode is
load-bearing rather than tidiness: without it CPython puts the script's own
directory first on `sys.path` and honours `PYTHONPATH`, so a `json.py` dropped
beside the staged file — or a `PYTHONPATH` line appended to `$GITHUB_ENV` —
would execute PR-authored code inside the process holding the token. No Python
from the PR-head checkout is executed with `GH_TOKEN` present. A staging
failure gates the mint (`steps.publisher.outcome == 'success'`) and is reported
in the run summary.

Residual, documented in `docs/eumemic-bot-review.md` rather than claimed away:
the agent shares a runner with the publish step, so it can still shadow
`python3`/`git` via `$GITHUB_PATH`, overwrite a runner binary outright, or
scrape `EUMEMIC_BOT_PRIVATE_KEY` if a later step exposes it. Only a separate
identity or container closes that.

### Tenant preload isolation

The sandbox image installs `busybox-static`. Privileged runsc execution starts
that static binary directly, uses its `chroot` applet to enter the read-only
operator-image mount, and only then starts the operator dynamic loader with its
existing trusted library path and `bash -p`. Neither a tenant loader nor tenant
`/etc/ld.so.preload` is consulted before the trusted shell. The operator
command-binding preamble continues to pin every external egress command to
operator-image paths.

Two things the chroot moved, fixed in `b2f0ce4f`:

- **`/etc/resolv.conf`.** Post-chroot it is the operator image's, on a
  read-only mount, so `setup._RESOLV_PREAMBLE`'s write fails and its `|| true`
  hides that. The image shipped no resolver of its own (Docker masks that path
  at runtime, and a `RUN` redirect never reaches the layer), so glibc fell back
  to 127.0.0.1, every host resolved to zero addresses, and Limited egress built
  an empty allow-list while Unrestricted skipped its secret-egress DNAT for
  want of a `$PROXY_IP`. `docker/sandbox-resolv.conf` is now COPYed to
  `/etc/resolv.conf` after the last `RUN` step. This is also the stronger
  property: the allow-list now derives from trusted image content instead of a
  tenant file the sidecar races to overwrite first.
- **`PATH`.** The exec's `PATH` still named `{operator_root}/usr/...`, which
  resolves to nothing once rooted inside that mount. It is now
  `/usr/sbin:/usr/bin` — post-chroot those *are* the operator image's — so a
  command `_RUNSC_OPERATOR_COMMANDS` forgets degrades to a trusted operator
  binary, as the comment always claimed, rather than to "command not found".

Neither regression was reachable from the existing tests: the shadow test
redirects `/etc/resolv.conf` to a tmp path and does not model the chroot, and
the runsc e2e suite runs on a weekly cron, not on PR CI. Added:
`tests/unit/sandbox/test_sandbox_resolv_conf.py` (baked nameserver ==
`setup._EMBEDDED_DNS_ADDRESS`; COPY, not RUN; after every RUN) and
image-contract tests that read the layer through `docker cp` from an unstarted
container, assert BusyBox ships the `chroot` applet, pin each link of the exec
chain at its absolute path, and run the whole chroot → loader → `bash -p` chain
once. `docker/sandbox-resolv.conf` was added to the `sandbox_changed` CI filter
and to `build-sandbox.yml`'s push trigger, both pinned by
`tests/unit/test_detect_filter_sync.py`.

## Verification

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/sandbox/test_runsc_operator_shadow.py
49 passed in 3.37s

$ uv run pytest tests/unit -q -n 4
6161 passed, 9 warnings in 51.27s

$ uv run ruff check src tests
All checks passed!

$ uv run ruff format --check src tests
1091 files already formatted

$ uv run mypy src tests
Success: no issues found in 1091 source files

$ git diff --check
(no output)
```

Docker is not available in this workspace, so the sandbox image-contract tests
(including the four new ones) skip locally:

```text
$ uv run pytest -q tests/e2e/test_sandbox_image_contract.py
51 skipped in 0.19s
```

CI builds the changed Dockerfile and runs them. No push was performed.
