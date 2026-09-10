# Uncorrelated review — `botpost2410e` tip `63cff5f5` (aios#2410)

**Verdict: changes requested — 2 High findings, both fixed on this review
branch (`botpost2410erev`, tip `b2f0ce4f`).**

Both properties TASK.md asks for are *structurally* right at `63cff5f5`: the
publisher is extracted from `github.event.pull_request.base.sha`, and the
privileged runsc exec enters the read-only operator image through a static
BusyBox `chroot` before any dynamic loader starts. Neither was carried far
enough. Finding 1 leaves a working path for PR-authored code to run inside the
process holding `GH_TOKEN` — the exact property the change exists to establish.
Finding 2 is not a security regression but a functional one: the chroot silently
took DNS with it, and nothing in unit tests or PR CI would have caught it.

## Scope

Verified against TASK.md items 1–4. Not reviewed: Track G (excluded). Nothing
was pushed, merged, or opened as a PR. Docker is unavailable in this workspace,
so image-contract tests were reasoned about and written but not executed.

---

## F1 (High) — the staged publisher could still import PR-authored code

`.github/workflows/eumemic-bot-review.yml` at `63cff5f5` staged the publisher
to a fixed path and ran it plainly:

```yaml
git show "$BASE_SHA:scripts/eumemic_bot_review.py" > "$TRUSTED_PUBLISHER_PATH"   # ${{ runner.temp }}/eumemic_bot_review.py
...
run: python3 "$TRUSTED_PUBLISHER_PATH" publish
```

TASK.md's acceptable fix is explicit that the property covers "`scripts/eumemic_bot_review.py`
**and any module it imports** for publish". Three ways the head still reaches
that process:

1. **`sys.path[0]`.** CPython puts the *script's own directory* first on
   `sys.path`. That directory is `RUNNER_TEMP`, which the agent step — earlier,
   unsandboxed, `danger-full-access`, PR-authored prompt — can write. A
   `json.py` (or `subprocess.py`, `urllib/`) left there is imported by the
   publisher, inside the process holding `GH_TOKEN`. Confirmed by running
   CPython, not inferred.
2. **`PYTHONPATH`.** Same outcome via a line appended to `$GITHUB_ENV`, which
   the agent step can also write; the publish step inherits it.
3. **The fixed destination.** `> "$TRUSTED_PUBLISHER_PATH"` opens whatever is
   at that path. The agent knows the name in advance and can pre-create a
   symlink or FIFO there.

Staging from base bought nothing against any of these: the token-holding
process still ends up executing bytes the PR author chose.

**Fixed** in `e0ba994a`:

- stage into `dir="$(mktemp -d "${RUNNER_TEMP}/trusted-publisher.XXXXXXXX")"`
  and export `path=` as a step output. `mktemp -d` fails rather than reuse an
  existing path, so a staged file is one that step wrote;
- run `python3 -I "$TRUSTED_PUBLISHER_PATH" publish`. Isolated mode drops both
  the script directory and `PYTHONPATH` from `sys.path`;
- a new unit test pins the whole shape (`mktemp -d`, the `$GITHUB_OUTPUT`
  hand-off, the exact `-I` argv), so this cannot regress to the plain form.

The prior round's REVIEW.md argued against a `RUNNER_TEMP` copy because that
directory "is writable by the same agent, so adding one would buy false
assurance". That reasoning is why the fixed path shipped; it is right about the
directory and wrong about the conclusion — the answer is to stop trusting the
directory (`mktemp -d` + `-I`), not to stop staging.

`docs/eumemic-bot-review.md` claimed the residual surface was closed. It is
not, and the doc now says so plainly: the agent shares a runner with the
publish step, so `$GITHUB_PATH` can still shadow `python3`/`git`, runner
binaries can be overwritten outright, and only a separate identity or container
closes that.

## F2 (High) — the chroot broke DNS for every runsc egress provision

`4c359ef0` is correct about `/etc/ld.so.preload`. But `chroot` changes which
`/etc/resolv.conf` glibc reads, and that file is load-bearing on this path.

`setup._RESOLV_PREAMBLE` is prepended to both the Limited lockdown apply script
and the Unrestricted DNAT-only script:

```python
f"printf 'nameserver {_EMBEDDED_DNS_ADDRESS}\n' > /etc/resolv.conf 2>/dev/null || true\n"
```

Post-chroot the target is the operator image mounted read-only (`--mount
type=image,...` in `create()`), so the write fails and `|| true` discards the
failure. `getent ahostsv4` then reads the *image's* `/etc/resolv.conf` — which
the image never had: Docker masks that path at runtime, and a `RUN` redirect
writes to the build-time mount, never to the layer. glibc's no-resolv.conf
fallback is 127.0.0.1, and nothing serves DNS there inside the sandbox netns.

Consequences, in the code's own terms:

- Limited: `resolve_ipv4` returns nothing for every allowed host → an empty
  allow-list under `-P OUTPUT DROP`. Fails *closed*, so not a bypass, but every
  Limited sandbox on runsc is an egress blackhole.
- Unrestricted: `PROXY_IP=$(resolve_ipv4 <alias> | head -n1)` is empty → the
  secret-egress DNAT is skipped.

Why nothing caught it: `test_runsc_operator_shadow.py` runs the real scripts but
rewrites `/etc/resolv.conf` to `tmp_path` and does not model the chroot, and
`gvisor-validation.yml` — the only place the runsc egress path executes — is
`workflow_dispatch` plus a weekly `cron: 37 9 * * 2`. This would have shipped
green and surfaced as "gVisor sandboxes have no network".

**Fixed** in `b2f0ce4f`: `docker/sandbox-resolv.conf` (`nameserver
127.0.0.11`), COPYed to `/etc/resolv.conf` after the last `RUN` step. COPY
because RUN cannot reach the layer; last because the `apt-get` steps need the
daemon's resolver, not the netns-local one.

This is the better property, not just a repair. Before, the sidecar *overwrote*
a tenant-controlled file and hoped to win the race; now the resolver the
allow-list is derived from is trusted image content that the tenant cannot
reach at all — which is the same argument the chroot itself rests on.

### F2a (Medium, folded into the same commit) — dangling operator `PATH`

`_RUNSC_OPERATOR_EXEC_ENV` still set

```python
("PATH", f"{_RUNSC_OPERATOR_ROOT}/usr/sbin:{_RUNSC_OPERATOR_ROOT}/usr/bin"),
```

which resolves to nothing once the exec is rooted *inside* that mount. The
comment beside it promises that a command `_RUNSC_OPERATOR_COMMANDS` forgets
"degrades to *operator binary under the tenant's loader*, never to *tenant
binary*" — with a dangling `PATH` it degrades to "command not found" instead,
turning a silent-but-safe miss into a provision failure. Now `/usr/sbin:/usr/bin`,
which post-chroot *are* the operator image's. The argv unit test was pinning
the broken value and now pins the correct one with the reasoning inline.

### Tests added for F2

Both regressions were outside every gate on PR CI, so:

- `tests/unit/sandbox/test_sandbox_resolv_conf.py` — the baked nameserver must
  equal `setup._EMBEDDED_DNS_ADDRESS`; the Dockerfile must reach it by `COPY`
  (not `RUN`) and after every `RUN`.
- `tests/e2e/test_sandbox_image_contract.py` — read `/etc/resolv.conf` out of
  the *layer* via `docker cp` from a created-but-unstarted container (a running
  one has Docker's own file bind-mounted over it, which is exactly why the gap
  was invisible); assert BusyBox ships the `chroot` applet (it is a
  compile-time applet list, and a build without it would leave tenant
  `ld.so.preload` in force under `--privileged`); pin each link of the exec
  chain at its absolute path; and execute the real chroot → loader →
  `bash -p` chain once, with `/` standing in for the operator mount so it runs
  without a runsc daemon.
- `docker/sandbox-resolv.conf` added to the `sandbox_changed` CI filter and to
  `build-sandbox.yml`'s push trigger — otherwise a change to it would pull a
  stale `:smoked` image and leave the published image stale. Both are pinned by
  `tests/unit/test_detect_filter_sync.py`, whose `bin/tool` test is now
  parametrized over every file the Dockerfile COPYs.

---

## Checklist verdicts

| # | Requirement | Verdict |
|---|---|---|
| 1 | Publisher from trusted base when `GH_TOKEN` present; staging failure blocks the mint and is reported; agent phase may use PR head | **Was incomplete** (imports + fixed path) → now holds. Mint is gated on `steps.publisher.outcome == 'success'`; the summary step lists `publisher` among the failures it reports; the agent step still runs `python3 scripts/eumemic_bot_review.py agent` from the head checkout, as allowed. |
| 2 | Privileged runsc exec ignores tenant loader config; operator-root / `bash -p` / library-path design preserved; unit test pins argv | **Holds.** BusyBox is static (`busybox-static`, `ii`-pinned in the contract test) so the tenant loader and `/etc/ld.so.preload` never run; `chroot` is Full Support in gVisor and the applet is present in Debian bookworm's `busybox-static`. `test_docker_runtime_argv.py` pins the seven-token chain. The chroot's *side effects* were the defect (F2/F2a), not the mechanism. |
| 3 | Branch keeps prior #2410 / gVisor / harness / mint-after-agent history; `origin/master` is an ancestor | **Holds.** `origin/master`, `8bfa17f3`, `63cff5f5`, `d1da0cd9`, `27faf31a`, `e7f73af7`, `789174bc`, `ffef992c` are all ancestors of `HEAD`; 20 commits ahead of `origin/master`; no rebase needed. |
| 4 | DONE.md claims match reality | **Was accurate for what it claimed** — both shas exist, both ancestry claims check out, and `48 passed` reproduced exactly at `63cff5f5`. Rewritten here for the new work (49 / 6161, plus the residual it previously did not mention). |

## What I did not verify

- No Docker daemon in this workspace: the four new image-contract tests and the
  existing ones skip (`51 skipped`). They are the only check that
  `/etc/resolv.conf` actually lands in the layer, and CI is where that runs.
- The runsc path itself needs `gvisor-validation.yml`, which is cron-only. The
  DNS defect above is reasoned from the code and from Docker's documented
  resolv.conf behaviour, not observed on a live Sentry.
- **Recommendation for a follow-up (not done here, out of scope):** the reason
  both F2 and F2a could ship is that the only executable check of the runsc
  operator path runs weekly. A PR that touches `sandbox/backends/docker.py` or
  the sandbox Dockerfile should trigger `gvisor-validation.yml`, not wait for
  Tuesday.

## Local results

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py tests/unit/sandbox/test_docker_runtime_argv.py tests/unit/sandbox/test_runsc_operator_shadow.py
49 passed

$ uv run pytest tests/unit -q -n 4
6161 passed, 9 warnings in 51.27s

$ uv run ruff check src tests && uv run ruff format --check src tests
All checks passed! / 1091 files already formatted

$ uv run mypy src tests
Success: no issues found in 1091 source files
```
