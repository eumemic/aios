# Review — runsc egress apply in the target Sentry (`6d7ae8e0`)

Uncorrelated review of the gVisor Validation green fix on branch `gvisorgrnrev`
(same tip as the implementer branch). Scope per the review brief: the
runsc-specific egress apply, the runc path staying unchanged, Limited staying
fail-closed, the workflow's `perf` exclusion, test coverage of the new
argv/exec path, and DONE.md's claims versus what the tests actually prove.
Track G / Coolify and browser F4 were out of scope and not touched.

## Verdict

The **mechanism is right**. `docker exec` into the target container is the
correct answer to #2310 / gvisor#170: two runsc containers sharing a Linux netns
get separate Sentries, so a `--network container:` sidecar programs its own
netstack and the target's tables stay empty. Rules have to be written from
inside the target's Sentry, and `docker exec --privileged` is the only way to
get `NET_ADMIN` there (`docker exec` has no `--cap-add`).

The **isolation claim as committed was not true**, and I fixed it. DONE.md
states:

> Tenant changes persisted in the sandbox root therefore cannot replace the
> binaries trusted to install or verify the rules.

As committed, four separate paths let the tenant do exactly that. All four are
fixed on this branch; the claim now holds, and is pinned by an executing test
rather than by prose.

The workflow change is correctly scoped. DONE.md does **not** overclaim
greenness — it says plainly that local unit tests "does not prove the scheduled
job green on a real runsc daemon", which is the honest position.

One **pre-existing red test** shipped on `6d7ae8e0`; fixed (F5).

---

## Findings

### F1 — `awk`, `sort`, `head` ran from the tenant filesystem (serious)

`setup._RESOLVE_IPV4_FN` is
`getent ahostsv4 "$1" | awk '{print $1}' | sort -u`, and `_nat_dnat_lines`
runs `PROXY_IP=$(resolve_ipv4 ... | head -n1)`. The committed preamble shadowed
only `iptables*`, `ip6tables*`, `getent` and `grep`, so `awk`, `sort` and `head`
resolved through the tenant's `PATH` — inside the tenant's mount namespace,
against a durable, tenant-writable root.

This is not a theoretical gap: `awk` chooses the addresses fed to
`iptables -d`. A tenant `awk` emitting `0.0.0.0/0` turns the Limited
allow-list into `-A OUTPUT -d 0.0.0.0/0 -p tcp --dport 443 -j ACCEPT` — a
blanket egress bypass that still exits 0 and still passes the `-P OUTPUT DROP`
read-back verify. The lockdown reports green while open.

**Fixed:** the shadow map covers every external command the generated scripts
run, and `PATH` is repointed into the operator root so a *future* unshadowed
command degrades to "not found" rather than "tenant binary".

### F2 — tenant-authored container env was inherited by the exec (serious)

`EnvironmentConfig.env` is a free-form `dict[str, str]` with no key validation,
injected at `docker run --env`, and `docker exec` inherits the container's
environment. Two families of injection followed:

* `LD_PRELOAD` / `LD_AUDIT` — passing `--library-path` to `ld.so` changes where
  it *searches*; it does not stop it honouring these. Either one runs tenant
  code inside the "operator-trusted" process.
* `BASH_ENV`, `ENV`, `SHELLOPTS`, `BASHOPTS`, `BASH_FUNC_*` — honoured by
  `bash -c`. `BASH_FUNC_iptables%%` would replace the shadow function itself.

This hole is **new to the runsc path**: the runc sidecar is a fresh container
built from the operator image with operator environment, so it never had it.

**Fixed:** the exec passes `--env` scrubs for the loader vars and pins `PATH`,
and runs `bash -p` (privileged mode), which ignores `BASH_ENV`/`ENV`/
`SHELLOPTS`/`BASHOPTS`/`CDPATH`/`GLOBIGNORE` and refuses `BASH_FUNC_*` import.
`POSIXLY_CORRECT` is cleared separately: bash reads it at startup regardless of
`-p`, and posix mode rejects the hyphenated `iptables-legacy` function name — a
tenant-triggerable provisioning failure rather than a bypass, but still
tenant-controlled behaviour.

### F3 — two path families escaped the read-only mount (serious)

Everything under `$OP` is only trustworthy if resolving it never leaves the
mount. Two committed paths did not satisfy that:

* `$OP/usr/sbin/iptables`, `$OP/usr/sbin/ip6tables` (and `/usr/bin/awk`, once
  shadowed) are **update-alternatives** symlinks to `/etc/alternatives/<name>`.
  That target is **absolute**, so it resolves in the *tenant* root. Latent for
  `iptables` today only because `_IPTABLES_BACKEND_SELECT` always prefers the
  `-legacy` name; it would have become live the moment `awk` was shadowed
  naively.
* `$OP/lib64/ld-linux-x86-64.so.2` and `$OP/bin/bash` depend on the Debian
  usr-merge symlinks (`/bin`, `/lib`, `/lib64`, `/sbin`) having *relative*
  targets. True in bookworm, but that is a base-image property, not a
  guarantee — a bump that made one absolute would silently redirect the loader
  and the shell into the tenant root, with no test failing.

**Fixed:** every shadowed path is now the real file under `/usr`
(`/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2`, `/usr/bin/bash`,
`iptables-legacy`, `mawk`), and two tests assert no `/etc/alternatives` target
and no usr-merge-prefixed path can be reintroduced.

### F4 — a missing operator root failed *open-ish*, not closed

A container created before this path existed, or a daemon that ignores
`--mount type=image`, has no operator root. The committed preamble would then
fail with an opaque `ld.so` error whose exit status depends on which command
happened to run first.

**Fixed:** the preamble presence-checks the loader and every shadowed binary up
front and exits `90` with a named message. `apply_network_lockdown` already
fails closed on nonzero, so Limited provisioning refuses — the required
behaviour. Pinned by
`test_preamble_fails_closed_when_operator_root_is_missing`.

### F5 — `6d7ae8e0` shipped a failing unit test

`tests/unit/test_gvisor_validation_workflow.py::test_gvisor_workflow_mirrors_docker_e2e_setup_and_runs_runsc_shard`
pins the workflow's pytest invocation as an exact string. The commit changed
`-m docker` to `-m 'docker and not perf'` and did not update the pin, so the
test is red on `6d7ae8e0` (confirmed by checking out HEAD clean). DONE.md's
"Local proof" ran only two files and so did not see it.

**Fixed:** pin updated to the new selector, plus an assertion that
`continue-on-error` never appears — excluding the advisory mark must not become
a licence to make docker failures non-fatal.

### F6 — the `--mount type=image` requirement was undocumented and unprobed

The runsc path now needs Docker Engine 28+ with the containerd image store
(`features.containerd-snapshotter`). The workflow enables it; nothing else in
the tree records that runsc has acquired a hard daemon dependency, and nothing
verifies the daemon actually honoured the flag. On a runner that ignores it,
every Limited provisioning fails deep in the E2E suite with an error that looks
unrelated to the daemon config.

**Fixed:** documented on `AIOS_SANDBOX_RUNTIME` in `config.py`, and the
workflow now probes `docker run --mount type=image` immediately after the
daemon restart, failing with a `::error::` that names the actual cause.

### F7 — control flow around the runtime branch

The committed code branched on `runtime == "runsc"`, then re-tested the same
condition twice more after the branch (`if runtime and runtime != "runsc"`,
`if runtime != "runsc"`) to finish assembling the non-runsc argv. Correct, but
it left the runc argv split across three places.

**Fixed:** one `if/else`; each branch builds its own complete argv.

---

## Behaviour changes worth an operator's attention (documented, not "fixed")

* **`/etc/resolv.conf`.** Under runsc the script runs in the target's mount
  namespace, so `setup._RESOLV_PREAMBLE` now rewrites the *sandbox's own*
  `resolv.conf` instead of a throwaway sidecar's. It loses Docker's
  `options ndots:0 edns0 trust-ad` line. This is net security-**positive** —
  the allow-list is built from what `getent` returns, so a tenant-poisoned
  `resolv.conf` would otherwise choose which addresses get an `ACCEPT` — and
  lossless in practice on a user-defined network, where Docker writes the same
  `127.0.0.11`. Now stated in the `run_netns_sidecar` docstring.
* **`--privileged` is wider than `--cap-add NET_ADMIN`.** `docker exec` has no
  `--cap-add`, so the exec gets the full capability set where the runc sidecar
  got one capability. It is one ephemeral operator process, and the sandbox's
  own processes still hold no `NET_ADMIN`, so the tenant-facing property is
  unchanged — but it is a real widening of the operator surface and is now
  called out in the docstring rather than left implicit.
* **`image` is unused on the runsc path.** The operator root is whatever
  `create()` mounted, which is `settings.docker_image` — deliberately the
  operator image, not `spec.image`, so a browser or custom sandbox image still
  gets operator-trusted tools. Every in-tree caller passes the same value, so
  there is no live divergence; documented rather than changed.

## Confirmed correct, no change made

* **runc path untouched.** `create()` adds the mount only under
  `spec.runtime == "runsc"`; `run_netns_sidecar`'s non-runsc branch is the same
  `docker run --rm --network container:<id> --cap-add NET_ADMIN [--runtime rt]
  <image> bash -c` it always was. Now pinned by two tests (one for a non-runsc
  named runtime, one for `runtime=None`) — the committed change had converted
  the only test covering that shape into a runsc test, leaving it uncovered.
* **Workflow narrowing is by mark only.** `-m 'docker and not perf'` matches the
  precedent in `code-validation.yml:627` verbatim. `perf` is a single class
  (`TestAdvisoryScalingBackstop`, 3 tests, one file), documented NON-GATING
  under #1661. Every `docker`-marked test still runs and still fails the job;
  the job is not `continue-on-error`, and there is no standing skip. The
  workflow now says *why*, as `code-validation.yml` does.
* **Limited stays fail-closed.** No error suppression was added; the new
  failure mode (F4) is an explicit nonzero exit.
* **DONE.md does not overclaim greenness.** It explicitly disclaims proving the
  scheduled job green and names `workflow_dispatch` as the real proof. Its
  *isolation* claim was the overclaim, and that is now made true rather than
  softened.

## What I could not verify here

No Docker daemon in this environment (`docker: command not found`), so
everything below is unverified by execution and remains for the
`workflow_dispatch` run:

1. that `docker exec --privileged` actually confers `CAP_NET_ADMIN` inside a
   runsc Sentry;
2. that gVisor's netstack accepts the full generated ruleset — specifically
   `-m conntrack --ctstate ESTABLISHED,RELATED` and nat-table `DNAT`;
3. that the runner's Engine honours `--mount type=image` (the new probe answers
   this in the first 30 seconds of the job rather than 20 minutes in).

The task's "do not claim the scheduled job is green without a real runsc proof
path" still stands: this review does not make that claim.

## Changes

| File | Change |
|---|---|
| `src/aios/sandbox/backends/docker.py` | Operator-root constants; `_runsc_operator_preamble()` with presence check + full command shadow set; env scrubs + `bash -p`; real `/usr` paths; single `if/else`; docstring |
| `src/aios/config.py` | `AIOS_SANDBOX_RUNTIME=runsc` now documents the Engine 28 + containerd-image-store requirement |
| `.github/workflows/gvisor-validation.yml` | Why `perf` is excluded; `--mount type=image` capability probe |
| `tests/unit/sandbox/test_runsc_operator_shadow.py` | **New.** Executes the real generated scripts against a synthetic operator root and fails on any command bash resolves outside it; fail-closed and path-shape assertions |
| `tests/unit/sandbox/test_docker_runtime_argv.py` | runsc exec shape; operator-binary/env-scrub assertions; restored non-runsc sidecar coverage; `create` omits the mount by default |
| `tests/unit/test_gvisor_validation_workflow.py` | Pin updated to the new selector (F5); `continue-on-error` guard; image-mount probe assertion |

`uv run mypy src tests`, `uv run ruff check src tests`,
`uv run ruff format --check src tests`, and `uv run pytest tests/unit -q -n 4`
(6107 passed) all pass. The worktree is clean apart from this file and the
review commit; nothing is pushed and no PR is open.
