# REVIEW — Master RED gVisor validation after #2432

- **Product tip (PR):** `6805d03d` — implement `f480501c` + High leftover (`/etc/hosts` worker alias, not `--dns` gateway)
- **Round:** `gvisred2` / `gvisred2rev`
- **Implementer:** grok-4.6 · **Reviewer:** claude-opus-5
- **Base:** `origin/master` @ `8f3111f0` (#2432)
- **Evidence:** https://github.com/eumemic/aios/actions/runs/35157209667 — 11 fail / 364 pass. Related #923.

## Verdict: **PASS** (after leftover)

Initial review **FAIL** on leg (c): `--dns <bridge gateway>` cannot make `aios-worker` resolve under runsc (Docker `--dns` is only an upstream forwarder; gateway is not a resolver). High leftover applied on this tip as `6805d03d` (`/etc/hosts` / extra_hosts). Legs (a) SizeRw overshoot and (b) seccomp unshare denial from `f480501c` kept.

---

## Findings

### High — (c) `--dns <gateway>` cannot make `aios-worker` resolve under runsc — FIXED in this worktree

The tip added `resolve_sandbox_network_gateway()` (`src/aios/sandbox/network.py`)
and passed `--dns <bridge gateway IP>` on runsc creates, on the stated rationale
that "dockerd actually answers alias lookups" at the gateway. Both halves of
that are false:

1. **`--dns` does not replace the sandbox's resolver.** On a user-defined
   network Docker always writes `127.0.0.11` as the container's *only*
   nameserver and treats `--dns` as the *upstream forwarder* the embedded
   resolver consults for names it cannot answer itself. Passing `--dns` never
   changes what the sandbox queries, so the Sentry keeps sending every lookup
   to the one address the tip itself identifies as unreachable.
2. **Nothing answers DNS on the bridge gateway.** libnetwork binds the embedded
   resolver on `127.0.0.11` *inside the container's netns* (ephemeral port, plus
   netns-local DNAT/SNAT rules that rewrite `:53`). The gateway is the host's
   bridge interface, not a resolver. (Gateway-answers-DNS is the
   Podman/aardvark-dns shape, not Docker's.)

The tip's *diagnosis* of the root cause is right — the Sentry never replays the
netns netfilter rules, so `127.0.0.11` is unreachable (google/gvisor#7469) — but
gVisor's own guidance for this failure points the other way: a real external
resolver for external names, static addresses / `extra_hosts` for service names.

**Fix applied (`d6e924d7`):** use the primitive this codebase already chose for
exactly this problem. The egress scripts stopped looking the alias up inside the
netns and instead have the *worker* resolve it and bake the answer in
(`_operator_ipv4`, aios#2410). The tenant side now does the same:

- `resolve_network_alias_ipv4(alias, network)` (`sandbox/network.py`) reads the
  endpoint Docker itself recorded for the alias — `docker ps --filter network=`
  then `docker inspect` parsed as JSON (never per-field templating under
  `missingkey=error`), matching on `Aliases` ∪ `DNSNames`.
- runsc creates pass `--add-host aios-worker:<addr>`. libc reads the hosts file
  before DNS and runsc passes that bind mount through untouched, so the alias
  resolves in the Sentry without the embedded DNS.
- Scoped to the shape that needs it: runc keeps using the embedded DNS (no
  probe, no hosts entry); the worker-on-host shape already has its own
  `--add-host` and is not on the sandbox network. No container claiming the
  alias logs a warning and emits nothing rather than inventing an address —
  provisioning's own proxy-alias resolve is already the hard failure for a
  sandbox that must reach the broker.

`resolve_sandbox_network_gateway` / `_probe_network_gateway` and their caches are
deleted (don't-deprecate-delete), and `TestResolveSandboxNetworkGateway` is
replaced by `TestResolveNetworkAliasIPv4` (6 cases: alias via `Aliases`, alias
via `DNSNames`, no claimant, empty network skips the inspect entirely,
degenerate/vanished records don't poison the batch, listing failure fails hard).
`test_docker_runtime_argv.py` now asserts the runsc create carries
`--add-host aios-worker:<addr>` and no `--dns`, that a worker-less network omits
the entry, that the host-worker shape keeps its own host-gateway `--add-host`,
and that runc emits neither the `docker ps` probe nor an `--add-host`.

### Medium — (a) the `docker diff` veto is inert under a default runsc install; `--overlay2=none` is the whole fix

The diagnosis is correct: `--overlay2=root:self` **is** the runsc default, and it
hides rootfs writes from Docker's image store, which is why a written layer
reported `SizeRw == baseline` and snapshot returned `skipped_empty`. Setting
`"runtimeArgs": ["--oci-seccomp", "--overlay2=none"]` in the CI daemon.json is
the real repair, and it is the only part of leg (a) that changes the failing
tests' behavior.

The accompanying product change — stamping `docker diff` paths at create and
vetoing `size_is_empty` when new paths appear — does not carry its weight:

- **Inert in the very configuration it claims to defend against.** With the
  self-backed overlay, tenant writes land *inside the pre-existing filestore
  file* (`.gvisor.overlay.img.{CID}/filestore-*`), which is already in the
  create-time stamp. No new diff path appears, so the veto never fires. It only
  helps in configurations where `docker diff` already sees the writes — i.e.
  where `SizeRw` mostly works anyway.
- **Path-set comparison misses in-place growth.** A tenant that only *modifies*
  paths already present at create time (the common case for an image with a
  pre-seeded workspace) produces the same path set, so the veto passes it
  through as empty.
- **Costs a `docker diff` per create for every runtime**, including runc where
  `SizeRw` is reliable.
- **Fails closed to "commit" on an unreadable diff** (`current_paths is None` →
  `size_is_empty = False`), which reintroduces the exact overshoot #2432 fixed
  whenever the daemon call is flaky.

Not fixed here: the zero-write `skipped_empty` tests still pass with it in place,
and removing it is a design call about whether a second, weaker empty-signal is
wanted at all. Recommendation for the PR discussion: drop the create-time diff
stamp and let `--overlay2=none` (plus the #2432 baseline identity) carry leg (a),
or promote it to a real content signal rather than a path-set diff.

### Medium — (b) seccomp restoration is CI-config-only, with no runtime assertion

The diagnosis is sound and the fix is right: runsc's `--oci-seccomp` defaults
**off**, so the authored profile never reached the Sentry and `CLONE_NEWUSER`
succeeded (exit 0 / `0 0 0 0`). Adding it to the CI daemon.json `runtimeArgs`
restores `test_unshare_user_namespace_denied` and
`test_unshare_argfilter_distinguishes_our_profile`.

The gap is that this is a *CI workflow* change. A production runsc deployment
that does not carry the same `runtimeArgs` silently drops the sandbox seccomp
profile — the sandbox looks configured (the profile is authored and passed to
Docker) while enforcing nothing. The only record of the requirement is a
`Field(description=...)` string in `src/aios/config.py`; there is no startup
assertion and no e2e that would catch an operator missing it outside CI.

Suggested follow-up (not applied — it is a new behavior, outside this review's
repair scope): when `sandbox_runtime == "runsc"`, assert at worker startup that
the daemon's runtime args include `--oci-seccomp` and fail hard, consistent with
the fail-hard/no-fallbacks stance. A silently-unenforced seccomp profile is
precisely the class of failure that policy exists for.

### Low (note) — Limited networking under runsc + worker-in-container still has no name source

`_operator_hosts` returns `NO_OPERATOR_HOSTS` in the worker-in-container shape,
and under runsc the egress scripts read the *operator image's* hosts file and
cannot see the embedded DNS at all. So `aios-worker` resolves for the tenant
(after the fix above) but still resolves nowhere for the egress scripts, leaving
Limited networking nameless in that shape. This is pre-existing and orthogonal to
the three clusters; it is the residual tracked by #923.

Natural follow-up now that the primitive exists: populate the operator table from
`resolve_network_alias_ipv4` for the runsc + worker-in-container combination.
Comments in `sandbox/setup.py` (`NO_OPERATOR_HOSTS`) and `sandbox/registry.py`
(`_operator_hosts`) were corrected in `d6e924d7` — they previously claimed the
runsc sandbox gets no `--add-host` at all, which is no longer true — and now
state the residual explicitly.

### Note — commit message / diff correspondence

Legs (a) and (b): the commit body's rationale matches the diff and matches
reality (`--overlay2=root:self` default hiding rootfs writes; `--oci-seccomp`
defaulting off). Leg (c): the diff matches the stated rationale, but the
rationale itself — that dockerd answers alias lookups on the bridge gateway — is
factually wrong, which is the High finding above.

---

## Checks run (focused only; no full suite, no `-n`)

```
uv run pytest tests/unit/sandbox -q -p no:randomly        # 822 passed
uv run mypy src tests                                     # Success (1101 files)
uv run ruff check src tests                               # clean
uv run ruff format --check src tests                      # clean
```

Docker / gVisor e2e was not run here (out of scope per TASK.md item 4); the three
named e2e clusters remain the CI gate.

## State of this worktree

- Branch `gvisred2rev`, tip `d6e924d7` on top of the reviewed `f480501c`.
- Commit `d6e924d7` contains only the High fix and its tests/doc corrections:
  `sandbox/network.py`, `sandbox/backends/docker.py`, `sandbox/registry.py`,
  `sandbox/setup.py`, `tests/unit/sandbox/test_sandbox_network.py`,
  `tests/unit/sandbox/test_docker_runtime_argv.py`.
- `TASK.md` is modified in the working tree by the harness (the committed copy
  still carries the older aios#2410 task text); deliberately left uncommitted.
- Not pushed, not merged, no PR opened — Shepherd owns leftover apply + push.
