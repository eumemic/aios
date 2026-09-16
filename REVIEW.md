# Uncorrelated review — aios#2410 tip `59b62e3b`

- **Round**: `botpost2410o` (implementer agent `botpost2410p`, grok-4.6)
- **Checker**: claude-opus-5, worktree `aios-botpost2410prev`, branch `botpost2410prev` (maker ≠ checker)
- **Tip reviewed**: `59b62e3b` "fix(sandbox): resolve refresh hosts from an operator-controlled table"
- **Verdict**: **PASS**

## 1. Rebase (TASK §1) — verified

```
origin/master            = 7df8b5d8
HEAD                     = 59b62e3b
git merge-base --is-ancestor origin/master HEAD → true
origin/gvisorgrn         = 59b62e3b   (tip is pushed, PR #2410 updated)
```

Master `7df8b5d8` (post-#2422) is an ancestor of HEAD, so the branch is a true
rebase, not a merge. The #2422 credential/DNAT product is intact on the branch
(`dns_port` threading, `--add-host`/`dnat_target` wiring, the
`credential resolver port missing` fail-closed backstop all read as master's).
`tests/unit/test_gvisor_validation_workflow.py` + `tests/unit/test_detect_filter_sync.py`
(the conflict-resolution surface) pass: 15 passed.

## 2. High — refresh resolves from an operator-controlled source — **closed**

The fix is one primitive, which is the right shape:

- `setup.ResolveScope` (`PROVISION` / `REFRESH`) selects which name sources the
  emitted `resolve_ipv4` may consult, and `scope` is a **required** keyword on
  `build_egress_resolve_script` — a new call site cannot silently inherit the
  unsafe arm.
- `setup._operator_ipv4_fn` bakes a worker-computed name table into the script
  as a shell `case`, consulted **before** anything inside the netns.
- `REFRESH` emits **no** `/etc/hosts` read at all (verified by rendering the
  real script, not just by reading the diff):

```
resolve_ipv4() {
  _operator_ips=$(_operator_ipv4 "$1")
  if [ -n "$_operator_ips" ]; then printf '%s\n' "$_operator_ips"; return 0; fi
  busybox nslookup "$1" 127.0.0.11 2>/dev/null | awk ... | sort -u
}
```

- `registry._refresh_egress_once` is the only `REFRESH` call site and passes
  `state.operator_hosts` stamped at provision. The three `_resolve_egress_hosts`
  call sites are exactly two `PROVISION` (both inside `_stamp_egress_state`,
  reached only from `_apply_egress_rules`) and one `REFRESH`.
- `_apply_egress_rules` has exactly two callers, both immediately after
  `backend.create` on the session and run provisioning paths — no post-provision
  re-apply path re-opens the `PROVISION` arm mid-session.
- The tenant cannot reach the other refresh inputs: the sandbox is **not**
  granted `NET_ADMIN` (docker.py:364), so installed-rule readback is not
  steerable, and all `:53` is DNATed to the worker-controlled resolver.

Input validation on the baked table is real, not assumed: names are re-checked
against `_SCRIPT_HOSTNAME_RE` at emit time (they land in an unquoted `case`
pattern, so glob/metacharacter injection is excluded) and addresses go through
`ipaddress.IPv4Address` (an AAAA literal can never reach the IPv4-only
`iptables -d`). Both raise `ValueError` rather than emitting.

## 3. Medium — runsc/operator chroot can resolve the worker alias — **closed**

The `docker.py` hunk is documentation only; the actual fix is the same operator
table, which is the correct resolution rather than a dodge. `spec.host_gateway_alias`
is `WORKER_NETWORK_ALIAS` exactly when the worker is **not** in a container
(spec.py:1238) — i.e. precisely the shape that gets `--add-host aios-worker:host-gateway`
(docker.py:416) — and `registry._operator_hosts` bakes `{alias: resolve_host_gateway()}`
for that shape and an empty table otherwise. Because the table is consulted
ahead of every in-netns lookup, `$PROXY_IP` now resolves identically under runc
and under the runsc operator chroot, which never saw the sandbox's hosts file.

`network.resolve_host_gateway` asks the daemon the same question the sandbox
asks (`docker run --rm --network none --add-host <unique-probe-alias>:host-gateway
--entrypoint cat <image> /etc/hosts`) rather than guessing the bridge gateway,
which is the right call — it is correct on Docker Desktop (`192.168.65.2`) where
`docker network inspect bridge` is not. Process-lifetime cache under an
`asyncio.Lock` (safe: 3.13 `Lock` has no loop binding at construction); success-only
caching, so a transient probe failure is not poisoned in. Fails hard on probe
error / missing entry / non-IPv4 — consistent with the house "no fallbacks" rule,
and the alternative (a lockdown that silently installs no credential redirect) is
worse.

## 4. Tests (TASK §3) — pass

```
tests/unit/sandbox/test_sandbox_dns_resolution.py
tests/unit/sandbox/test_sandbox_network.py
tests/unit/sandbox/test_egress_refresh.py
tests/unit/sandbox/test_runsc_operator_shadow.py      → 86 passed
```

Adjacent surface, also green (focused files only; **no** full suite, **no** `-n`,
per the ops constraint):

```
tests/unit/test_networking.py
tests/unit/test_sandbox_registry.py
tests/unit/sandbox/test_docker_runtime_argv.py
tests/unit/sandbox/test_placeholder_rotation_331.py   → 277 passed
tests/unit/test_gvisor_validation_workflow.py
tests/unit/test_detect_filter_sync.py                 → 15 passed
```

`uv run mypy src/aios/sandbox/{setup,network,registry}.py` → clean.
`uv run ruff check src tests` → clean; `ruff format --check` → clean.
(`mypy src/aios/sandbox tests/unit/sandbox` was SIGKILLed at 137 in this
container — resource limit, not a type error; the per-file run is authoritative.)

The tests are load-bearing rather than shape-asserting: `_resolve()` executes the
**real emitted shell** under `bash` against a stub `busybox` and a fixture hosts
file, so `test_refresh_scope_never_reads_the_hosts_file` actually demonstrates
that a poisoned `10.0.0.9 example.com` is inert on the refresh arm, and
`test_refresh_scope_emits_no_hosts_file_read_at_all` pins absence rather than
mere ordering. `test_refresh_resolve_bakes_operator_hosts_without_reading_tenant_hosts`
pins the same property one level up, at the sweep.

## 5. DONE.md — matches reality

Every claim checks out: the rebase base (`7df8b5d8`, #2422), "both arms consult a
worker-baked operator table first; only the provision arm still reads the netns
hosts file", the host-gateway probe as the Medium fix, and the four focused test
files. It carries no SHAs, but neither did the previous rounds' DONE.md in this
lineage (`677af073`), so that is the established convention here, not drift.

## Findings (all non-blocking; none affect the verdict)

**F1 — Medium, outside this round's scope: the `PROVISION` hosts read is now
redundant, and its stated premise is not strictly true.**
`setup.py` justifies the provision arm's `/etc/hosts` read as "the only way a
`--add-host` alias resolves at all" and "no tenant process yet". Both are now
weaker than the comments claim:

1. *Redundant*: the only `--add-host` the backend ever passes is
   `{host_gateway_alias}:host-gateway` (docker.py:416), and that is exactly what
   `_operator_hosts` bakes into the table — which is consulted **first**, so the
   hosts entry for that name is unreachable. Every other name the scripts resolve
   (`allowed_hosts`, `PACKAGE_REGISTRY_HOSTS`, credential hosts) is a DNS name.
   The hosts read therefore contributes nothing except a path by which a
   pre-lockdown writer can choose the address an allowed name gets an ACCEPT for.
2. *Premise*: two pre-lockdown in-container execution paths exist.
   `install_packages` runs before `_apply_egress_rules` (registry.py:715-719) and
   `pip install`/`apt-get install` run package-authored code as root; and on the
   snapshot-resume path `run_image = spec.snapshot_image`, a commit of a
   previously tenant-owned rootfs, whose PID 1 (`tail -f /dev/null` via tini)
   resolves through tenant-replaceable binaries. Either can rewrite `/etc/hosts`
   before the sidecar reads it.

   Severity is genuinely limited: that window already has unrestricted egress,
   and a poisoned provision-time ACCEPT is **self-healing** — the DNS-only
   refresh arm evicts it after `_EGRESS_EVICT_AFTER_SUCCESSES=3` ×
   `_EGRESS_REFRESH_INTERVAL_SECONDS=30.0` ≈ 90s.

   Suggested follow-up (not for this round — TASK §2 explicitly sanctions keeping
   hosts-first at provision): drop the hosts read from the provision arm too and
   dissolve `ResolveScope` entirely. One resolver, one enum fewer, and the
   residual closes.

**F2 — Low, doc accuracy.** `test_refresh_resolve_bakes_operator_hosts_without_reading_tenant_hosts`
and the `EgressRefreshState.operator_hosts` comment justify carrying the table on
the refresh state with "if the table were dropped here, the next three ticks
would start evicting the alias's own rules". The refresh host set is
`credential_hosts | limited_hosts` and never contains `WORKER_NETWORK_ALIAS`
(the refresh script takes the proxy address from `state.proxy_ip`, already
resolved), so today the table is defensive on that arm, not load-bearing. The
mechanism is right and worth keeping; the rationale overstates it.

**F3 — Low, informational.** Host-worker deployments now take one extra
`docker run` on the first provision of a worker process and hard-fail provisioning
if the probe fails. That is the intended fail-closed posture and is documented,
but it is a new startup dependency for local dev / e2e. The cache is
process-lifetime, so a daemon `--host-gateway-ip` change needs a worker restart —
correctly called out in the docstring.

## Scope notes

Not run here, by instruction: full unit suite, `pytest -n`, and the e2e suite
(`tests/e2e/test_sandbox_image_contract.py` was touched by this tip and needs
Docker). No push, no merge, no new PR; no gVisor Validation dispatch.
