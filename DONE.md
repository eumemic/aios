# Done — aios#2410, round `botpost2410m`: hosts-first `resolve_ipv4` (shape 1)

Branch `botpost2410m`, on top of `d21d3862` — which is the `origin/gvisorgrn`
tip, so no rebase was needed (`origin/master` is an ancestor). One product
commit. **Not pushed**, no PR; the Shepherd pushes.

Shape 1 as scoped: `/etc/hosts` first, then `busybox nslookup`. Shape 2 (IP
injection) is **not** in this diff — see "What this does NOT fix" below.

---

## The regression

`9b246ab7` replaced `getent ahostsv4` with `busybox nslookup <host>
127.0.0.11` because no sidecar shape can supply the `/etc/resolv.conf` glibc
needs (that reasoning still stands, and `getent` stays out). But `getent` had
been doing **two** lookups, not one: nsswitch's `files` before its `dns`. The
replacement kept only `dns`.

`aios-worker` has two resolution paths, as `aios/sandbox/network.py` says in
its own module docstring: Docker's embedded DNS when the worker sits on the
sandbox network, and `/etc/hosts` when the worker runs on the **host** — the
e2e and host-worker shape, where the sandbox is created with `--add-host
aios-worker:host-gateway` (`backends/docker.py`, `spec.host_gateway_alias`).
Docker writes an `--add-host` alias into the container's `/etc/hosts` and does
**not** publish it to the embedded resolver, so a DNS-only lookup sees only the
first path.

The failure chain, all of it silent:

```
resolve_ipv4 aios-worker  →  (nothing)
PROXY_IP=$(… | head -n1)  →  ""
if [ -n "$PROXY_IP" ]      →  false, entire nat block skipped
apply script               →  exit 0
```

which is CI's `nat OUTPUT carries no DNAT rule after apply`, and the six
credential-swap/placeholder failures behind it: with no DNAT, :443 to a
credential host goes DIRECT to the real upstream carrying the opaque
`AIOS_SECRET_PLACEHOLDER_*`, so the proxy's swap never runs.

Bisect from the task: `e10f4e07` green, `2e0225cc` red — `9b246ab7` sits
between them.

## The fix

`src/aios/sandbox/setup.py`, `_RESOLVE_IPV4_FN`:

```sh
resolve_ipv4() {
  _hosts_ips=$(awk -v name="$1" '{ sub(/#.*/, "") } $1 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { for (i = 2; i <= NF; i++) if ($i == name) { print $1; next } }' /etc/hosts 2>/dev/null | sort -u)
  if [ -n "$_hosts_ips" ]; then printf '%s\n' "$_hosts_ips"; return 0; fi
  busybox nslookup "$1" 127.0.0.11 2>/dev/null | awk '/^Name:/ { answer = 1 } /^Address:/ && answer && $2 != "127.0.0.11" && $2 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { print $2 }' | sort -u
}
```

Properties that were deliberately preserved:

* **No `getent`, no `/etc/resolv.conf`.** The hosts half is read with `awk`
  directly. `test_no_script_touches_resolv_conf_or_getent` still pins both
  strings out of every generated script, unchanged.
* **No new binary on the lockdown path.** `awk` and `sort` were already in
  `_RUNSC_OPERATOR_COMMANDS`, so the runsc shadow map needs no entry and
  `test_script_runs_no_unshadowed_command` passes as-is.
* **IPv4-only on both halves.** The hosts scan keeps only dotted quads, so the
  `::1 localhost`/`fe80::` lines every `/etc/hosts` carries can never reach an
  IPv4 `iptables -d` and abort the apply under `set -e` — the same property the
  DNS half's AAAA filter has.
* **Fail-closed, and `set -e`-safe.** A miss on both halves prints nothing, so
  the host gets no rule. Both halves are pipelines ending in `sort -u`, so
  neither a missing/unreadable `/etc/hosts` (awk exits 2) nor an nslookup
  failure can return nonzero out of a `$(…)` capture and abort the caller.
* **`#` comments stripped before field splitting**, so a commented-out entry an
  operator believes is gone cannot install a rule.

`_HOSTS_FILE`, `_HOSTS_LOOKUP_AWK`, `_NSLOOKUP_PARSE_AWK`, `_HOSTS_LOOKUP_CMD`
and `_NSLOOKUP_CMD` are named constants purely to keep the two awk programs
readable and to let a test retarget the lookup at a fixture file; the emitted
script is one function, as before, emitted once per script.

## Oracles

The old DNS oracles all query names the stub resolver answers, so none of them
could see this bug. The new ones in
`tests/unit/sandbox/test_sandbox_dns_resolution.py` all query a name the stub
resolver **fails** (or answers differently), so the hosts file is the only
thing that can produce the assertion:

* `test_add_host_alias_resolves_without_dns` — hosts-only name resolves.
* `test_hosts_file_is_consulted_before_dns` — *order*: a hosts entry
  short-circuits the DNS query rather than merging with it.
* `test_dns_still_answers_when_the_hosts_file_has_no_entry` /
  `…when_there_is_no_hosts_file` — hosts-first is a prefix, not a replacement,
  and a missing file is a miss rather than an abort.
* `test_hosts_aliases_and_comments_are_parsed_like_the_resolver_would`,
  `test_hosts_ipv6_entries_never_reach_the_ipv4_rules`,
  `test_partial_name_matches_are_not_answers` — the parse itself.
* `test_hosts_only_proxy_alias_still_installs_the_dnat` — **the oracle for the
  CI failure**, asserted at the rule, not at the lookup. It runs the real
  generated DNAT-only apply script against recording `iptables` shims, a stub
  busybox that fails `aios-worker`, and a fixture hosts file holding the
  `--add-host` line, then asserts the two `-t nat -A OUTPUT … -j DNAT
  --to-destination 172.17.0.1:49152` rules are recorded.

That last test was checked against the **pre-fix** helper (same script with
`resolve_ipv4` reduced to its DNS-only body): the script still exits 0, prints
`AIOS_EGRESS_SKIPPED api.secret.com  proxy alias aios-worker has no IPv4
address`, and records **zero** DNAT rules. So it fails on the old code for the
same reason CI did, which is what makes it an oracle rather than a restatement
of the helper.

`_resolve()` in that module now also points the helper at a fixture hosts file
instead of the test machine's `/etc/hosts` — without that the pre-existing DNS
assertions would silently depend on whatever a developer or CI image left
there.

## What this does NOT fix (stated, not hidden)

* **runsc.** The runsc exec chroots into the read-only operator image before
  running the script, so it reads the *operator* image's `/etc/hosts`, which
  carries no alias. The lookup misses and falls through to DNS — i.e. exactly
  today's behaviour. No regression, no fix. Closing it needs the address
  resolved outside the netns and injected into the script (shape 2), which is
  explicitly out of scope this round. The red lane is runc, where the sidecar
  joins with `--network container:<id>` and Docker bind-mounts the *target's*
  `/etc/hosts` into it — alias included — so this is the lane hosts-first
  greens.
* **Tenant-writable `/etc/hosts` on the refresh path.** On the runc shape the
  file the sidecar reads is the sandbox's, and root inside the sandbox can
  write it. Provision-time applies are out of reach (the lockdown lands before
  any tenant code runs), but `build_egress_resolve_script` runs on every
  refresh tick, by which point the tenant has had the container. A tenant entry
  cannot introduce a **name** — only operator-configured hosts are ever looked
  up — but it can choose the **address** an already-allowed name resolves to,
  and so the address a refreshed `ACCEPT`/`DNAT` is installed for. Documented in
  full on `build_egress_resolve_script`'s docstring and summarised at
  `_RESOLVE_IPV4_FN`. Closing it needs the same out-of-netns resolution shape 2
  needs.

  Worth putting in front of the Shepherd explicitly: **DNS-first with a hosts
  fallback would fix the reported bug without that exposure** (`aios-worker` is
  a DNS miss either way, so the fallback would still catch it, while a tenant
  hosts entry for a name that *does* resolve would be ignored). It was not
  taken because the task pins the order — "awk `/etc/hosts` **first** then
  busybox nslookup" — and because hosts-first is what `getent` did and what
  every other resolver in the container still does, so the firewall's idea of a
  name matches the container's. Flagging the trade, not re-deciding it.

## Also in the diff

`backends/docker.py`: the `_RUNSC_OPERATOR_STATIC_COMMANDS` comment said
`resolve_ipv4` "runs `busybox nslookup`"; it now says it *falls back* to it and
notes the hosts scan needs no binary beyond the already-shadowed `awk`/`sort`.
Comment only.

## Checks

```
uv run ruff check src tests            All checks passed!
uv run ruff format --check src tests   1098 files already formatted
uv run mypy src                        Success: no issues found in 333 source files
uv run mypy tests                      Success: no issues found in 765 source files
uv run pytest tests/unit               6210 passed, 0 failed  (437 files, see caveat)
uv run pytest tests/unit/sandbox/test_sandbox_dns_resolution.py   16 passed
```

**Caveat on how the unit suite was run.** This host has ~1.5 GB free of 16 GB
(the rest is not this worktree's), and a single long-lived process is
OOM-killed regardless of `-n`: `uv run mypy src tests` dies with rc 137, as
does `uv run pytest tests/unit` serially — twice, at 24% and at 75%, with no
summary line. So both were run split: mypy as `src` then `tests` (above), and
the unit suite as 18 sequential chunks of 25 files, each its own process
(chunk 9 was itself OOM-killed and re-run as five batches of five). Every
chunk exited green; the 6210 is their sum, covering all 437 files with no
file skipped. Under `-x -n 4` the same tree produced 14 failures in modules
this diff does not touch (`test_import_hygiene`, `test_tool_broker`,
`test_config`, `test_git_proxy`, …); each passes on its own and all pass in
the chunked run, i.e. host memory pressure under parallelism, the same
flakiness the previous round documented — not this change.

E2E was not run here (no Docker in this container). The e2e assertion this
round targets is `nat OUTPUT carries no DNAT rule after apply`; the unit-level
stand-in for it is `test_hosts_only_proxy_alias_still_installs_the_dnat`,
which asserts on the recorded `iptables -t nat -A OUTPUT … -j DNAT` lines of
the real generated script and which was confirmed to fail against the pre-fix
helper.
