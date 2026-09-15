# Uncorrelated review — aios#2410 fixround `botpost2410m`

**Verdict: PASS.**

| | |
|---|---|
| Reviewed tip | `88c0d3b2625d204380adf4c712bf9255101f0ecc` |
| Final HEAD | this `docs(review)` commit on `88c0d3b2` (no product changes) |
| Review branch | `botpost2410mrev` (this worktree; not pushed) |
| Implementer | claude-opus-5 on `botpost2410m` |
| Checker | grok-4.6 on `botpost2410mrev` |
| Parent / `origin/gvisorgrn` | `d21d38623ecc0a13e9c011ba5fb591b1e9ddb0d4` — FAIL review-docs only; parent of the product commit |
| `origin/master` | `63337f26378ddd2c5ce1567faed60f38d8a5a63e` — ancestor of the tip |

One product commit ahead of `origin/gvisorgrn`. Rebase was not needed (`origin/gvisorgrn` and `origin/master` are both ancestors). Shape 2 (out-of-netns IP inject) is not in the diff. No push, no PR.

---

## Scope vs TASK

TASK asked for shape 1 only:

1. `resolve_ipv4` (the single helper used for PROXY_IP / `dnat_target` alias resolution): awk `/etc/hosts` **first**, then busybox nslookup.
2. Must not restore `getent`.
3. Document the tenant-writable hosts caveat on the refresh path.
4. Hosts-only oracle/unit if feasible.
5. Rebase if needed; true DONE.md for this round; do not push.

All five hold. Files in `d21d3862..88c0d3b2` are exactly those TASK listed: `src/aios/sandbox/setup.py`, `src/aios/sandbox/backends/docker.py` (comment only), `tests/unit/sandbox/test_sandbox_dns_resolution.py`, `DONE.md`.

---

## Findings

No blocking findings. The round does the thing the last three rounds skipped.

### Hosts-first lands in the emitted scripts

`setup._RESOLVE_IPV4_FN` is no longer DNS-only. Evaluated at this tip it is:

```sh
resolve_ipv4() {
  _hosts_ips=$(awk -v name="$1" '{ sub(/#.*/, "") } $1 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { for (i = 2; i <= NF; i++) if ($i == name) { print $1; next } }' /etc/hosts 2>/dev/null | sort -u)
  if [ -n "$_hosts_ips" ]; then printf '%s\n' "$_hosts_ips"; return 0; fi
  busybox nslookup "$1" 127.0.0.11 2>/dev/null | awk '/^Name:/ { answer = 1 } /^Address:/ && answer && $2 != "127.0.0.11" && $2 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { print $2 }' | sort -u
}
```

That function is interpolated into `build_iptables_script`, `build_secret_egress_dnat_script`, and `build_egress_resolve_script` — the three scripts that actually call `resolve_ipv4` for PROXY_IP / allow-list / credential-host lookups. There is no sibling resolver. `build_egress_refresh_script` continues to take an already-resolved `(proxy_ip, port)` and does not look names up; the names it consumes come from the resolve script, which now has hosts-first.

The CI failure chain this is aimed at (`resolve_ipv4 aios-worker` → empty `PROXY_IP` → skipped nat block → `nat OUTPUT carries no DNAT rule after apply`) is closed on the runc lane, where `--network container:` bind-mounts the sandbox's `/etc/hosts` (including `--add-host aios-worker:host-gateway`) into the sidecar.

Properties checked against the emitted text, not the Python comments:

- `/etc/hosts` is consulted **before** `busybox nslookup` in every generated script that defines the helper.
- `getent` is absent from those scripts. `test_no_script_touches_resolv_conf_or_getent` still pins both `getent` and `/etc/resolv.conf` out; `TestIPv4OnlyResolution.test_no_script_resolves_through_glibc` does the same on a wider script set.
- IPv4-only on both halves (dotted-quad on `$1` for hosts, `$2` for DNS). AAAA / `::1` cannot reach `iptables -d`.
- Both halves are pipelines ending in `sort -u`, so a missing hosts file or a failed nslookup cannot abort `set -e` (no `pipefail` on these scripts). Covered by `test_dns_still_answers_when_there_is_no_hosts_file`.
- `awk` / `sort` were already in `_RUNSC_OPERATOR_COMMANDS`; no new unshadowed binary. `test_script_runs_no_unshadowed_command` still applies.

### getent stays out; shape 2 was not smuggled

`docker.py` product diff is the `_RUNSC_OPERATOR_STATIC_COMMANDS` comment only — it now says `resolve_ipv4` *falls back* to busybox nslookup and that the hosts scan needs no extra shadow. `spec.host_gateway_alias` is still `--add-host <alias>:host-gateway`. No address is resolved on the host and baked into the script.

Runsc remains DNS-fallback-on-miss, as documented: the exec chroots into the operator image and reads *that* `/etc/hosts`, which has no alias. TASK called that shape 2 and out of scope. DONE.md states it rather than hiding it.

### Caveat is on the refresh path

`build_egress_resolve_script`'s docstring carries the tenant-writable `/etc/hosts` caveat in full (refresh tick; tenant can retarget an already-allowed *name*, not admit a new one). `_RESOLVE_IPV4_FN`'s comment block summarises the same and points at that function. That is what TASK asked for.

DONE.md also flags the order trade (DNS-first-with-hosts-fallback would have avoided the exposure while still catching a DNS-miss `aios-worker`) and correctly notes the task pinned hosts-first. Not re-decided here.

### Hosts-only oracles assert something real

Prior DNS oracles all queried names the stub resolver answers, so they could not see this bug. New tests in `tests/unit/sandbox/test_sandbox_dns_resolution.py` query names the stub **fails** (or answers differently):

- `test_add_host_alias_resolves_without_dns` — hosts-only name, DNS miss.
- `test_hosts_file_is_consulted_before_dns` — order: hosts short-circuits rather than merging.
- `test_dns_still_answers_when_the_hosts_file_has_no_entry` / `…when_there_is_no_hosts_file` — prefix, not replacement; missing file is a miss, not an abort.
- Parse tests: aliases, `#` comments, IPv6 lines, partial-name non-matches.
- `test_hosts_only_proxy_alias_still_installs_the_dnat` — the CI-failure oracle at **rule** level: real `build_secret_egress_dnat_script`, recording iptables shims, stub busybox that fails `aios-worker`, fixture hosts with `172.17.0.1 aios-worker`, then asserts the two `-t nat -A OUTPUT … -j DNAT --to-destination 172.17.0.1:49152` lines. A DNS-only helper would skip the nat block and record zero DNAT rules.

`_resolve()` now retargets `_HOSTS_FILE` at a fixture, so the old DNS assertions no longer depend on the test machine's `/etc/hosts`.

### DONE.md vs reality

DONE.md is a true report for **this** round (`botpost2410m`), not a stale i/k write-up. It correctly states:

- tip is one product commit on `d21d3862` / `origin/gvisorgrn`; no rebase;
- root cause (`9b246ab7` dropped nsswitch `files`);
- the emitted helper shape;
- getent still banned;
- runsc / shape 2 out of scope;
- tenant-writable hosts on refresh;
- e2e not run here.

The `6210 passed` unit-suite figure is a chunked sum under host memory pressure, with an explicit caveat that `-n 4` OOMs. Not independently re-counted here. Targeted tests were re-run (below). DONE does not claim e2e green.

---

## Checks run this review

```
git fetch origin gvisorgrn master
# origin/gvisorgrn = d21d3862, ancestor of HEAD
# origin/master   = 63337f26, ancestor of HEAD

uv run pytest tests/unit/sandbox/test_sandbox_dns_resolution.py \
              tests/unit/sandbox/test_runsc_operator_shadow.py \
              tests/unit/test_networking.py::TestIPv4OnlyResolution -q
# 36 passed

uv run ruff check / format --check on the three Python files touched
# clean
```

Emitted scripts inspected via `setup.build_*`: hosts-first present, `getent` / `/etc/resolv.conf` absent.

E2E was not run (no Docker in this container). CI remains the e2e oracle; this review does not claim those six credential-swap tests green.

---

## Nits (non-blocking, not fixed)

- `DockerBackend.run_netns_sidecar` still describes `_RESOLVE_IPV4_FN` as “passes the embedded resolver's address to busybox nslookup as an argument” (`docker.py` ~1343). True of the DNS half; it does not mention the new hosts scan. The nearby `_RUNSC_OPERATOR_STATIC_COMMANDS` comment *was* updated. Incomplete, not wrong.
- `_nat_dnat_lines` still says “proxy-alias DNS miss” for the empty-`PROXY_IP` guard. The miss can now be hosts+DNS.

Neither is in TASK scope and neither changes behaviour.

---

## Verdict

**PASS.** Shape 1 is in the emitted helper, getent is still pinned out, the hosts-only DNAT oracle would have failed on the pre-fix helper, shape 2 was not smuggled, DONE.md matches this tip, rebase was unnecessary. Remaining e2e proof is CI's, not this host's.
