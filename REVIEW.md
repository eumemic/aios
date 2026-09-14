# Uncorrelated review — implementer tip `3daa40cd` (eumemic/aios#2422)

Maker ≠ checker. Implementer: gpt-5.6-sol on `trigswap4`. Reviewer:
claude-opus-5 on `trigswap4rev`. Scope: the new tip **only**
(`3daa40cd` — *fix(sandbox): route redirected credential DNS replies*), against
the previously reviewed tip `8f07930b`.

## Verdict

**FAIL as submitted.** The commit is correct as far as it goes but is
**necessary-and-not-sufficient**: it repairs the DNS *request* direction and
leaves the *reply* direction still dropped by the kernel, so all four functional
swap legs would have stayed red with the **identical** symptom
(`HTTP_STATUS=000` / empty recorder) — i.e. the fixround would have burned
another CI cycle reporting no change.

Fixes applied on this branch (see *What I changed*). After them I judge the tip
**ready for Shepherd to force-push**, with the explicit caveat that Docker is
unavailable in both workspaces, so **CI remains the only oracle** for the four
legs. No e2e result is claimed here.

## What the commit gets right (each claim audited)

| Claim | Verdict | Evidence |
|---|---|---|
| MASQUERADE is correctly scoped | **Holds** | `_CREDENTIAL_DNS_SNAT` carries only `-d "$PROXY_IP" -p {udp,tcp} --dport {dns_port} -j MASQUERADE`. It cannot catch general egress: the destination is the resolved proxy alias and the port is this session's ephemeral resolver port. |
| Idempotent | **Holds** | `-N … 2>/dev/null \|\| -F …` (create-or-empty) and `-C POSTROUTING -j … 2>/dev/null \|\| -I POSTROUTING -j …` (link-if-absent). A reprovision cannot accumulate duplicates and cannot fail under `set -e`. |
| Safe on reprovision — does not wipe Docker's POSTROUTING | **Holds** | `build_iptables_script` flushes only `-F OUTPUT` and `-t nat -F OUTPUT`. `nat POSTROUTING` is never flushed; only the **private** chain is (`-F AIOS_CRED_DNS_SNAT`). Docker's `DOCKER_POSTROUTING` link and libnetwork's embedded-resolver SNAT survive. |
| Verify uses the read-back spelling, not the apply spelling | **Holds** | New greps are `-qE` and tolerate the implicit match module: `'-p {proto}( -m {proto})? --dport {dns_port} -j MASQUERADE'`. The chain-link grep `'-j AIOS_CRED_DNS_SNAT'` is spelling-invariant. Consistent with `registry._EGRESS_RULE_RE`'s `(?: -m tcp)?` / `(?:/32)?` precedent. |
| Verify fails closed on a missing chain | **Holds** | `iptables -t nat -S AIOS_CRED_DNS_SNAT` exits non-zero when the chain is absent, and the script is `set -e`. |
| `dns_port` is now mandatory when `dnat_hosts` is non-empty | **Holds** | `ValueError` in `build_lockdown_verify_script`; both call sites (`apply_network_lockdown`, `apply_secret_egress_dnat`) pass it. |
| No regression of `8f07930b` | **Holds** | The four pre-existing chokepoint greps are untouched; their spelling-tolerance tests still pass. |

Root-cause narrative in `DONE.md` is also correct for the request direction: with
`/etc/resolv.conf` pointing at Docker's embedded resolver (`127.0.0.11`), the
kernel selects `127.0.0.1` as the source **before** nat OUTPUT rewrites the
destination, and `nf_ip_route_me_harder` re-routes with `FLOWI_FLAG_ANYSRC`,
so the packet leaves on `eth0` still carrying a `127/8` source and is discarded
as a **martian source** on the bridge. MASQUERADE is the right repair for that.

## The hole: the reply is still dropped (why all four legs stay red)

The commit fixes only the request. Trace the reply:

1. Request leaves as `172.17.0.x:sport → worker:dns_port` (post-MASQUERADE).
   conntrack's original tuple still records the **pre-NAT** source `127.0.0.1`.
2. The worker resolver answers. The reply arrives on the sandbox's `eth0`.
3. `nat PREROUTING` reverses the SNAT — the **destination** becomes
   `127.0.0.1` — and this happens **before** the input routing decision
   (`nf_nat_ipv4_pre_routing` runs at `NF_INET_PRE_ROUTING`).
4. `ip_route_input_slow` then evaluates a **loopback destination on a
   non-loopback in-device**:

   ```c
   if (ipv4_is_loopback(daddr)) {
           if (!IN_DEV_NET_ROUTE_LOCALNET(in_dev, net))
                   goto martian_destination;
   }
   ```

   `net.ipv4.conf.*.route_localnet` defaults to **0** and Docker never sets it
   inside a container netns — libnetwork's
   `daemon/libnetwork/resolver_unix.go` (which installs `DOCKER_OUTPUT` /
   `DOCKER_POSTROUTING` in the container netns) contains no `route_localnet`
   write; moby only sets it host-side for published-port DNAT.

So the answer is silently discarded, the credential lookup times out, curl
exhausts its 25 s bound, and the observable symptom after `3daa40cd` is
**byte-identical** to the symptom before it. Both apply and read-back verify
stay green throughout — exactly the "green verify, red functional leg" shape
this fixround already burned three commits on.

The sysctl **cannot** be set from the lockdown sidecar: `run_netns_sidecar`
starts an unprivileged container (`--cap-add NET_ADMIN` only), where `/proc/sys`
is a read-only mount, and Docker refuses `--sysctl net.*` for a container that
shares another container's netns. The sandbox's own `docker run` is the only
place it can be applied.

## What I changed (on `trigswap4rev`)

1. **`SandboxSpec.route_localnet` (`backends/base.py`) + `--sysctl
   net.ipv4.conf.all.route_localnet=1` (`backends/docker.py`).** The completing
   half of the transport fix. `all.*` is enough: the kernel's
   `IN_DEV_NET_ROUTE_LOCALNET` is an OR over `all` and the device.
2. **Gated on credentials, not unconditional (`sandbox/spec.py`):**
   `route_localnet=bool(env_var_credentials)` — true for exactly the sessions
   that install the chokepoint. It is keyed on the **same** value that feeds the
   mount snapshot's `VAULT_CREDENTIAL` tuples, so attaching or detaching a
   credential already recycles the sandbox; the sysctl and the chokepoint
   cannot drift apart. The browser spec (no credentials) keeps the default
   `False`.
3. **Filter INPUT guard (`setup._nat_dnat_lines`), paying for the sysctl.**
   `route_localnet=1` also makes this netns's `127.0.0.0/8` reachable from the
   `aios-sandbox` bridge, whose ICC is **on** (`network.py` passes only
   `--ipv6=false`; only `aios-browser` sets `enable_icc=false`) — a sibling
   sandbox with `CAP_NET_RAW` could address a loopback service directly. So the
   same sidecar run installs, idempotently:

   ```
   "$IPT" -C INPUT '!' -i lo -d 127.0.0.0/8 -m conntrack --ctstate NEW -j DROP 2>/dev/null || \
   "$IPT" -A INPUT '!' -i lo -d 127.0.0.0/8 -m conntrack --ctstate NEW -j DROP
   ```

   `--ctstate NEW` is load-bearing: the redirected DNS answer is **ESTABLISHED**
   by the time it reaches filter INPUT (conntrack runs in PREROUTING; nat's
   `LOCAL_IN` source rewrite runs *after* filter), so a blanket DROP would kill
   the very flow the sysctl exists to allow. `-m conntrack` is already a
   dependency of the Limited script, and both sidecars run the same operator
   image. `-C`/`-A` because INPUT is never flushed.

   **Deliberately NOT added to the read-back verify.** It is defence-in-depth,
   not part of the chokepoint, and this branch has already spent three commits
   (`adaeb155`, `b8054a4c`, `3daa40cd`) on read-back spelling. An unvalidated
   new grep converts a hardening rule into a total provision outage; the
   asymmetry is not worth it. Its presence is pinned by unit tests on the
   emitted script instead.
4. **Fail-closed matrix extended (`tests/unit/test_networking.py`).** The commit
   added three new chokepoint rules but left
   `test_each_missing_chokepoint_rule_fails_closed` parametrized over the old
   four, so none of the new verify greps had a fail-closed test.
   Now eight: `dns_udp`, `dns_tcp`, `dns_snat_chain`, `dns_snat_jump`,
   `dns_udp_snat`, `dns_tcp_snat`, `sentinel_dnat`, `sentinel_reject`.
5. **`_run_verify`'s fake `iptables` made per-chain.** It answered *any*
   `-t nat …` invocation with the whole nat blob, so `-S POSTROUTING` and
   `-S AIOS_CRED_DNS_SNAT` were indistinguishable and a grep aimed at one chain
   could be satisfied by a rule in another. It now filters by chain and exits 1
   for a chain that does not exist — which is what makes the new
   `dns_snat_chain` case a real test.
6. **New tests:** `--sysctl` present iff `route_localnet` (both networking
   modes); the INPUT guard emitted by both `build_iptables_script` and
   `build_secret_egress_dnat_script`; `route_localnet` tracks
   `env_var_credentials` through `build_spec_from_session`.

## Validation (local; Docker unavailable — no e2e result is claimed)

* `uv run mypy src tests` — clean, 1098 files.
* `uv run ruff check src tests` / `ruff format --check src tests` — clean.
* `uv run pytest tests/unit -q -n 4` — **6238 passed**.
* Targeted: `test_networking.py`, `sandbox/test_credential_dns.py`,
  `sandbox/test_egress_refresh.py`, `sandbox/test_egress_refresh_live_path.py`,
  `sandbox/test_spec_env_var_credentials.py`, `test_sandbox_spec.py`,
  `sandbox/test_docker_runtime_argv.py`, `sandbox/test_docker_seccomp_argv.py`
  — 233 passed.
* Generated scripts were rendered and `bash -n`-checked (the `'!'` quoting in
  the INPUT guard is shell-safe).

**Not verified locally, by construction:** kernel/netfilter behaviour. There is
no Docker daemon and no `unshare -n` permission in this workspace, so the
martian-destination claim rests on the kernel source semantics quoted above plus
the moby source read, not on an observation. That is the same epistemic status
as the implementer's own root cause.

## Residual risks CI must settle

1. **gVisor.** `runsc` does not implement `route_localnet`
   (google/gvisor#14625 is still open), so under `AIOS_SANDBOX_RUNTIME=runsc`
   the credential chokepoint cannot work at all. This is **pre-existing** — the
   design has depended on NATing a loopback flow since `3daa40cd` — and it does
   not block the four legs, because `settings.sandbox_runtime` defaults to
   `None` (runc) in CI. It is *not* a startup hazard: `runsc/boot/loader.go`
   reads only `fs.nr_open` from `linux.sysctl` and ignores unknown keys, so the
   flag is inert rather than fatal there. Worth its own issue.
2. **Interception still swallows Docker's embedded name resolution.** All `:53`
   is redirected to the worker resolver, which forwards unknown names to the
   worker's upstream and knows nothing of Docker container names — so
   in-sandbox resolution of `aios-worker` / the proxy alias does not work. This
   predates the tip (it arrived with the `-I OUTPUT -p udp --dport 53` rule) and
   nothing in this fixround makes it worse; flagging it, not fixing it here.

## Shepherd readiness

**Ready to force-push `trigswap4rev` onto the PR branch** and ready for an
eumemic-bot published review, on the understanding that the four e2e(docker)
legs are unverified locally in both workspaces and CI is the decision point. If
the legs are still red after this, the next thing to instrument is the reply
path directly (`conntrack -L` plus `iptables -t nat -L -v -n` counters inside
the sandbox netns), not another verify-grep change.

No push, no PR, no merge, no Track G performed.
