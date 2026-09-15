# Done

## #2422 functional swap follow-up

The branch is rebased onto `origin/master` at `63337f26` with the true #2042
name-based resolver/sentinel path, the refresh-sweep sentinel guard, and the
iptables `-S` spelling fixes retained.

**Root cause of `HTTP_STATUS=000` / the empty recorder.** The prior fix made
Docker-container DNS leave the netns by OUTPUT-DNATing queries from Docker's
embedded resolver address (`127.0.0.11`) to the worker's per-session resolver.
That changed only the destination. The kernel had already selected
`127.0.0.1` as the source for the original loopback destination; after DNAT,
the resulting `127/8`-sourced packet could not cross the Docker bridge. The
credential DNS query therefore timed out, curl exhausted its 25-second bound
and printed `HTTP_STATUS=000`, and neither run nor trigger ever opened a TLS
connection to the secret-egress proxy or recorder. This is why the apply and
read-back greps could be green while all four functional legs stayed red.

Evidence: on this host a UDP socket connected to `127.0.0.11:53` selects
`127.0.0.1`, while the same socket connected to an external destination selects
the interface address. The failed CI run took the curl timeout and showed
`HTTP_STATUS=000` plus `recorder.requests=[]` in both networking modes, with no
provision failure. The only new runtime seam shared by those four legs was the
loopback-to-worker DNS redirect.

**Fix.** The shared Limited/Unrestricted rule generator now links an
idempotent private nat POSTROUTING chain and MASQUERADEs only UDP/TCP traffic
already redirected to the worker resolver port. That gives the DNS request a
bridge-routable source and lets conntrack restore the reply to the sandbox's
original resolver socket. Reprovision flushes only the private chain, never
Docker's POSTROUTING rules. The fail-closed verify now requires the chain link
and both protocol-specific MASQUERADE rules in addition to the name-based DNS
DNAT, sentinel DNAT, and sentinel REJECT; this is a functional transport fix,
not another relaxation of a grep.

Validation in this Docker-less workspace:

* `uv run pytest -q tests/unit/sandbox/test_credential_dns.py tests/unit/test_networking.py tests/unit/sandbox/test_egress_refresh.py tests/unit/sandbox/test_egress_refresh_live_path.py` — 185 passed.
* The two requested run-origin Docker tests collect and skip only because Docker
  is unavailable locally; CI remains the functional oracle for all four legs.

No push or merge performed.

## Root cause (evidence-backed)

Credential-host interception was keyed on **sampled IP addresses**, not on the
name. `_nat_dnat_lines` in `src/aios/sandbox/setup.py` generated one
nat-OUTPUT DNAT per address that the provisioning sidecar's single
`getent ahostsv4` happened to return. `api.github.com` serves a ~60s-TTL
rotating pool and answers with only a subset per query, so that set is a
*sample*, not the pool — an address nobody sampled is the ordinary case.

This is not a hypothesis: master's own `setup.py` carried it as a **documented
KNOWN RESIDUAL** (eumemic/aios#2042) stating the exact two-mode behavior the
two red legs show —

* **Limited** falls through to the terminal `-P OUTPUT DROP`: the flow is
  dropped, the request fails, the recorder stays empty.
* **Unrestricted** keeps filter policy `ACCEPT`: the flow egresses *directly*
  to the real upstream carrying the literal `AIOS_SECRET_PLACEHOLDER_*`, never
  reaching the proxy, so neither the swap nor the #331 fail-loud fence runs.

The residual was pinned behaviourally by `TestCredentialHostEgressVerdict`
(`tests/unit/test_networking.py`), whose docstring names its own failure as
"the acceptance signal for #2042, not a regression".

**Why trigger-origin is redder than run-origin** (TASK item 4): there is no
trigger-specific provision path — `run_trigger_step`
(`src/aios/harness/trigger_runner.py:580`) calls the same
`sandbox_registry.get_or_provision(...)`, so both legs share one chokepoint.
The asymmetry is timing, not code: the run test curls promptly after provision,
while a trigger must first become due and be dispatched, so far more of the
~60s TTL has elapsed and a rotated, unsampled address is much likelier. The
generic DNAT was never sound; the trigger leg just samples the race later.

**Not an IPv6/`-4` failure.** The IPv6-only story stays rejected. It is also
now moot rather than merely unproven: the resolver answers AAAA/HTTPS/SVCB for
a credential name with **NODATA** (`credential_dns.py:answer`), so the sandbox
cannot obtain an IPv6 address or an `ipv4hint` for a credential host at all.
That is strictly stronger than curl `-4`, which is why no `-4` hygiene is
carried here.

## Fix

Name-based interception (#2042) **rebased onto current master**, not applied
wholesale. Each credential proxy owns a worker-controlled resolver
(`src/aios/sandbox/credential_dns.py`) seeded from `_allowed_hosts` — the same
frozenset that gates leaf minting, so policy has one source. It answers every
credential name with the fixed non-routable sentinel `169.254.53.53` and never
forwards those names; everything else is forwarded verbatim. All sandbox `:53`
(udp+tcp) is DNATed to it with `-I` at the top of nat OUTPUT so no in-netns
resolver — including Docker's embedded 127.0.0.11 — answers first. The netns
then carries exactly one credential rule, keyed on our own constant. An address
nobody sampled cannot bypass the proxy because the name can no longer resolve
to it inside the sandbox. Both modes install a byte-identical chokepoint.

Fail-closed throughout: the sentinel routes nowhere (a broken DNAT denies
rather than leaks); non-:443 sentinel traffic is REJECTed; a resolver that
cannot bind fails proxy start and the provision; a proxy-alias DNS miss is
`exit 1` instead of silently skipping the nat block; the read-back verify
asserts all four chokepoint rules; and the registry refuses a DNAT target
without a resolver port rather than falling back to any address-keyed shape.
The refresh sweep never ADDS a per-address credential DNAT again (that was the
sampling machinery) — it only retires legacy ones from pre-#2042 sessions, and
it can never retire the chokepoint itself: the sentinel is excluded from the
delete set by construction. Without that exclusion the sweep was one aged-out
pin away from deleting the only credential rule (in-sandbox DNS answers every
credential name with the sentinel, so the stamp's read-back pins it, and the
legacy delete shape is byte-identical to the provisioned rule) with nothing
left to re-add it.

Master behavior preserved across the rebase: the #2365 SSRF absolute-form
request-target check, the #2113 ClientHello passthrough, #2309 browser
deny-internal egress, the #2276/#2274 browser control plane, #2331 snapshot
pool-budget LRU reclaim, #2411 snapshot-reset retirement, the #2104/#2124
fail-closed egress inventory, and the #2193 provision report (credential hosts
now report INSTALLED unconditionally — coverage is complete by construction).

## Verification

* `uv run mypy src tests` — clean, 1098 files.
* `uv run ruff check src tests` / `ruff format --check` — clean.
* `uv run pytest tests/unit -q` — 6173 passed.
* Docker is unavailable in this environment, so the two trigger e2e legs were
  not run locally; CI is the oracle. Both collect.

No push, no PR.

## #2422 CI verify follow-up (corrected)

**First attempt (`ed19d9d4`) named the wrong cause and was a no-op.** It
widened the two sentinel greps to also accept a `/32`-canonicalized address —
but those greps are `-d 169.254.53.53.*--dport 443 -j DNAT` and
`-d 169.254.53.53.*-j REJECT`, and the `.*` already spans `/32`. Both spellings
matched before the change and after it; nothing about CI behaviour moved.

**Actual cause.** `iptables -S` does not echo the apply command back — it
re-prints each rule through iptables' own formatter, which renders a `--dport`
match together with the protocol match module the parser implicitly loaded:

```
applied:  "$IPT" -t nat -I OUTPUT -p udp --dport 53 -j DNAT --to-destination …
printed:  -A OUTPUT -p udp -m udp --dport 53 -j DNAT --to-destination 172.17.0.5:5353
                           ^^^^^^
```

The two `:53` DNAT assertions were written against the *apply* spelling
(`'-p udp --dport 53 -j DNAT'`, no `.*`), so they could **never** match a
correctly installed chokepoint on any backend. Under `set -e` that aborts the
verify sidecar, so every credentialed provision failed its read-back — Limited
and Unrestricted alike. The callers' error text is static, which is why the two
red legs *reported* different causes for the same failed grep:
`apply_network_lockdown` says "OUTPUT policy is not DROP after apply" and
`apply_secret_egress_dnat` says "nat OUTPUT carries no DNAT rule after apply",
regardless of which assertion actually failed. That accounts for both root
errors in TASK.md and for all four listed e2e tests (both trigger-swap legs and
both run-origin legs), while the apply itself exited 0 — consistent with the
logs showing the *verification* message, not the apply message.

In-tree corroboration that this is the real read-back format: `registry.py`'s
`_EGRESS_RULE_RE` — which parses the same `iptables -S OUTPUT` output — already
carries `(?:/32)?` and `(?: -m tcp)?`, and the captured fixtures in
`tests/unit/sandbox/test_egress_refresh*.py` are all of the form
`-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 -j ACCEPT`.

**Fix.** All four chokepoint assertions are now EREs (`grep -qE`) matching the
read-back spelling and tolerating both renderings of each varying field —
`(/32)?` on the sentinel address, `( -m tcp)?` / `( -m udp)?` on the protocol
match — while still requiring every semantic field of the rule. The redundant
`||` fallbacks from `ed19d9d4` are removed. Fail-closed is unchanged: dropping
any one of the four rules still fails the verify.

**Test gap closed.** This shipped because `TestBuildLockdownVerifyScript`'s fake
`iptables` emitted `-A OUTPUT -j DNAT --to-destination 1.2.3.4:443` and was only
ever invoked with `dnat_hosts=()`, so no test ran the nat greps against realistic
output. `_run_verify` now renders the real chokepoint in both backend spellings
and can omit individual rules; new tests assert both spellings verify green on
both the Limited and DNAT-only legs, and that omitting any one of the four rules
still fails closed. They fail against `ed19d9d4` in the `canonical` spelling —
i.e. they reproduce the CI failure — and pass after the fix.

`uv run mypy src tests` clean (1098 files); `ruff check` / `ruff format --check`
clean; `uv run pytest tests/unit -q` 6181 passed (one pre-existing xdist-only
flake in `test_jobs_app_layering.py`, which also flakes on the unmodified tip).
Docker is unavailable locally, so the e2e legs remain CI's to confirm.

## #2422 round 4 — the four functional swap legs (uncorrelated review of `22e4d7e1`)

The `22e4d7e1` tip added `--sysctl net.ipv4.conf.eth0.route_localnet=1`
alongside the existing `all` sysctl and rewrote this file down to that claim.
Both the claim and the change are wrong, and the real cause of all four legs is
elsewhere. Reverted; the accumulated root-cause history above is restored (the
tip deleted 186 lines of it).

### The `eth0` sysctl is a no-op against its stated cause

`22e4d7e1`'s premise was that "setting only `net.ipv4.conf.all.route_localnet=1`
did not reliably enable that check for Docker's already-created `eth0`". The
kernel reads this flag through `IN_DEV_ORCONF`, i.e.
`conf.all.route_localnet || conf.<dev>.route_localnet`, evaluated per packet at
route time — it is not copied into the device at creation. Demonstrated
directly in a user namespace, with the veth created BEFORE `all` was written:

```
--- initial: all=0 veth0=0
INPUT test  (martian dest, 127.0.0.1 via veth0):  RTNETLINK answers: Invalid argument
OUTPUT test (loopback saddr out veth0):           RTNETLINK answers: Invalid argument
--- now: all=1 veth0=0        (veth0 predates this write)
INPUT test  with ONLY all=1:  local 127.0.0.1 from 10.9.9.2 dev lo     (rc=0)
OUTPUT test with ONLY all=1:  10.9.9.2 from 127.0.0.1 dev veth0        (rc=0)
```

`all=1` alone lifts both martian checks on a pre-existing device. The added
sysctl cannot change any packet's fate.

### …and it is an active provisioning hazard

`docker run --sysctl net.ipv4.conf.eth0.*` is applied by runc at task creation,
before the network endpoint exists, and fails the whole `docker run` with
"no such file or directory" on engines that do not carry moby#47686 (moby#47619,
docker/cli#4990); newer engines migrate such keys to per-endpoint
`com.docker.network.endpoint.sysctls` DriverOpts. So the change trades zero
upside for a failure mode that aborts provisioning for EVERY credentialed
sandbox — strictly worse than four red asserts. `docker.py` now emits `all` and
only `all`, with the ORCONF reasoning and the moby hazard recorded at the site
so it is not "fixed" again; the unit assertion is exact (`== [all]`) rather than
a containment check.

### Actual root cause: the DNS reply leaves the worker with the wrong source

The #2042 chokepoint makes DNS the **first UDP hop** from a sandbox to the
worker. Every other hop — tool broker, git proxy, secret-egress proxy — is TCP,
where an accepted socket's local address is pinned to the SYN's destination.
UDP has no such pinning, and `CredentialDnsResolver` binds `0.0.0.0` and
answered with a plain `transport.sendto(...)`, which lets the ROUTE pick the
reply's source address.

The worker is multihomed. In CI (`ubuntu-latest`, pytest on the host, so
`is_running_in_container()` is False) the sandbox gets
`--add-host aios-worker:host-gateway`, and `host-gateway` resolves to the
**default bridge** gateway (`docker0`, 172.17.0.1) — not to the gateway of the
user-defined `aios-sandbox` bridge the sandbox actually lives on. The query
therefore arrives addressed to 172.17.0.1, while the route back to the sandbox
leaves via `br-aios-sandbox`, so the reply was emitted with source 172.18.0.1.
The sandbox's conntrack entry expects 172.17.0.1, the tuple does not match, and
the reply is discarded before the un-SNAT/un-DNAT can restore it. glibc's stub
resolver uses a **connected** socket, so even without conntrack the kernel drops
a reply from the wrong peer: the process never sees it and the lookup times out.
curl then exhausts `--max-time 25` and prints `HTTP_STATUS=000`; no TLS
connection is ever opened, so the recorder stays empty — identically in both
networking modes, and immune to every iptables spelling, MASQUERADE and
`route_localnet` fix this fixround has tried. That is exactly the shape of the
four surviving legs.

Evidence (both Docker-free):

* Two-netns repro — unconnected client sees the asymmetry directly:
  `reply b'REPLY' FROM ('10.2.0.1', 5353) (queried 10.1.0.1:5353)`; the same
  exchange from a *connected* client: `FAILED: TimeoutError`.
* Pure loopback repro (now a unit test): a `0.0.0.0`-bound server queried at
  `127.0.0.2` by a connected client — `IP_PKTINFO=False: FAILED: TimeoutError`
  / `IP_PKTINFO=True: got b'REPLY'`.

### Fix

`credential_dns.py` now does what every real resolver does (BIND, unbound,
dnsmasq): it enables `IP_PKTINFO` on the UDP socket, reads the query with
`recvmsg`, and echoes the queried local address back on the reply via `sendmsg`
with an `in_pktinfo` control message whose `ipi_spec_dst` is the address the
kernel itself computed as the correct reply source (`fib_compute_spec_dst`).
The reply is then sourced from the address the sandbox queried, whichever of the
worker's addresses that is, so conntrack matches and the stub resolver accepts
it.

This needs the raw socket rather than a `DatagramTransport` (a transport cannot
attach control messages), so the UDP listener is a `loop.add_reader` on the
bound socket; per-query tasks, truncate-at-512, SERVFAIL-on-error, the
never-log-the-query rule and `stop()` semantics are unchanged, and the TCP path
is untouched. `IP_PKTINFO` is Linux-only; on a developer macOS worker the
socket option is absent and the reply falls back to `sendto`, which is correct
there because that topology is not multihomed between sandbox and worker.

Alternatives rejected: binding the resolver to a specific address (the worker
cannot know which of its addresses a given sandbox will use), and repointing the
`aios-worker` alias at the sandbox bridge gateway (much larger blast radius —
the alias also carries the tool broker, git proxy and echo URLs).

### Verification

* `uv run pytest tests/unit/sandbox/test_credential_dns.py -q` — 19 passed. The
  new `TestReplySourceAddressMatchesTheQueriedAddress` leg FAILS with
  `TimeoutError` against `22e4d7e1`'s resolver and passes after the fix.
* `uv run pytest tests/unit/test_networking.py tests/unit/sandbox -q` — 693 passed.
* `uv run mypy src tests` clean (1098 files); `ruff check` / `ruff format --check`
  clean; `uv run pytest tests/unit -q -n 4` green.
* Docker is unavailable in this workspace, so the four e2e legs remain CI's to
  confirm; no claim of e2e green is made here.

No push, no PR.
