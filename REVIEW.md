# Uncorrelated review — implementer tip `22e4d7e1` (eumemic/aios#2422)

Maker ≠ checker. Implementer: gpt-5.6-sol on `trigswap5`. Reviewer:
claude-opus-5 on `trigswap5rev`. Scope: the new tip **only**
(`22e4d7e1` — *fix(sandbox): enable route_localnet on Docker interface*),
against the previously reviewed tip `074553a5`.

## Verdict

**FAIL.** The tip is a **no-op** against the failure it claims to fix, its
stated root cause is **falsified** by direct experiment, and it introduces a
provisioning **hazard** that can abort `docker run` for every credentialed
sandbox. It also deleted 186 lines of evidence-backed root-cause history that
TASK.md requires. Shipping it would have burned another CI cycle reporting no
change on all four legs.

Fixes applied on this branch (see *What I changed*): the per-interface sysctl is
reverted, and the **actual** cause of the four legs — the credential DNS reply
leaving the worker with the wrong source address — is fixed. Docker is
unavailable in this workspace, so **CI remains the only oracle**; no e2e result
is claimed here.

## Finding 1 — the `eth0` sysctl cannot change any packet's fate

The commit's premise, quoted from its `DONE.md`:

> Setting only `net.ipv4.conf.all.route_localnet=1` did not reliably enable
> that check for Docker's already-created `eth0`, so the reply was discarded as
> a martian.

That is not how the flag is read. The kernel evaluates it through
`IN_DEV_ORCONF` — `conf.all.route_localnet || conf.<dev>.route_localnet` —
at route time, per packet. `conf.all` is consulted dynamically; it is not
snapshotted into a device when the device is created, so "already-created
`eth0`" is not a condition that exists.

Demonstrated in a user namespace, with the veth created **before** `all` was
written (`unshare -rn`, `ip route get` exercising both the un-DNATed-reply
direction and the loopback-source direction):

```
--- initial: all=0 veth0=0
INPUT test  (martian dest, 127.0.0.1 via veth0):  RTNETLINK answers: Invalid argument
OUTPUT test (loopback saddr out veth0):           RTNETLINK answers: Invalid argument
--- now: all=1 veth0=0        (veth0 predates this write)
INPUT test  with ONLY all=1:  local 127.0.0.1 from 10.9.9.2 dev lo     (rc=0)
OUTPUT test with ONLY all=1:  10.9.9.2 from 127.0.0.1 dev veth0        (rc=0)
```

`all=1` alone lifts both martian checks on a device that predates it. The added
sysctl is redundant by construction.

## Finding 2 — …and it is an active provisioning hazard

`docker run --sysctl net.ipv4.conf.eth0.*` is handed to runc and applied at
**task creation**, before the network endpoint exists. On engines that do not
carry moby#47686 (which migrates `eth0` sysctls to per-endpoint
`com.docker.network.endpoint.sysctls` DriverOpts) this fails the whole
`docker run` with "no such file or directory" — moby#47619, docker/cli#4990.
Whether CI's engine is new enough is not something this PR should be betting
on: the downside is that **every credentialed sandbox fails to provision**,
against an upside of exactly zero (Finding 1). Reverted.

## Finding 3 — the real root cause of all four legs (not addressed by the tip)

**The credential DNS reply leaves the worker with the wrong source address.**

The #2042 chokepoint made DNS the first **UDP** hop from a sandbox to the
worker. Every other chokepoint hop — tool broker, git proxy, secret-egress
proxy — is TCP, where an accepted socket's local address is pinned to the SYN's
destination, which is exactly why only the new DNS hop breaks.
`CredentialDnsResolver` binds `0.0.0.0` and replied with
`self._transport.sendto(response, addr)`, so the **route** picked the reply's
source address.

The worker is multihomed, and in the e2e topology the two directions do not
agree on which address is "the worker":

* CI's e2e docker shard runs `pytest tests/e2e` directly on `ubuntu-latest`
  (no `container:` key in the job), so `is_running_in_container()` is False and
  `spec.host_gateway_alias` is `aios-worker` → the sandbox gets
  `--add-host aios-worker:host-gateway`.
* Docker resolves `host-gateway` to the **default bridge** gateway
  (`docker0`, e.g. 172.17.0.1) — *not* to the gateway of the user-defined
  `aios-sandbox` bridge (e.g. 172.18.0.1) the sandbox actually sits on.
* So the query arrives addressed to 172.17.0.1, while the route back to the
  sandbox leaves via `br-aios-sandbox` and the reply is sourced 172.18.0.1.

The sandbox's conntrack entry expects a reply from 172.17.0.1; the tuple does
not match, so the un-SNAT/un-DNAT never runs. Independently, glibc's stub
resolver uses a **connected** UDP socket, so the kernel drops a datagram from
the wrong peer before the process ever sees it. The lookup times out, curl
exhausts `--max-time 25` and prints `HTTP_STATUS=000`, no TLS connection is ever
opened so the recorder stays empty — **identically in both networking modes**,
and immune to every iptables-spelling, MASQUERADE and `route_localnet` change
this fixround has tried. That is precisely the shape of the four surviving legs,
and it is why "green verify, red functional leg" persisted across three tips.

Evidence, both Docker-free:

* **Two-netns repro.** Unconnected client, wildcard-bound server:
  `reply b'REPLY' FROM ('10.2.0.1', 5353) (queried 10.1.0.1:5353)` — the reply
  source is not the queried address. Same exchange from a *connected* client:
  `FAILED: TimeoutError`.
* **Pure loopback repro** (now a unit test): `0.0.0.0`-bound server, connected
  client querying `127.0.0.2` — `IP_PKTINFO=False: FAILED: TimeoutError` /
  `IP_PKTINFO=True: got b'REPLY'`.

## Finding 4 — necessary-and-sufficient assessment

| Question | Answer |
|---|---|
| Is the `eth0` sysctl necessary? | **No** — `all` is an OR with the per-device value (Finding 1). |
| Is it sufficient? | **No** — it changes nothing, and the legs fail for an unrelated reason (Finding 3). |
| Is `route_localnet` itself still needed? | **Yes** — the un-DNATed reply still restores a `127.0.0.11` destination on a non-`lo` device. `all` alone, kept. |
| Is the MASQUERADE from `3daa40cd` still needed? | **Yes** — it fixes the *request* direction's martian source. Untouched. |
| Does anything else regress? | No behavior change outside the two sysctl argv entries and the UDP listener. |

## Finding 5 — DONE.md was gutted

`22e4d7e1` rewrote `DONE.md` from 200 lines to 18, deleting the #2042
root-cause analysis, the `iptables -S` read-back-spelling analysis, and the
request-direction martian-source analysis — and replacing them with the claim
Finding 1 falsifies. TASK.md requires "DONE.md with evidence-backed root cause".
Restored, with this round's evidence appended.

## Test adequacy

The tip's unit change (`sysctls == [all, eth0]`) asserted the argv it had just
written, which is the failure mode the review is supposed to catch: it cannot
distinguish a correct chokepoint from a broken one, because the sysctl is not
observable in any test that does not run a container. The assertion is kept
exact but now pins `all`-only, and its docstring records *why* `eth0` must not
come back.

The real gap was that **no test exercised the reply direction of the DNS hop at
all** — every existing `test_credential_dns.py` case talks to `127.0.0.1` from
an unconnected socket, which is precisely the configuration that cannot observe
this bug. That gap is now closed on loopback, with no Docker required.

## What I changed

1. **`54db852d` — `fix(sandbox): set only the aggregate route_localnet sysctl`.**
   `docker.py` emits `--sysctl net.ipv4.conf.all.route_localnet=1` and nothing
   else; the ORCONF reasoning and the moby per-interface-sysctl hazard are
   recorded at the call site. `test_networking.py` asserts the exact list.
2. **`5b54f9e1` — `fix(sandbox): source credential DNS replies from the queried
   address`.** The resolver enables `IP_PKTINFO`, reads queries with `recvmsg`,
   and replies with `sendmsg` carrying an `in_pktinfo` whose `ipi_spec_dst` is
   the address the query was sent to (`fib_compute_spec_dst` — the kernel's own
   answer to "what source would you use to reply to this packet"). This needs
   the raw socket rather than a `DatagramTransport` (a transport cannot attach
   control messages), so the UDP listener is a `loop.add_reader`; per-query
   tasks, truncate-at-512, SERVFAIL-on-error, the never-log-the-query rule and
   `stop()` semantics are unchanged, and TCP is untouched. `IP_PKTINFO` is
   Linux-only: on a developer macOS worker the option is absent and the reply
   falls back to `sendto`, which is correct there because that topology is not
   multihomed between sandbox and worker. New regression test
   (`TestReplySourceAddressMatchesTheQueriedAddress`) is red with
   `TimeoutError` against `22e4d7e1` and green after. `DONE.md` restored and
   extended.

Alternatives considered and rejected for the reply-source fix: binding the
resolver to a specific address (the worker cannot know which of its addresses a
given sandbox will use — the sandbox picks it via `/etc/hosts` or Docker DNS),
and repointing the `aios-worker` alias at the sandbox-bridge gateway (much
larger blast radius: the same alias carries the tool broker, the git proxy and
the echo URLs).

The #2042 name-based path is **kept** in full — sentinel, resolver, NODATA on
AAAA/HTTPS/SVCB, fail-closed verify. No wholesale master revert; no change to
the request-direction MASQUERADE or to any verify grep.

## Residual risks (flagged, not fixed here)

1. **`host-gateway` points at the wrong bridge.** The reply-source fix makes the
   asymmetry harmless, but the sandbox still reaches the worker at the *default*
   bridge address while living on `aios-sandbox`. That works only because the
   host's docker0 address is reachable from the sandbox bridge. Pinning
   `--host-gateway-ip` (or resolving the alias to the `aios-sandbox` gateway)
   would remove the asymmetry at the source; it touches every chokepoint hop, so
   it belongs in its own change, not in a fixround.
2. **gVisor/runsc inertness of `--sysctl net.*`** (carried over from the
   `074553a5` review): inert rather than fatal there, still worth its own issue.
3. **Interception swallows Docker's embedded name resolution** for in-sandbox
   lookups of container names (predates this fixround; unchanged).

## Verification in this workspace

* `uv run mypy src tests` — clean, 1098 files.
* `uv run ruff check src tests` / `ruff format --check src tests` — clean.
* `uv run pytest tests/unit/sandbox/test_credential_dns.py -q` — 19 passed;
  the new leg fails against `22e4d7e1`'s resolver, passes after the fix.
* `uv run pytest tests/unit/test_networking.py tests/unit/sandbox -q` — 693 passed.
* `uv run pytest tests/unit -q` — 6239 passed (serial). Note for whoever reads a
  local xdist run: `-n 4` is not a usable signal in this workspace — the box has
  ~200 MB free, the OOM killer takes a worker down mid-run (`[gwN] node down`,
  confirmed in `dmesg`), and that cascades into a scattered set of unrelated
  failures (image resize, litellm, ssh loopback…). Every one of them passes
  serially. CI's own `-n 4` runs on a machine that is not memory-starved.
* Docker is absent (no binary, no socket), so the four e2e legs were **not** run.
  **CI is the oracle; no e2e green is claimed.**

## Final HEAD

Last code commit: **`5b54f9e1`** (`fix(sandbox): source credential DNS replies
from the queried address`). The branch tip is the docs-only commit that carries
this file, on top of it.

No push, no PR, no merge, no Track G performed.
