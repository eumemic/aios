# Done

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

## #2422 CI verify follow-up

The read-back verifier now accepts both forms emitted by `iptables -S` for the
sentinel address (`169.254.53.53` and `169.254.53.53/32`).  Some CI backends
canonicalize the apply rule to `/32`, so the previous exact bare-address grep
reported a missing DNAT/REJECT despite successful installation; Limited then
surfaced this as the misleading OUTPUT-DROP verification error.  DNS (UDP/TCP),
sentinel HTTPS DNAT, and sentinel REJECT checks remain independent and
fail-closed.  `python3 -m compileall` and `git diff --check` pass; pytest and
Docker are unavailable locally.
