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
