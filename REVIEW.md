# Uncorrelated review of implementer tip `ed19d9d4` (#2422)

**Verdict: FAIL** (as submitted). The implementer's stated root cause is wrong and
the patch is a provable no-op; the real defect was adjacent and untouched. Fixed
on this branch (`trigswap3rev`).

## Verdict on the implementer's claim

> "`build_lockdown_verify_script` exact-matched sentinel `169.254.53.53`, but some
> CI `iptables -S` backends canonicalize to `169.254.53.53/32`, so DNAT/REJECT
> looked missing."

**Not true, and the change fixes nothing.** The two greps in question were

```
-d 169.254.53.53.*--dport 443 -j DNAT
-d 169.254.53.53.*-j REJECT
```

They are not exact matches — the `.*` immediately after the address already spans
a `/32` suffix. Both spellings matched before `ed19d9d4` and both match after:

```
$ printf -- '-A OUTPUT -d 169.254.53.53/32 -p tcp -m tcp --dport 443 -j DNAT --to-destination 172.17.0.5:9443\n' \
    | grep -q -- '-d 169.254.53.53.*--dport 443 -j DNAT' && echo MATCHES
MATCHES
```

The commit added a `|| <same grep with /32>` fallback to each — dead alternation
that doubles the sidecar's `iptables` invocations and leaves a comment asserting
a cause that isn't the cause. CI behaviour is unchanged by it.

## Issue 1 (blocker, root cause) — the `:53` DNAT greps can never match

`iptables -S` does not echo the apply command back; it re-prints each rule through
iptables' own formatter, which renders a `--dport` match together with the
protocol match module the parser implicitly loaded:

```
applied:  "$IPT" -t nat -I OUTPUT -p udp --dport 53 -j DNAT --to-destination "$PROXY_IP:5353"
printed:  -A OUTPUT -p udp -m udp --dport 53 -j DNAT --to-destination 172.17.0.5:5353
                           ^^^^^^  inserted by the formatter
```

The two DNS assertions were written against the **apply** spelling and carry no
`.*`:

```
"$IPT" -t nat -S OUTPUT | grep -q -- '-p udp --dport 53 -j DNAT'
"$IPT" -t nat -S OUTPUT | grep -q -- '-p tcp --dport 53 -j DNAT'
```

so they match **no backend, ever** — not a CI-specific canonicalization, a
universal one. Under `set -e` that aborts the verify sidecar, so *every*
credentialed provision failed its read-back while the apply exited 0.

This explains the reported symptoms exactly, including why the two legs report
different causes for one failed grep: both callers' error strings are static.
`apply_network_lockdown` (`setup.py:1137`) says "OUTPUT policy is not DROP after
apply" and `apply_secret_egress_dnat` (`setup.py:1247`) says "nat OUTPUT carries
no DNAT rule after apply" *regardless of which assertion failed*. TASK.md's two
root errors are the verification messages, not the apply messages — which already
pins the failure to the read-back rather than to the apply, the proxy alias,
`dns_port`, or `credential_dns` binding.

Corroboration in-tree that this is the real read-back format (so this is not
inference from memory): `registry.py:165` `_EGRESS_RULE_RE` parses the same
`iptables -S OUTPUT` output and already carries `(?:/32)?` **and** `(?: -m tcp)?`;
every captured fixture in `tests/unit/sandbox/test_egress_refresh*.py` is of the
form `-A OUTPUT -d 1.1.1.1/32 -p tcp -m tcp --dport 443 -j ACCEPT`.

**Fix applied** (`src/aios/sandbox/setup.py`): all four chokepoint assertions are
now EREs matching the read-back spelling, tolerant of both renderings of each
varying field and still requiring every semantic field of the rule. The dead `||`
fallbacks are removed.

```
"$IPT" -t nat -S OUTPUT | grep -qE -- '-d 169\.254\.53\.53(/32)? -p tcp( -m tcp)? --dport 443 -j DNAT'
"$IPT" -t nat -S OUTPUT | grep -qE -- '-p udp( -m udp)? --dport 53 -j DNAT'
"$IPT" -t nat -S OUTPUT | grep -qE -- '-p tcp( -m tcp)? --dport 53 -j DNAT'
"$IPT" -S OUTPUT        | grep -qE -- '-d 169\.254\.53\.53(/32)? -j REJECT'
```

The sentinel-address dots are now escaped (they were unescaped wildcards before),
and the `:443` DNAT grep is *tighter* than what it replaces: `.*` between the
address and `--dport` is now the specific `-p tcp( -m tcp)?`.

## Issue 2 (why this shipped) — the verify unit test never ran the nat greps

`TestBuildLockdownVerifyScript._run_verify` built a fake `iptables` that emitted
`-A OUTPUT -j DNAT --to-destination 1.2.3.4:443` for any `-t nat` call and a bare
policy line for filter — and every caller passed the default `dnat_hosts=()`, so
the four nat/filter greps were only ever asserted as **substrings of the generated
script**, never executed against realistic output. A test that pins the grep text
cannot catch a grep that doesn't match reality.

**Fix applied** (`tests/unit/test_networking.py`): `_run_verify` now renders the
real chokepoint as `iptables -S` prints it, parametrized over both backend
spellings (`canonical` = `/32` + `-m tcp`/`-m udp`; `bare` = neither), with an
`omit` hook to drop individual rules and an `assert_drop` passthrough for the
Unrestricted leg. New tests:

- `test_full_chokepoint_passes_against_real_iptables_s_output[canonical|bare]`
- `test_dnat_only_full_chokepoint_passes[canonical|bare]`
- `test_each_missing_chokepoint_rule_fails_closed[dns_udp|dns_tcp|sentinel_dnat|sentinel_reject]`
- `test_v4_drop_absent_fails_with_chokepoint_installed`

Checked against `ed19d9d4`: the `canonical` variants **fail** (exit 1) and the
`bare` variants pass — precisely isolating the `-m udp`/`-m tcp` rendering as the
defect and confirming the `/32` story was never it. All pass after the fix.

## Confirmed still fail-closed (review item 3)

- Limited `-P OUTPUT DROP` read-back (`grep -qx`) and the guarded v6 DROP are
  untouched; `set -e` still first line on all three script shapes.
- `test_each_missing_chokepoint_rule_fails_closed` proves dropping **any one** of
  the four rules (DNS udp, DNS tcp, sentinel `:443` DNAT, sentinel REJECT) still
  fails the verify — the widened patterns did not loosen into "some DNAT exists".
- `test_v4_drop_absent_fails_with_chokepoint_installed` proves a fully-present
  chokepoint does not mask a missing DROP.
- No cross-table false positive: the filter chain's `-p udp -m udp --dport 53 -j
  ACCEPT` does not satisfy the nat `-j DNAT` grep (verified against a realistic
  captured ruleset).
- `assert_drop=False` still omits the DROP/v6 assertions and keeps all four nat
  assertions; `dnat_hosts=()` still emits no nat reference at all.

## Scope (review items 4, 5)

Minimal and #2422-only: one function's grep patterns plus its test. No IPv6/`-4`
hygiene reopened; no `credential_dns`, `_nat_dnat_lines`, apply-script, registry,
refresh-sweep, or e2e change. #2421 untouched. `_EGRESS_RULE_RE` and
`build_egress_refresh_script` were audited for the same defect — they are already
spelling-tolerant and have no read-back grep, so nothing to change there. The
browser deny-internal verify uses `grep -qF` on true CIDR prefixes (`/16`, `/8`,
…), which iptables prints verbatim — unaffected. Not pushed, no PR.

## Leftover risk

1. **CI is still the oracle.** Docker is unavailable here, so the e2e legs are
   unrun locally. The claim is that the read-back now matches what `iptables -S`
   prints; if a provision still fails, the next thing to read is the verify
   sidecar's stderr, *not* the caller's static error text — see Issue 1.
2. **Only the first failure is visible.** Because the callers' messages are
   static and `set -e` aborts on the first failed assertion, a future verify
   failure will again mis-report its cause. Surfacing the failing assertion
   (e.g. `set -x`, or the sidecar's stderr in the `SandboxBackendError`) would
   have turned this fixround into a log read. Deliberately left out of scope —
   it changes error plumbing on both callers.
3. **The DNAT targets are not verified.** The read-back proves a `:53` DNAT and a
   sentinel `:443` DNAT exist, not that they point at *this session's* resolver
   port / proxy port. Pre-existing (the `:443` assertion never checked its target
   either); closing it means threading `dns_port`/`proxy_port` into
   `build_lockdown_verify_script`. Low value in-netns — nothing else installs a
   nat OUTPUT DNAT there — but it is the remaining gap between "a chokepoint
   landed" and "our chokepoint landed".
4. **Pre-existing xdist flakiness**, unrelated to this change:
   `tests/unit` under `-n 4` intermittently fails/errors in
   `test_jobs_app_layering.py` and `tests/unit/sandbox/test_secret_egress_proxy.py`.
   Reproduced on the unmodified tip `ed19d9d4` (6172 passed + 1 unrelated error);
   all pass serially.

## Verification run

- `uv run mypy src tests` → Success, 1098 files.
- `uv run ruff check src tests` → All checks passed; `ruff format --check` clean.
- `uv run pytest tests/unit -q -n 4` → 6181 passed (+9 new), 1 pre-existing flake.
- `uv run pytest tests/unit/test_networking.py -q` → 138 passed.
- New tests run against `ed19d9d4`'s `setup.py` → 3 failed (reproduces CI).

## Final HEAD

Branch `trigswap3rev`, two commits on top of `ed19d9d4`:

- `5cfbe61b` — `fix(sandbox): match the read-back spelling in lockdown/DNAT verify (#2422)`
  (the product + test change; this is the commit CI should be read against)
- the commit carrying this REVIEW.md, which is the branch tip

REVIEW_DONE
