# Uncorrelated review — trigger-swap DNAT e2e legs

- **Branch:** `trigswap2rev` (worktree `/workspace/aios-trigswap2rev`)
- **Reviewed tip:** `6fe45254` (on `9441ef39`, both on master tip `0b07495e`)
- **Verdict: fail** — the reviewed tip was a wholesale revert of master, not an
  integration. The *diagnosis* it claims is correct and is kept; the delivery
  destroyed thirteen shipped master behaviors and left the tree unable to
  typecheck or even collect its unit tests. Fixed on this branch.
- **Final code HEAD:** `197c9c45` — all review fixes. `HEAD` is the
  commit that adds this file on top of it (a commit cannot carry its own
  sha); `git log --oneline -1` prints it.
- Not pushed; no PR opened; still on `trigswap2rev`.

## What the tip actually did

`9441ef39` applied the July-2026 #2042 PR tip as a **whole-file replacement**
onto `0b07495e`. Evidence, all from the reviewed tip:

| file | tip's diff vs master | true #2042 delta |
| --- | --- | --- |
| `src/aios/sandbox/registry.py` | 126 insertions / **1148 deletions** | 40 / 5 |
| `src/aios/sandbox/secret_egress_proxy.py` | 29 / **289** | 29 / 1 |
| `src/aios/sandbox/setup.py` | rewritten | +410 / −313 |
| `tests/unit/test_networking.py` | master suites deleted | additive |

Behaviors deleted by the revert (each verified restored at grep parity after
the rebase): #2365 SSRF absolute-form request-target rejection; #2113
ClientHello raw dispatcher + `_relay_unrecognized_sni` passthrough; #2276/#2274
account-browser substrate and control plane; #2309 browser L3 deny-internal
egress; #2331 snapshot pool-budget LRU reclaim; #2411 snapshot-reset
retirement; #2104/#2124 `egress_unread_hosts` fail-closed inventory; #2193
typed egress-provision lifecycle events; `sandbox_owner_kind()`'s callers.

Consequences at the reviewed tip:

- `uv run mypy src` — **8 errors** (`spec.py:798,942` unexpected
  `networking_mode`/`owner_id`; `browser.py:71` missing
  `get_or_provision_browser`; `browser_control.py:286,287,501` missing
  `owner_lock`/`release_browser`/`touch_browser`).
- `uv run pytest tests/unit` — **could not collect**:
  `ImportError: cannot import name 'egress_unread_hosts' from
  'aios.sandbox.setup'` (`tests/unit/sandbox/test_egress_refresh.py:26`).

So "unit 130 passed on two test files" in the tip's DONE.md was true only of
those two files; the tree as a whole did not run.

## TASK verification items

1. **Is #2042's name-based path already on origin/master?** **No** — the task's
   premise is false. `src/aios/sandbox/credential_dns.py` does not exist on
   `origin/master`, and master's `_nat_dnat_lines` generated one DNAT per
   sampled address. Integrating it is therefore the right move, not redundant.
   Master's own `setup.py` carried the defect as a **documented KNOWN
   RESIDUAL** naming exactly the two failure modes the red legs show, and
   `TestCredentialHostEgressVerdict` pinned it behaviourally ("the acceptance
   signal for #2042, not a regression"). The implementer's root cause is
   **confirmed**, not rubber-stamped.
2. **Does the fix install credential DNS + sentinel DNAT on the trigger path,
   fail-closed?** Yes. There is no trigger-specific provision path:
   `run_trigger_step` (`src/aios/harness/trigger_runner.py:580`) calls the same
   `sandbox_registry.get_or_provision(...)`, which reaches `_apply_egress_rules`
   → `apply_network_lockdown` (Limited) or `apply_secret_egress_dnat`
   (Unrestricted). Both emit the byte-identical `_nat_dnat_lines` block. I
   rendered the generated scripts for both modes plus the read-back verify and
   audited them rule by rule. Fail-closed: resolver bind failure fails proxy
   `start()` and the provision; a proxy-alias DNS miss is `exit 1` rather than a
   skipped nat block; non-`:443` sentinel traffic is REJECTed; the sentinel is
   non-routable, so a broken DNAT denies rather than leaks; the registry
   *refuses* a DNAT target with no resolver port instead of falling back to any
   address-keyed shape. Resolver host set and DNAT host set both derive from the
   same `cred.allowed_hosts`, so intercepted names and TLS-terminated names
   cannot drift.
3. **IPv6 / `-4` story.** Not revived. It is now *moot* rather than merely
   unproven: the resolver answers AAAA/HTTPS/SVCB for a credential name with
   **NODATA**, so the sandbox cannot obtain an IPv6 address or an `ipv4hint`
   for a credential host at all — strictly stronger than curl `-4`. No `-4`
   hygiene is carried.
4. **Run-origin vs trigger-origin.** Same code path (item 2), so the asymmetry
   is *timing*: `test_run_env_var_placeholder.py` curls promptly after
   provision, while a trigger must first become due and be dispatched, so much
   more of the ~60s TTL has elapsed and a rotated, unsampled address is far
   likelier. The address-keyed DNAT was never sound; the trigger leg just
   samples the race later.
5. **Hygiene.** Branch is 3 ahead / **0 behind** `origin/master`. The tip's
   DONE.md shas/files matched the commits, but its *claims* did not match the
   tree (above). `is_run_owner_id` was **not** real or needed: at `6fe45254` its
   only callers were in the *reverted* `registry.py`; under a correct rebase it
   is dead code (master discriminates with `sandbox_owner_kind()`, per the
   CLAUDE.md "kind, never a boolean flag" rule). Removed — `src/aios/ids.py` is
   now byte-identical to master.

## Second finding: the sweep could retire the chokepoint

Found while auditing the rebased refresh sweep — a fail-open hole in #2042
itself, not in the rebase:

In-sandbox DNS answers every credential name with the sentinel, so
`_stamp_egress_state`'s rule read-back sees the one provisioned sentinel DNAT,
and `_seed_pinned_from_installed` pins it like any other address.
`build_egress_refresh_script`'s `legacy_dnat_tail` delete is **byte-identical**
to that provisioned rule, and since #2042 nothing ever *adds* a credential
DNAT. One tick whose resolve came back without the sentinel would therefore
age the pin out and **permanently delete name-based interception** — under
Unrestricted that is direct egress carrying the literal placeholder, i.e. the
original defect reintroduced by the fix's own maintenance path.

Fixed by excluding `CREDENTIAL_SENTINEL_IP` from the delete set by
construction, covered by
`test_live_refresh_never_retires_the_credential_sentinel_dnat`, which drives
the pin all the way to eviction (verified load-bearing: the test fails with the
guard reverted).

## Fixes applied on this branch (`197c9c45`)

- Located the true #2042 base (`d4644691`, #2041) and did a real 3-way rebase:
  `git checkout origin/master -- <files>` + `git diff d4644691 6fe45254 |
  git apply -3`, resolving eight conflicts by hand. Net diff vs master is now
  additive (1755 / 330); `registry.py` is 46 changed lines, not 1274.
- Conflict resolutions kept master's side wherever the two disagreed: #2193
  report rows re-added inside the sentinel block (INSTALLED unconditionally —
  coverage is complete by construction, no per-host skip remains); master's
  fail-closed `egress_unread_hosts` inventory kept in the refresh builder with
  only the credential *add* removed; `apply_secret_egress_dnat` keeps master's
  `EgressProvisionResult` return; the proxy's `start()` keeps #2113's raw
  dispatcher and starts the resolver first (fatal on failure).
- Sentinel excluded from the refresh delete set (above).
- Removed dead `is_run_owner_id`.
- Retargeted the two master tests whose only observed "add" was the removed
  per-address credential DNAT (`test_egress_refresh.py`,
  `test_egress_refresh_live_path.py`) onto the limited-host ACCEPT shape that
  remains, preserving each test's stated subject.
- Replaced two stale e2e comments that described the old `-d <ip>` pinning
  (`tests/e2e/test_trigger_fire_env_var_swap.py`,
  `tests/e2e/test_run_env_var_placeholder.py`). `_SWAP_HOST = api.github.com`
  and the `--resolve`-free curl are deliberately unchanged: the honest
  chokepoint exercise is the point, and no e2e weakening was used.
- Rewrote DONE.md to the evidence-backed root cause.

## Gates

- `uv run mypy src tests` — `Success: no issues found in 1098 source files`.
- `uv run ruff check src tests` — `All checks passed!`;
  `ruff format --check` — `1098 files already formatted`.
- `uv run pytest tests/unit -q` — **6173 passed**, 0 failed (123s).
- openapi/SDK snapshot invariants unaffected (no API-layer change); their
  tests pass.
- Docker is unavailable in this environment, so
  `test_trigger_swap_fires_under_unrestricted_dnat_only` and
  `test_trigger_swap_fires_under_limited` were **not run locally**; both
  collect, and Code Validation is the oracle. Expected-green rests on the
  argument in item 2, not on a local run.

## Residual risk

The chokepoint now depends on DNS interception rather than on address samples.
If the `-I` DNS DNAT rules were ever absent while the sandbox ran, a credential
name would resolve to a real address and — under Unrestricted — egress direct.
That is checked at provision by the read-back verify (all four chokepoint rules
asserted) and is no longer reachable through the refresh sweep after the fix
above, but it is the one invariant the design now rests on and is worth an
explicit eye in review of any future change to `_nat_dnat_lines` or
`build_lockdown_verify_script`.
