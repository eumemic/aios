# Uncorrelated review — trigger-swap DNAT e2e (`a57622ac`)

Reviewer: Claude Opus 5, branch `trigswaprev` (worktree `/workspace/aios-trigswaprev`).
Under review: `4d2d6661` (`test(e2e): force IPv4 for trigger secret-egress swap`)
and `a57622ac` (DONE.md). Correction committed as `7da362e5`. Nothing pushed, no
PR opened.

## Verdict: **fail**

The fix does not fix the failure. The diagnosis is on the wrong layer, and the
`-4` flag it motivates is inert against the reported signature: `api.github.com`
publishes no AAAA record, the sandbox bridge has no IPv6 at all, and the Limited
leg carries an explicit `ip6tables -P OUTPUT DROP` — so there is no IPv6 path for
curl to have taken on either leg. MASTER RED will still be red.

The change is not *harmful* — pinning the client address family matches the
IPv4-only chokepoint and is defensible hygiene — so I kept it rather than
reverting. What I removed is the false claim attached to it, in both the source
comment and DONE.md, because a confidently-wrong root cause left in the tree
costs the next implementer more than the flake does.

## Findings

### Fatal

**1. The IPv6 root cause is factually impossible; `-4` cannot change any
observable behaviour on either leg.**

Three independent disproofs, any one sufficient:

* **There is no AAAA to prefer.** `getent ahosts api.github.com` returns only
  the IPv4 address. Method validated against a control on the same resolver:
  `getent ahosts google.com` does return a real `2607:f8b0:…` AAAA, so the
  resolver is not stripping v6. `github.com` likewise has no AAAA. curl cannot
  prefer an answer that does not exist.
* **There is no v6 route out.** The `aios-sandbox` bridge is created
  `--ipv6=false` (`src/aios/sandbox/network.py:61`), so the container holds no
  global IPv6 address. Even a hypothetical AAAA would fail `connect()` and fall
  back to IPv4 within happy-eyeballs — it could not reach a real upstream and
  silently drain the recorder.
* **The Limited leg is doubly covered.** `_IP6TABLES_LOCKDOWN_LINES`
  (`src/aios/sandbox/setup.py`) applies `ip6tables -P OUTPUT DROP` on the
  Limited path. A v6 bypass therefore cannot explain
  `test_trigger_swap_fires_under_limited` being red — yet TASK.md reports
  exactly that leg red on `ae70600b`. The claimed single root cause does not
  cover half the evidence it is offered for.

Corroborating: the run-origin sibling `tests/e2e/test_run_env_var_placeholder.py:82`
issues the **identical** curl (same host, same `--resolve`-free resolution, no
`-4`) and is green on the same runs. A deterministic address-family preference
would take it down too, and would fail 100% of the time rather than flaking.

### Serious

**2. The actual root cause is already documented in-tree and was not consulted.**

TASK.md names the family (`#2019 / #2250 / #2300 / #2042`). `#2042` is the
"KNOWN RESIDUAL" block above `_nat_dnat_lines` in `src/aios/sandbox/setup.py`,
which describes this exact failure in the repo's own words: the sidecar installs
one `-d <ip>` DNAT per address a **single** `getent ahostsv4` happened to return,
and "a rotating pool — api.github.com serves a ~60s-TTL set and returns only a
SUBSET per query — means that set is a SAMPLE, not the pool."

curl re-resolves at fire time, seconds after the sidecar sampled. On a miss:

* **Unrestricted** — filter policy stays ACCEPT, no DNAT matches, the request
  egresses **DIRECT** to the real host. Recorder empty, curl exits 0, audit `ok`.
* **Limited** — no DNAT and no per-host ACCEPT, so it falls through to
  `-P OUTPUT DROP`. Recorder empty, curl times out.

Both terminate at `assert len(recorder.requests) == 1` with `[]` — the reported
signature, on both legs. This is pinned behaviourally by
`TestCredentialHostEgressVerdict` (`tests/unit/test_networking.py:1250`), whose
`_verdict()` returns `direct` / `blocked` for exactly this unsampled-address
case. It also explains the shape of the evidence the IPv6 story cannot: a
per-run coin flip, with Unrestricted red on `127d8453` and Limited red on
`ae70600b`.

The egress-refresh loop that would re-pin rotated addresses (`#1950`,
`794285d3`) is started by the **worker** (`src/aios/harness/worker.py:590`); the
e2e drives `run_trigger_step` directly with no worker, so nothing corrects the
sample mid-test.

**3. DONE.md's verification claim was wrong and understated what was runnable.**

DONE.md stated "`pytest` and `python` commands are not installed" and that
verification was limited to bytecode compilation. In this same worktree
`uv run pytest tests/unit/test_networking.py -q` passes 121 tests in 8s, and
`uv run mypy src tests` / `ruff check` / `ruff format --check` all run. Only
Docker is genuinely absent. The three pre-commit checks CLAUDE.md mandates were
available and were not run.

### Ruled out during review (recorded so the next pass need not redo it)

* **`#2365` SSRF authority-rebind (`b3cdc118`)** — TASK.md lists it as a
  candidate. The guard rejects only non-origin-form request-targets
  (`secret_egress_proxy.py:897`); `curl https://host/trigger-swap-probe` sends
  origin-form. Not implicated.
* **Vault bound after session creation.** `_provision_swap_session_and_trigger`
  calls `set_session_vaults` *after* `docker_harness.start()`, which would strand
  the sandbox without credential DNAT if `start()` provisioned eagerly. It does
  not — `tests/e2e/harness.py:397` only writes agent/session/message rows, and
  provisioning happens lazily inside the fire. Ordering is correct.
* **Proxy instance shared across tests.** `SecretEgressProxy` is constructed
  per-provision (`src/aios/sandbox/spec.py:798`, `:942`), not a worker-wide
  singleton, so the function-scoped `monkeypatch` of `__init__` cannot be
  outrun by a sibling module's proxy.
* **Unrestricted DNAT wiring.** `_apply_egress_rules`
  (`src/aios/sandbox/registry.py:1045`) installs the DNAT-only script for
  Unrestricted-with-credentials. No deterministic gap.

## What I changed, and what I deliberately did not

Committed as `7da362e5`:

* Rewrote the `_SWAP_COMMAND` comment to state that `-4` is a **standing guard**
  — the DNAT is IPv4-only by design (`resolve_ipv4` / `getent ahostsv4`), so a
  future AAAA rollout on a credential host would route the placeholder around the
  proxy under Unrestricted — and not a fix for the observed flake, whose cause it
  now names.
* Rewrote DONE.md to report the branch honestly as *not fixed*, with the
  disproof, the real cause, and corrected verification claims.

I did **not** attempt the real fix. Removing the public-DNS dependency means
either making the container address the exact IP the sidecar pinned (which needs
the installed DNAT read back — `_read_installed_egress_rules` is private and the
`/sessions/:id/egress` surface exposes no IPs) or swapping the credential host
for a netns-deterministic name. Both are test redesigns whose failure modes live
entirely inside Docker, which is absent here. Landing one blind would risk
converting a ~10% flake into a 100% failure, which is a worse outcome than a red
CI with an accurate diagnosis. Recommendation for the next pass, in order of
preference:

1. Provision the sandbox before creating the trigger, read the installed DNAT
   address, and build the command with `--resolve <host>:443:<that ip>`. This
   removes only the DNS nondeterminism: the nat-OUTPUT DNAT is still what moves
   the packet to the proxy, so a genuinely broken DNAT still leaves the recorder
   empty and the test still fails. It does not weaken secret-egress, MitM, or
   DNAT properties.
2. Failing that, retarget `_SWAP_HOST` at a name that resolves deterministically
   inside the sandbox netns, accepting that the test then no longer exercises a
   real internet credential host.

The underlying product hole is `#2042` itself (name-based interception); that is
explicitly a larger change and out of scope here.

## Review checklist

| TASK.md item | Result |
| --- | --- |
| 1. Diagnosis matches code/path reality | **No** — wrong layer; disproved three ways |
| 2. `-4` is the smallest durable fix | **No** — inert; it masks nothing, but fixes nothing |
| 2b. Weakens secret-egress / MitM / DNAT? | No — security properties untouched |
| 3. Both legs covered by the change | Covered, but the change is a no-op on both |
| 3b. Run-origin swap e2e compared | Yes — identical curl without `-4`, green; contradicts the diagnosis |
| 4. Branch not behind origin/master | Yes — 2 ahead, 0 behind `0b07495e` before review commits |
| 4b. DONE.md shas/files match reality | Shas yes; root cause and verification claims no — corrected |

## Verification run for this review

`uv run mypy src tests` (Success, 1096 files), `uv run ruff check src tests`
(passed), `uv run ruff format --check src tests` (clean),
`uv run pytest tests/unit/test_networking.py -q` (121 passed), and
`pytest tests/e2e/test_trigger_fire_env_var_swap.py --collect-only` (2 tests).
Docker is absent, so the e2e legs remain CI-only.

## Final HEAD

Branch `trigswaprev`, 4 commits ahead of `origin/master` @ `0b07495e`, 0 behind.
HEAD is the commit adding this REVIEW.md; the last code-bearing commit is
`7da362e5`. `4d2d6661` and `a57622ac` are the changes under review.
