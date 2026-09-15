# Review — aios#2410 fixround tip `f3c847d6` (branch `botpost2410jrev`)

**Verdict: FAIL** — blocking.

Maker: gpt-5.6-sol on `botpost2410j`. Checker: claude-opus-5 on `botpost2410jrev`.
Scope reviewed: `2e0225cc..f3c847d6` (one commit). Nothing was pushed; no PR opened.

## Summary

TASK asked for three items. **Item 1 (the actual product fix) was not attempted.**
The entire round is a one-line YAML change.

```
$ git diff --stat 2e0225cc..HEAD
 .github/workflows/eumemic-bot-review.yml | 1 +
 1 file changed, 1 insertion(+)
```

| TASK item | Status |
|---|---|
| 1. Fix credential-swap / DNAT so the four swap + placeholder e2e go green | **NOT DONE** — zero code changed |
| 2. Restore `persist-credentials: false` on the agent-job checkout | Done, and correct |
| 3. Rebase onto origin/master if behind | N/A — not behind (verified) |

Success criterion "e2e(docker) green" is **not met and cannot be met** by this tip:
the failing code is byte-identical to the base the failure was reported against.

---

## Finding 1 (BLOCKING) — the DNAT / credential-swap regression is untouched

`f3c847d6` changes no Python. `src/aios/sandbox/setup.py` at this tip is identical
to `2e0225cc`, the exact tree that produced the reported failure. The six e2e
failures will reproduce unchanged.

### Root cause, from the CI oracle (not inference)

Job log of the cited run (`34822598014`, job `103907475055`, head `2e0225cc`):

```
6 failed, 404 passed, 3 skipped
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_bash_env_var_placeholder_round_trip - KeyError: 'stdout'
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_unrestricted_dnat_only - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_limited       - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_limited - assert 0 == 1 (recorder.requests == [])
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_unrestricted_dnat_only - assert 0 == 1
FAILED tests/e2e/test_env_var_placeholder_materialized.py::test_placeholder_visible_in_container_secret_absent - KeyError: 'stdout'
```

All six are one cause, logged explicitly in the same job:

```
sandbox.secret_egress_dnat_verify_failed  exit_code=1
run_sandbox.provision_failed: "secret-egress DNAT verification failed …:
  nat OUTPUT carries no DNAT rule after apply; refusing to run an
  env-var-credentialed sandbox whose secret-swap DNAT is unverified"
```

The `KeyError: 'stdout'` and `HTTP_STATUS=000`/empty-recorder symptoms are downstream
of that fail-closed provision abort, not independent bugs.

Two further facts from the log narrow it precisely:

* **The apply script exits 0** — `network_lockdown_failed` appears **0 times**. So the
  script runs to completion; it simply emits no DNAT rule. That happens only when
  `resolve_ipv4` returns nothing: the block is guarded by `if [ -n "$PROXY_IP" ]`
  (`setup.py:460`) and `for ip in $ips` (`setup.py:470`).
* The Limited-path message *"OUTPUT policy is not DROP after apply"* (`setup.py:1037`)
  is a **static string** on any nonzero verify exit; `build_lockdown_verify_script`
  also greps `nat -S OUTPUT` for `-j DNAT` (`setup.py:926`). Both families therefore
  reduce to the same missing-DNAT cause. The message is misleading and cost
  diagnosis time — worth fixing while in here.

### Bisected to the busybox change (CI evidence, both sides)

| tip | run | e2e result |
|---|---|---|
| `e10f4e07` (before `9b246ab7`) | 34807834078 | **1 failed, 408 passed** — only `test_image_layer_carries_the_embedded_dns_resolver` |
| `2e0225cc` (after `9b246ab7` + `dca77114`) | 34822598014 | **6 failed, 404 passed** — the whole swap/placeholder family |

The swap family was **green** immediately before the DNS rework and red immediately
after. `9b246ab7` replaced `getent ahostsv4 "$1"` (with `_RESOLV_PREAMBLE` writing
`nameserver 127.0.0.11`) by `busybox nslookup "$1" 127.0.0.11` + an awk parse;
`dca77114` then added a `$2 != "127.0.0.11"` exclusion. TASK's hypothesis
("nslookup-by-arg path may have broken credential swap") is **confirmed**.

### Why the new oracles did not catch it — the replacement tests cover the wrong context

`tests/e2e/test_sandbox_image_contract.py::test_busybox_nslookup_answers_from_the_embedded_dns`
did **not** fail in run 34822598014, and `tests/unit/sandbox/test_sandbox_dns_resolution.py`
passes locally (I ran `tests/unit/sandbox tests/unit/test_networking.py`: **681 passed**).
So the applet exists, the output shape is as assumed, and the awk parse is right —
in a **plain container attached by `--network <name>`**, which is what that oracle runs
(`docker run --network <net> … /usr/bin/busybox nslookup <alias> 127.0.0.11`).

The lockdown resolves in a different context: a sidecar joined with
`--network container:<id>` (`backends/docker.py:1370-1381`), running `bash -c` under
`set -e` **after** `"$IPT" -F OUTPUT` and `"$IPT" -t nat -F OUTPUT` (`setup.py:688-689`,
`769`). The unit test drives the real emitted function against a **stub busybox** the
implementer wrote, so it pins the assumed shape, not reality. Net effect: green oracles
sitting beside a red functional path — the oracle asserts the wrong context.

I did **not** determine the exact mechanism (no Docker here, and the two candidate
explanations — the `nat -F OUTPUT` flush removing Docker's `-A OUTPUT -d 127.0.0.11/32
-j DOCKER_OUTPUT` redirect, vs. a busybox-specific query behaviour — are not separable
from the log). I am deliberately not guessing: this is the fail-closed path that gates
whether a credentialed sandbox runs at all, and a speculative edit that cannot be
verified here would be worse than the honest finding. Per TASK's own instruction, this
is documented as blocking rather than rewritten.

### Recommended next step for the implementer

The apply script already prints `AIOS_EGRESS_SKIPPED <host>` / `AIOS_EGRESS_INSTALLED <host>`
markers, but `apply_secret_egress_dnat` / `apply_network_lockdown` only feed stdout to
`_parse_egress_provision_result` on the **success** path — on the verify-failure path
stdout is never logged, which is why the CI log contains no `AIOS_EGRESS_*` line at all.
Logging `result.stdout` alongside the existing `exit_code` in the two verify-failure
warnings (`setup.py:1030`, `1134`) would say in one CI run whether the proxy alias or
the credential host failed to resolve. That is a two-line diagnostic, not the fix.

---

## Finding 2 (BLOCKING, record integrity) — DONE.md is stale and materially false

`DONE.md` was last committed in `9b246ab7`, the **previous** round. It was not touched
by `f3c847d6`. It documents a different branch and a different scope:

* Header: *"Branch `botpost2410i`, on top of `e10f4e07`"* — this branch is `botpost2410j(rev)` on `f3c847d6`.
* *"Two items were asked for. Both are done."* — this round's TASK asks three, and only item 2 was done.
* It describes the resolv.conf/nslookup and operator-image-mount work, i.e. the commits
  **already in the base** (`9b246ab7`, `e10f4e07`), and claims them as this round's output.
* Its closing claim *"the two replacement image-contract tests above are what CI must prove"*
  is now falsified: CI ran them, they passed, and the functional path is still red.

Read as a report on `f3c847d6`, DONE.md asserts completion of work that was not done and
is silent on the one item that was. Nothing in it describes `persist-credentials`.

I left DONE.md **unmodified on purpose**: it is the artifact under review and rewriting it
would destroy the evidence of the mis-claim. This review is the correction of record.

---

## Finding 3 (non-blocking) — item 2 is correct; two loose ends

The change itself is right and complete for its file:

```yaml
- uses: actions/checkout@v4
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    fetch-depth: 0
    persist-credentials: false
```

Verified by parsing the YAML: the `agent` job has exactly **one** checkout (now covered),
and the `publish` job has **none**, so there is no second site to fix. I checked the
agent job for anything that needs the credential:

* `scripts/eumemic_bot_review.py::_pin_checkout` may run `git fetch --no-tags --quiet
  origin <base_sha>` (line 308) and `_die`s if it fails. `eumemic/aios` is **public**
  (`gh repo view` → `"visibility":"PUBLIC"`), and `fetch-depth: 0` already brings the base
  history, so the anonymous fallback still works. **No regression.** It would break if the
  repo were ever made private — worth a comment at the checkout, not a blocker.
* No push/tag/remote write anywhere in the agent job.

Two loose ends:

1. `_drop_persisted_git_credentials()` (line 311) is now redundant — defensible as
   defence-in-depth, but its docstring (*"`_pin_checkout` is the only thing that needs the
   credential, so it goes as soon as that returns"*) now describes a state that no longer
   occurs. Neither the docstring nor the workflow records that the two mechanisms overlap.
2. The commit message is a bare subject line: no body and **no `Co-Authored-By` trailer**,
   against CLAUDE.md's *"Conventional commits … Substantial commit bodies"* and
   *"Co-Authored-By trailer on AI-authored commits"*. I did not amend it — rewriting the
   maker's commit would move a tip the review is pinned to.

---

## Item 3 — rebase: not needed (verified)

```
$ git rev-list --left-right --count origin/master...HEAD
0       35          # 0 behind, 35 ahead
$ git merge-base origin/master HEAD
63337f26…           # == origin/master tip
```

`origin/master` (`63337f26`, 2026-09-13) is an ancestor of HEAD. Fetched fresh from
`origin` during this review. No rebase required — correctly a no-op, though DONE.md
does not mention having checked.

## Checks run

* `uv run pytest tests/unit/sandbox tests/unit/test_networking.py -q` → **681 passed**.
  (Green, and that is precisely the problem in Finding 1 — the unit layer stubs busybox.)
* Workflow YAML parsed and asserted structurally (checkout count, `with:` contents).
* CI logs pulled via `gh` for runs 34822598014, 34807834078, 34666903810.
* Full `mypy`/`ruff`/unit suite not re-run: no Python changed in this round.
* e2e(docker) **not** run here — no Docker daemon. No claim of e2e green is made.

## Blockers to clear before this can pass

1. Actually fix the credential-swap/DNAT resolution regression bisected to `9b246ab7`
   (Finding 1), and prove it with a green e2e(docker) run.
2. Move the DNS oracle into the context that broke — resolution from the netns-joining
   sidecar after the filter/nat OUTPUT flush — so a green suite means a working swap.
3. Replace DONE.md with a true report of this round (Finding 2).

## Final HEAD

* **Reviewed code tip: `f3c847d6`** — the last commit carrying code, unchanged by this
  review. All findings above are against that tree.
* **Final HEAD of `botpost2410jrev`: this review commit**, the single commit after
  `f3c847d6` (`git log --oneline f3c847d6..HEAD` shows exactly one). It is docs-only:
  `REVIEW.md` + the round's `TASK.md`. A commit cannot embed its own hash, so the sha
  is left to `git rev-parse HEAD` rather than recorded here incorrectly.
* Nothing pushed, no PR opened, branch unchanged otherwise.
