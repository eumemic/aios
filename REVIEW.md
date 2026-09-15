# Uncorrelated review — aios#2410 fixround `botpost2410l`

**Verdict: FAIL (blocking).**

| | |
|---|---|
| Reviewed tip | `e07df12520521b1dcca7c711c6a48f4d20e530ae` |
| Review branch | `botpost2410lrev` (this worktree; not pushed) |
| Implementer | gpt-5.6-sol on `botpost2410l` |
| Checker | claude-opus-5 on `botpost2410lrev` |
| `origin/gvisorgrn` | `e07df125` — identical to the reviewed tip |
| `origin/master` | `63337f26` — an ancestor of the tip |

The round delivered nothing. The implementer's Codex session aborted at model
capacity before it ran, so `botpost2410l`'s HEAD is still round k's FAIL review
commit. Item 1 — the actual credential-swap/DNAT fix — has now been skipped for
three consecutive rounds (j, k, l), and the six e2e failures reproduce unchanged
at this exact tip.

This review re-confirms the pinned root cause against the source, and adds the
two facts the next round needs before it writes the fix: **which** `/etc/hosts`
each runtime's lockdown context actually reads, and why that makes "hosts first"
correct for the red lane but silently inert on the runsc lane.

---

## Findings

### 1. [BLOCKING] No product change. The tip is a review commit.

```
$ git diff --stat d27c242d..HEAD
 REVIEW.md | 342 ++++++++++++++++++-------------------
 TASK.md   |  25 ++--
```

`git log` on `botpost2410l` ends at `e07df125` *docs(review): … — FAIL*, which is
this round's **input**, not its output. The only working-tree change was the
uncommitted TASK.md rewrite (the round-l brief itself, authored by the shepherd).
`src/aios/sandbox/setup.py` and `src/aios/sandbox/backends/docker.py` are
byte-identical to `2e0225cc`, the commit the failure was first reported against.

TASK items 1 and 2 are untouched; item 3 (true DONE.md) is untouched; item 4
(rebase) needed no action; item 5 (do not push) held.

### 2. [BLOCKING] `resolve_ipv4` is still DNS-only.

`src/aios/sandbox/setup.py:365-371`, unchanged since `9b246ab7`:

```python
_RESOLVE_IPV4_FN = (
    'resolve_ipv4() { busybox nslookup "$1" ' + _EMBEDDED_DNS_ADDRESS + " 2>/dev/null"
    ...
```

There is no NSS path, no `/etc/hosts` read, and no fallback. A hostname that
exists **only** as an `--add-host` entry resolves to nothing. `PROXY_IP` at
`setup.py:460` is then empty, the `if [ -n "$PROXY_IP" ]` guard at `:461` skips
the whole DNAT block, apply still exits 0, and the read-back at `:1141` raises

> `nat OUTPUT carries no DNAT rule after apply; refusing to run an
> env-var-credentialed sandbox whose secret-swap DNAT is unverified`

which is exactly the error CI reports. The root cause pinned in round k's review
survives source inspection at this tip.

### 3. [BLOCKING] CI is red at this tip, with the same six failures.

Run `34921131742` (Code Validation, head `e07df125`), job `104229330589`
*e2e (docker)* — the newer run than the `34919782033` cited in the brief, and on
this tip rather than `d27c242d`:

```
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_bash_env_var_placeholder_round_trip - KeyError: 'stdout'
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_unrestricted_dnat_only - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_limited - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_limited - assert 0 == 1
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_unrestricted_dnat_only - assert 0 == 1
FAILED tests/e2e/test_env_var_placeholder_materialized.py::test_placeholder_visible_in_container_secret_absent - KeyError: 'stdout'
```

with the provisioning error above logged upstream of every one of them. `e2e
(docker)` is the only failing job in the run.

### 4. [BLOCKING] DONE.md is the stale `botpost2410i` report.

Its first line:

> `Branch `botpost2410i`, on top of `e10f4e07` … Two items were asked for. Both are done.`

Beyond being the wrong round, it is now **affirmatively misleading**: §1 "The
fix" presents the busybox-nslookup resolver as correct and fail-closed, with a
seven-item evidence chain, and says nothing about the regression it caused —
because it was written before CI ran it. Two live code comments cite it as the
authority for that design (`setup.py:329-330`, `docker.py:1342`), so the stale
report is load-bearing documentation for the defect it caused.

I have deliberately **not** rewritten DONE.md. A checker authoring the maker's
report for a round in which no work happened would destroy the only signal that
the round produced nothing.

### 5. [BLOCKING] The DNS oracles are structurally incapable of catching this.

TASK item 2 asked for them to move into the sidecar-after-flush context. They
did not move, and inspection shows why that matters more than "wrong context" —
both oracles resolve a name that **can never exhibit the bug**:

* `tests/e2e/test_sandbox_image_contract.py::test_busybox_nslookup_answers_from_the_embedded_dns`
  (`:243-288`) creates a throwaway user-defined network and resolves the
  container's **own network alias** — a name Docker's embedded DNS serves by
  construction. The failing name is one embedded DNS has never heard of.
* `tests/unit/sandbox/test_sandbox_dns_resolution.py` runs the real
  `_RESOLVE_IPV4_FN` against a **stub busybox** (`_BUSYBOX_STUB`, `:107-120`)
  that answers everything except `missing.example.com`. A hosts-only name is not
  representable in that stub.

Both were green through the entire regression. Any fix must land an oracle keyed
on *a name present only in `/etc/hosts`*, or the next resolver change will break
the same way.

### 6. Rebase: not needed. (TASK item 4 — verified, no action.)

`git merge-base --is-ancestor origin/master HEAD` succeeds (`origin/master`
`63337f26`), and the tip already equals `origin/gvisorgrn`. The PR branch stays
`gvisorgrn`.

### 7. Local checks on the unchanged tree.

`uv run pytest tests/unit/sandbox/test_sandbox_dns_resolution.py
tests/unit/test_networking.py -q` → **129 passed**. This is a statement about
the tree, not about the fix: the unit suite was green throughout the regression
(finding 5). Docker is absent in this environment (`docker: No such file or
directory`), so no e2e was run here and **no e2e-green claim is made** — CI is
the oracle.

---

## For the next round: which `/etc/hosts` each context reads

The brief prescribes "hosts first, then busybox nslookup". That is right for the
red lane, but the two runtimes do **not** see the same hosts file, and the fix
must say so or it will be assumed to cover both.

**The red lane is runc.** `code-validation.yml`'s `e2e (docker)` step sets no
`AIOS_SANDBOX_RUNTIME`, and `config.py:349` defaults `sandbox_runtime=None`, so
`run_netns_sidecar` takes the `else` branch at `docker.py:1369-1381`:
`docker run --rm --network container:<id>`. moby copies `HostsPath` and
`ResolvConfPath` from the joined container for `container:` network mode, so the
sidecar reads the **target sandbox's** `/etc/hosts` — which carries the
`--add-host aios-worker:host-gateway` entry from `docker.py:394`. That is why
`getent ahostsv4` was green at `e10f4e07` and why a hosts read fixes this lane.
(Stated from moby's documented behaviour; **not** re-verified here — no Docker in
this environment. It is the load-bearing assumption of the whole fix and the
implementer should confirm it empirically before relying on it. The same
question puts a question mark on `setup.py:320-322`'s claim that the sidecar
"inherits the IMAGE's" resolv.conf, which is part of the stated justification
for the `9b246ab7` redesign.)

**The runsc lane reads a different file.** `docker.py:1350-1364` `chroot`s into
`_RUNSC_OPERATOR_ROOT` before the script runs, so post-chroot `/etc/hosts` is the
**operator image's** — exactly as that method's own docstring already says about
`/etc/resolv.conf` (`:1335-1339`). It has no `aios-worker` entry, so hosts-first
resolves nothing there and `PROXY_IP` stays empty under runsc whenever the worker
runs on the host. Bounded, but must be stated:

* production is unaffected — the worker is containerized
  (`spec.py:1238` sets `host_gateway_alias=None` when
  `is_running_in_container()`), joins the network, and embedded DNS serves the
  alias on both runtimes;
* `gvisor-validation` is `workflow_dispatch` + weekly cron and has failed on
  `master` for five consecutive weeks, so it will not surface this.

Closing it properly means giving the script the address rather than a name for
the proxy alias (resolve worker-side, inject the literal), which is the brief's
shape 2 and should not be smuggled in under shape 1 without the shepherd's call.

**Two constraints the fix must respect.**

1. `tests/unit/sandbox/test_sandbox_dns_resolution.py::test_no_script_touches_resolv_conf_or_getent`
   pins `"getent" not in script`. The hosts lookup must therefore be a direct
   parse of `/etc/hosts` (awk — already in the operator shadow set), not a
   restored `getent`. Restoring `getent` would also restore the
   `/etc/resolv.conf` dependence that `9b246ab7` existed to remove.
2. Order is load-bearing and so is the tenant-writability caveat. On runc the
   file read is the tenant container's own `/etc/hosts`, writable by root inside
   the sandbox. A tenant that poisons it chooses `$PROXY_IP` and the learned
   addresses that the per-host ACCEPT/DNAT rules are built from — and the egress
   **refresh** path (`setup.py:705-725`) re-resolves *while the tenant is live*,
   so this is not only a provision-time window. This is a **restoration** of
   `e10f4e07` behaviour, not a new hole (`getent` read the same file through
   NSS), but it belongs in the comment block, and whether the hosts read should
   be narrowed to the proxy alias alone is a design call for the shepherd, not
   something to decide inside the fix.

---

## Why this review lands no fix

Per the review brief: when the implementer misses the work entirely, document it
as blocking rather than expanding into a rewrite unless a small correct fix is
clear and in-scope. It is not. What is missing is the round's **entire** scope —
product change, hosts-only oracle, and DONE.md — and the fix carries a tenant-
writability tradeoff and a runsc gap that want the shepherd's sign-off, not a
checker's unilateral commit. Maker ≠ checker holds: writing it here would leave
the result unreviewed.

**Recommendation:** re-run `botpost2410l` on a model with capacity, with findings
5 and the two constraints above folded into the brief.

**Final HEAD:** the product tip is unchanged at
`e07df12520521b1dcca7c711c6a48f4d20e530ae` — no source file was touched by this
round or by this review. `botpost2410lrev` adds exactly one commit on top of it,
this document (`git rev-parse botpost2410lrev`). Not pushed, no PR.
