# Uncorrelated review — aios#2410 fixround `botpost2410k`

**Verdict: fail (blocking).**

Checker: claude-opus-5 on `botpost2410krev`. Implementer: gpt-5.6-sol on
`botpost2410k`. Reviewed tip: `d27c242d`, which is also the PR head
(`gh pr view 2410` → `headRefOid d27c242d…`, branch `gvisorgrn`).

---

## What this round delivered

Nothing. `d27c242d` is the **previous round's FAIL review commit** (`REVIEW.md`
+ `TASK.md`, 0 source files). The only working-tree content was an uncommitted
rewrite of `TASK.md` — the k-round brief itself. No product commit landed on
`botpost2410k`.

```
d27c242d docs(review): uncorrelated review of #2410 tip f3c847d6 — FAIL   <- HEAD
f3c847d6 fix(ci): disable persisted agent checkout credentials            <- all of round j
2e0225cc style(sandbox): ruff format setup.py
dca77114 fix(sandbox): reject embedded resolver IP in DNS answers
9b246ab7 fix(sandbox): resolve DNS by argument, not /etc/resolv.conf
e10f4e07 fix(sandbox): allow-list the machines runsc can run on
```

`src/aios/sandbox/setup.py` and `src/aios/sandbox/backends/docker.py` at this
tip are byte-identical to `2e0225cc`, the commit the failure was reported
against. So TASK item 1 was skipped for the **second consecutive round**, and
every one of the six e2e failures reproduces unchanged.

CI at this exact tip — run `34919782033`, job `104225124271`, `e2e (docker)`:

```
6 failed, 404 passed, 3 skipped in 277.61s
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_bash_env_var_placeholder_round_trip - KeyError: 'stdout'
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_unrestricted_dnat_only - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_run_env_var_placeholder.py::test_run_swap_fires_under_limited - assert 'HTTP_STATUS=200' in ''
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_limited - assert 0 == 1
FAILED tests/e2e/test_trigger_fire_env_var_swap.py::test_trigger_swap_fires_under_unrestricted_dnat_only - assert 0 == 1
FAILED tests/e2e/test_env_var_placeholder_materialized.py::test_placeholder_visible_in_container_secret_absent - KeyError: 'stdout'
```

---

## Finding 1 (blocking) — item 1 not attempted

No Python change. Nothing further to assess on the requested work itself.

## Finding 2 (blocking, root cause) — `resolve_ipv4` lost the `/etc/hosts` half, so the proxy alias `aios-worker` no longer resolves

This is the defect `9b246ab7` introduced. It is not a parse bug, not a missing
applet, and not the nat flush. It is that **`aios-worker` is not a DNS name in
half of the sanctioned deployments** — and `busybox nslookup` can only speak
DNS.

`src/aios/sandbox/network.py:1-6` states the contract in-tree:

> Two resolution paths share one hostname (``aios-worker``): Docker's embedded
> DNS when the worker is on the sandbox network, ``/etc/hosts`` populated by
> ``--add-host`` when the worker runs on the host.

The second path is selected at `spec.py:1238`
(`host_gateway_alias=None if is_running_in_container() else WORKER_NETWORK_ALIAS`)
and emitted at `backends/docker.py:393-394` as
`--add-host aios-worker:host-gateway`. In that shape **no container holds the
`aios-worker` alias** — `ensure_sandbox_network` logs `network_worker_on_host`
and returns without joining anything (`network.py:72-79`) — so Docker's
embedded resolver has no service record for the name and forwards the query
upstream, where it is NXDOMAIN. The name exists **only** in the container's
hosts file.

* `getent ahostsv4 aios-worker` → NSS `files` → hosts-file hit. Worked.
* `busybox nslookup aios-worker 127.0.0.11` → DNS only, never NSS. Returns
  nothing.

The e2e lane runs in exactly that shape: the `e2e` job in
`.github/workflows/code-validation.yml:413` has no `container:` key, so pytest
runs on the runner host, `/.dockerenv` is absent, and every sandbox is created
with `--add-host aios-worker:host-gateway`.

### How that produces the exact CI symptom

The generated script (rendered from this tip) guards the whole DNAT block on
the alias resolving:

```sh
"$IPT" -t nat -F OUTPUT
PROXY_IP=$(resolve_ipv4 aios-worker | head -n1)
if [ -n "$PROXY_IP" ]; then
  ips=$(resolve_ipv4 api.github.com)
  for ip in $ips; do "$IPT" -t nat -A OUTPUT -d "$ip" ... -j DNAT --to-destination "$PROXY_IP:8443"; done
else
  printf '%s\t%s\n' 'AIOS_EGRESS_SKIPPED api.github.com' 'proxy alias aios-worker has no IPv4 address'
fi
```

`PROXY_IP` is empty → the `else` arm runs → **no DNAT rule is emitted and the
script still exits 0**. That is precisely what CI reports: `network_lockdown_failed`
appears **zero** times in the job log (apply exits 0) while the read-back verify
fails with

```
secret-egress DNAT verification failed …: nat OUTPUT carries no DNAT rule after apply
```

Under Limited the apply still installs the per-host filter ACCEPTs and
`-P OUTPUT DROP` (those are outside the guard), so the verify's first assertion
passes and it is the `-j DNAT` assertion at `setup.py:926` that trips. The
error string is static, which is why that leg reports the misleading
`OUTPUT policy is not DROP after apply` for the same missing-DNAT cause.

The blast radius is wider than the DNAT. `registry.py:1069-1078` passes
`extra_host_ports` as `(WORKER_NETWORK_ALIAS, tool_broker.port)` and
`(WORKER_NETWORK_ALIAS, git_proxy.port)`, emitted as
`for ip in $(resolve_ipv4 aios-worker); do … -j ACCEPT; done`. With the alias
unresolvable those loops are empty too, so a Limited sandbox that *did* provision
would have **no ACCEPT for its own tool broker or git proxy**.

### The nat flush is not implicated

Worth stating, because it is the obvious suspect and it is wrong. At `e10f4e07`
the identical `"$IPT" -t nat -F OUTPUT` ran before resolution and the DNAT rules
landed anyway (the swap e2e were green) — so DNS to `127.0.0.11` survives the
flush. The flush is not the delta; the loss of the NSS `files` lookup is.

### Bisect, read off CI on both sides

| tip | run | `e2e (docker)` |
|---|---|---|
| `e10f4e07` | `34807834078` | 1 failed / 408 passed — only `test_image_layer_carries_the_embedded_dns_resolver`; swap family **green** |
| `2e0225cc` | `34822598014` | 6 failed / 404 passed — swap family **red** |
| `d27c242d` | `34919782033` | 6 failed / 404 passed — identical set |

`dca77114` has no completed run (`34822361371` cancelled), so the window is
`9b246ab7`+`dca77114`; `2e0225cc` is a pure `ruff format` commit. `9b246ab7` is
the commit that replaced `getent` with `busybox nslookup`.

### Why every green check missed it

* `tests/unit/test_networking.py` and
  `tests/unit/sandbox/test_sandbox_dns_resolution.py` drive the real emitted
  `resolve_ipv4` against a **stub busybox**. A stub has no notion of NSS, so no
  unit test can observe a hosts-file-only name. 681 passed locally here.
* `tests/e2e/test_sandbox_image_contract.py::test_busybox_nslookup_answers_from_the_embedded_dns`
  resolves a **network alias on a user-defined network** — a name the embedded
  DNS genuinely serves. It passed in the failing run. TASK item 2 framed this as
  a sidecar/netns-context gap; the actual gap is the **resolution mechanism**:
  the oracle exercises the DNS-served name and never the `--add-host` name that
  is the one that broke. Moving it into the sidecar alone would not have caught
  this unless it also resolves `aios-worker` in a worker-on-host shape.

### What a fix has to do (not landed here — see "No code changed")

Two shapes, and the choice between them is a design decision that belongs to
the implementer, not the checker:

1. **Restore an NSS/hosts-file lookup in `resolve_ipv4`** (hosts file first,
   then `busybox nslookup`). Smallest diff and an exact return to the
   `e10f4e07`-green behaviour. Two caveats that must be stated rather than
   discovered later: the shared `/etc/hosts` is **tenant-writable** (the sidecar
   runs `--network container:<id>`, so moby bind-mounts the target's hosts
   file), which makes the Limited allow-list tenant-influenceable — pre-existing
   at `e10f4e07`, but squarely the "tenant-poisonable resolver" this PR's own
   design notes reject; and it does **not** fix runsc, whose chroot into the
   read-only operator root reads *that* image's `/etc/hosts` (no entry).
2. **Stop resolving operator infrastructure inside the sandbox netns.** The
   worker already knows where its own proxy and broker are; passing an IP into
   `dnat_target` / `extra_host_ports` instead of the alias removes the
   tenant-influenceable input entirely and fixes both runtimes. Larger, and it
   needs a decision on how the worker learns the `host-gateway` address.

## Finding 3 (blocking) — `DONE.md` is still the previous round's report

It opens `Branch botpost2410i, on top of e10f4e07`, claims "Two items were asked
for. Both are done", and describes work already present in the base. TASK item 3
asked for it to be replaced with a true report of **this** round; it is
unmodified. Left that way deliberately, as in round j: the checker ghost-writing
the implementer's report would launder the miss and destroy the audit trail.

## Item 4 — `persist-credentials: false` — **pass**

`.github/workflows/eumemic-bot-review.yml:37-41` still carries it on the agent
job's checkout, which is the only checkout in that workflow; the `publish` job
has none. Verified correct and complete in round j; unchanged here.

## Item 5 — rebase — **no action needed**

`origin/master` (`63337f26`) is an ancestor of HEAD. `git rev-list --left-right
--count origin/master...HEAD` → `0 36`.

---

## Local checks at this tip

`uv run ruff check src tests` and `uv run ruff format --check src tests` clean
(1098 files). `uv run pytest tests/unit/test_networking.py tests/unit/sandbox -q`
→ 681 passed. CI's `unit`, `lint`, `integration`, `connectors` and
`e2e (non-docker)` are all SUCCESS at this tip; `e2e (docker)` is the only red
check. Docker is not available in this worktree, so no e2e was run here and none
is claimed — CI is the oracle throughout.

## No code changed by this review

The root cause above is established from CI and from the repository's own
documented two-path contract, but neither candidate fix can be verified without
a Docker daemon, both touch the security-load-bearing egress path, and one
re-lands an input this PR's design notes explicitly reject. Landing an
unverifiable change to the fail-closed lockdown as a *checker* is the wrong
trade against handing the implementer a pinned, reproducible diagnosis. Per the
review brief, this is recorded as a blocking finding rather than a rewrite.

**Final HEAD: `d27c242d` + this review commit (see git log).**

Refs #2410
