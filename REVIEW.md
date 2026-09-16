# Uncorrelated review — aios#2410 tip `cfaba589`

- **Round**: `botpost2410q` (implementer agent `botpost2410r`, grok-4.6, worktree `aios-botpost2410q`)
- **Checker**: claude-opus-5, worktree `aios-botpost2410rrev`, branch `botpost2410rrev` (maker ≠ checker)
- **Tip reviewed**: `cfaba589` "fix(ci): keep the review proxy key out of the coding-agent harness"
- **Verdict**: **PASS**

State: `origin/master 7df8b5d8` is an ancestor of HEAD; `origin/gvisorgrn = aeccbeb1`,
so the tip is one product commit ahead and **unpushed**, as TASK says. Not pushed,
not merged, no PR opened by this review.

The tip touches four files and no `src/`:

```
.github/workflows/eumemic-bot-review.yml |  56 ++++-
docs/eumemic-bot-review.md               |  12 +-
scripts/eumemic_bot_review.py            | 351 +++++++++++++++++++++++++++---
tests/unit/test_eumemic_bot_review.py    | 271 ++++++++++++++++++----
```

---

## 1. High — the routed proxy key is not reusable by the harness (TASK §1) — **closed**

The design is the second arm of the finding's prescribed remedy ("run the model
without a reusable secret and proxy via a separately controlled process"), and
it is implemented end to end:

1. **Staging out of the agent step.** `secrets.*_PROXY_API_KEY` is expanded only
   in the `proxykey` step, which `install -m 600`s a file under `RUNNER_TEMP`
   and exits. The `agent` step's env holds `REVIEW_PROXY_KEY_FILE` — a *path* —
   and nothing else credential-shaped (workflow L72-L114).
2. **Read-then-unlink before the child exists.** `_proxy_key` reads and
   `unlink`s the file; it is called from `_broker_for` → `run_agent`, i.e.
   strictly before `_run_harness`'s `subprocess.run`. Asserted for real in
   `test_run_agent_unlinks_the_staged_key_and_does_not_hand_it_to_the_harness`,
   which checks `not staged.exists()` *inside* the fake `subprocess.run`.
3. **Seal before the key is read.** `run_agent_phase` orders
   `_pin_checkout` → `_drop_persisted_git_credentials` → `_seal_process` →
   `run_agent`; `test_main_scrubs_…` pins that order.
4. **The child gets only a loopback token.** `_agent_command` strips
   `_STRIPPED_ENV` and sets `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / pi's
   `models.json` `apiKey` to `broker.token`, with the base URL on `127.0.0.1`.

I did not take the tests' word for any of this. Independent verification:

**(a) The seal really closes `/proc` to the child.** Ran the real
`_seal_process()` in a Python process holding `SEKRIT=top-secret-value`, then
had a `bash` child try to read it:

```
child sees SEKRIT count / perms: 0
dr-x------ 2 root root 5 /proc/2172882/fd
self-read of own fd dir ok: True
```

`/proc/<launcher>` is reassigned to root, the child's read fails, and the
launcher can still read its own `/proc/self/fd` (which `subprocess` needs).
`_PR_SET_DUMPABLE = 4` is the correct `prctl.h` value.

**(b) The broker substitutes the credential and never forwards the token.**
Stood the real `_ProxyBroker` in front of a fake upstream:

```
json relay: 200 b'{"ok":true}' | upstream saw auth: Bearer REAL-KEY path: /v1/responses
sse headers: {... 'Content-Type': 'text/event-stream', 'Connection': 'close'}
  chunk @ 0.0 b'data: tok0\n\n'     <- streams incrementally, not buffered
  chunk @ 0.3 b'data: tok1\n\n'
  chunk @ 0.6 b'data: tok2\n\n'
upstream 429 relayed: 429 b'{"error":"upstream says no"}'
bad token: 401 {"error": {"message": "broker token is missing or wrong", ...}}
x-api-key auth accepted: 200
```

The `read1` choice is load-bearing and correct — SSE arrives token-by-token
rather than at completion. Upstream 4xx/5xx reach the harness with the real
status and body (so its own backoff still works), and both auth header shapes
(`Authorization: Bearer` for oai/xai, `x-api-key` for ant) authorize.

**(c) `_NoRedirect` holds.** A 302 from upstream is *not* followed by the
broker (which holds the real key); the `Location` is relayed to the harness,
which can only replay it with the worthless loopback token. Verified: the real
key never reached the redirect target.

**(d) Full end-to-end with the real `codex` binary**, default model route,
against the broker in front of a fake `/v1/responses` upstream:

```
provider: eumemic_oai_proxy   sandbox: danger-full-access   rc: 0
upstream hits: [{"path": "/v1/responses", "auth": "Bearer REAL-UPSTREAM-KEY", "len": 43284}]
token leaked upstream: False
artifact: ### Code review\n\nNo findings.
```

This is the strongest evidence available short of CI: codex accepts the
`-c model_provider=` / `model_providers.…` table with an `http://127.0.0.1`
`base_url`, picks up the token via `env_key="OPENAI_API_KEY"`, the broker swaps
it for the upstream key, and `--output-last-message` lands the artifact.

`--sandbox danger-full-access` is retained, which TASK explicitly allows.

## 2. Prior tip preserved (TASK §2) — **verified**

`git diff --stat 59b62e3b..HEAD -- src/ tests/unit/sandbox/` is **empty**: the
tip touches no product code, so the hosts-first / operator-controlled refresh
High and the Medium runsc `host-gateway` bake are byte-identical to the tip I
passed last round. Their focused tests still pass:

```
tests/unit/sandbox/{test_sandbox_dns_resolution,test_sandbox_network,
                    test_egress_refresh,test_runsc_operator_shadow}.py
→ 86 passed in 3.38s
```

## 3. Focused checks (TASK §3) — **green**

```
uv run pytest tests/unit/test_eumemic_bot_review.py -q   → 38 passed in 8.28s
uv run ruff check   scripts/… tests/unit/…              → All checks passed!
uv run ruff format --check scripts/… tests/unit/…       → 2 files already formatted
uv run mypy tests/unit/test_eumemic_bot_review.py scripts/eumemic_bot_review.py
                                                        → Success: no issues found
```

No full suite, no `-n`, per the ops constraint. The new tests are substantive
rather than shape-only: they stand a real broker up and assert on
`broker.base_url` / `broker.token`, and assert the upstream secret string is
absent from `env.values()` and from pi's generated `models.json`.

---

## Findings (all non-blocking)

### M1 — the seal's guarantee is real but bounded, and the prose does not say so

`_seal_process`'s docstring says `cat /proc/$PPID/environ` and reading our
memory "both fail for the agent", and `docs/eumemic-bot-review.md` says `/proc`
and memory "are closed to the child". Both are true for an *unprivileged*
child. On `ubuntu-latest`, the `runner` user has passwordless `sudo`, and the
agent runs as that user with `danger-full-access` — so a prompt-injected agent
can `sudo` and read the launcher's memory (where `_ProxyBroker._credential`
lives) regardless of the dumpable flag.

This is not a FAIL. The finding's remedy is implemented faithfully, TASK
explicitly permits `danger-full-access`, and no in-process scheme survives a
root-capable attacker. But the absolute phrasing invites a later reader to
treat the seal as a hard boundary. Worth (a) qualifying both comments to
"closed to an unprivileged child", and (b) considering the cheap hardening that
would actually close it: exec the harness under a different uid (`sudo -u`
/ `setpriv`), which also removes the agent's own `sudo`. Note the credit that
is due — `_STRIPPED_ENV`'s comment already states the `/proc/$PPID/environ`
limitation honestly; it is only the seal's own prose that overclaims.

Related and inherent, worth one line in the docs: while the launcher lives the
agent can drive the broker as an open relay to the proxy origin (arbitrary path
under that origin, real key stamped on). That is bounded spend inside the
900 s window, not a reusable secret, and the agent legitimately has model
access anyway — but "exfiltrate **or spend**" was the original wording, and only
the exfiltrate half is fully closed.

### M2 — a chunked request body is silently dropped

`_BrokerHandler._forward` reads the body from `Content-Length` only:

```python
length = int(self.headers.get("Content-Length") or 0)
body = self.rfile.read(length) if length else None
```

A harness that sends `Transfer-Encoding: chunked` gets its body replaced by an
empty one, with no error and no log. Demonstrated:

```
chunked req -> 200 b'{"ok":1}' | upstream body seen: b''
```

Likelihood is low in practice — codex/reqwest, Claude Code/undici and pi all
set `Content-Length` for JSON bodies, and the codex smoke above sent a 43 KB
framed body — but the failure mode is a mangled prompt or a confusing upstream
400 rather than a visible fault, which is exactly what "fail hard, no
fallbacks" exists to prevent. Either de-chunk `self.rfile`, or refuse with
`411` via the existing `_refuse` when `Transfer-Encoding` is present.

### L1 — unsupported verbs escape the broker's error envelope

Only `do_GET` / `do_POST` / `do_DELETE` are defined, so `PUT` / `PATCH` /
`HEAD` / `OPTIONS` fall through to `BaseHTTPRequestHandler`'s HTML 501:

```
PATCH -> 501 b'<!DOCTYPE HTML>\n<html lang="en">…'
```

No current route needs them, so this is cosmetic — but an HTML body where the
harness expects JSON is a worse diagnostic than `_refuse(501, …)` would be.

### L2 — `_seal_process` itself is never exercised by a test

Every test monkeypatches it (`test_main_scrubs_…` asserts only call ordering).
Nothing would catch a wrong `_PR_SET_DUMPABLE`, a botched `ctypes` signature,
or the errno handling. The probe in §1(a) is ~10 lines and runs in a
subprocess, so it is cheap and does not seal the pytest process; given the
broker's whole credential separation rests on that one syscall, it is worth
pinning.

### L3 — DONE.md is stale against this tip

`DONE.md` describes only the previous round's work (operator-table refresh
hosts, runsc `host-gateway`) and says nothing about the proxy-key isolation
that *is* this tip. TASK §4 anticipated this. Shepherd should refresh it before
push so the PR description matches `cfaba589`.

### Inconclusive, not a finding

CLI-level smokes of the `claude` and `pi` routes timed out in this sandbox with
zero upstream hits. The cause is local — this repo's `SessionStart`/`SessionEnd`
hooks (`uv run aios dev …`) hang under a captured-output child, and my pi
invocation passed the prompt positionally rather than on stdin. The argv for
both routes is **unchanged** by this tip (only the env *values* moved from the
real key to the broker token), and the broker-level probes above cover both the
`x-api-key` and `Bearer` paths, so I do not read this as a regression.

---

## Verdict

**PASS.** The High is closed the way the finding asked: the reusable proxy key
never enters the agent step's environment, is unlinked before the harness
exists, and is replaced for the child by a per-run loopback token that dies
with the launcher — verified with the real codex binary, not just with mocks.
The prior tip's product code is untouched and still green. The three Low and
two Medium items above are follow-ups, not blockers; M1's prose fix and L3's
DONE.md refresh are the two I would land with the Shepherd's leftover apply.
