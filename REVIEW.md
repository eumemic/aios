# Uncorrelated review — aios#2410 tip `e685033b`

- **Round**: `botpost2410t` (implementer grok-4.6, worktree `aios-botpost2410t`)
- **Checker**: claude-opus-5, worktree `aios-botpost2410trev`, branch `botpost2410trev` (maker ≠ checker)
- **Tip reviewed**: `e685033b` "fix(ci): let the dropped review agent prove the diff"
- **Verdict**: **FAIL** — one line short. The two fixes are the right shapes and I proved
  they work, but as committed the dropped agent cannot reach its own home directory, so
  the launcher still cannot complete a single review. F1 is not closed.

State: `origin/gvisorgrn = 47e0f73e`; HEAD is one product commit ahead and **unpushed**.
Not pushed, not merged, no PR opened. `TASK.md` is dirty (Shepherd's round file); untouched.
The tip touches four files and no `src/`: `DONE.md`, `docs/eumemic-bot-review.md`,
`scripts/eumemic_bot_review.py`, `tests/unit/test_eumemic_bot_review.py`.

## Method

The prior round's tests all monkeypatched the boundary, which is why both prior blockers
shipped green. This review drove the **real** `_run_harness` through a **real**
`sudo setpriv` drop to a **real** unprivileged `eumemic-review` user against a **real**
git repo owned by another uid, on a box with passwordless sudo (the `ubuntu-latest`
shape). Every finding below is a reproduction, not a reading.

## What passes

**High — privilege boundary: still closed.** Unchanged in substance and correct.
`_require_agent_user` (`scripts/eumemic_bot_review.py:342`) fails closed on uid 0, on a
shared euid, and on a user that can sudo (`_agent_user_may_sudo` returns "refuse" when it
cannot prove the contrary). `_setpriv_argv` (`:378`) is a pure extraction of the previous
inline argv — `--reuid/--regid/--clear-groups/--no-new-privs/--inh-caps=-all` — now shared
with the new self-check, no behavioural drift. `_run_harness` has no un-dropped path. The
reusable key stays in the launcher's `_ProxyBroker`; the spec handed across the boundary
carries only the loopback token. Verified live: `HARNESS: uid=998` (the agent user), while
the launcher stayed uid 1000. Broker+seal is not being passed off as the boundary anywhere
in the prose. **Not a FAIL on the High.**

**Medium — artifact gate: still closed.** `require_inspection_evidence` (`:564`) is
untouched, still called both inside `_run_harness` (`:963`) and again in `run_agent_phase`
before the write, and still accepts only the full 64-hex digest. Verified live: a
heading-only harness exits **3** with the banner and no traceback, and writes nothing:

```
FATAL: NO EVIDENCE OF INSPECTION — refusing to publish a verdict: ... no well-formed
`<!-- inspected: lines=<N> sha256=<HEX> -->` line ...
::error title=NO EVIDENCE OF INSPECTION...
EXIT=3
```

**Keep list: intact.** The diff touches no `src/` — hosts-first `resolve_ipv4` /
`build_resolve_ipv4_fn` with `ResolveScope` and operator-controlled `operator_hosts`
(`src/aios/sandbox/setup.py:565`, `:532`), the runsc gateway work, and
proxy-key-out-of-harness-env (`REVIEW_PROXY_KEY_FILE` staged in a separate step, read and
unlinked in `_proxy_key`, plus `_STRIPPED_ENV`) are unchanged.

**Prior non-blocking note 1 is fixed.** `harness = command[0]` is captured before the wrap
(`:925`), so diagnostics name codex/claude/pi again instead of `sudo`.

**Checks run** (focused only, per ops constraint — no full suite, no `-n`):
- `uv run pytest tests/unit/test_eumemic_bot_review.py -q` → **64 passed**
- `uv run ruff check` / `ruff format --check` on the two changed Python files → clean
- `uv run mypy scripts/eumemic_bot_review.py` → clean

## Blocking finding

### B1 — the agent's home is unreachable: the launcher temp root is `0700` and launcher-owned, so the dropped uid cannot traverse into `agent/`

The F2 fix correctly narrows the chown to `agent/` and leaves the `TemporaryDirectory`
root to the launcher (`:919-922`, `:518-519`). But `mkdtemp` always creates that root
**`0700`**, and nothing widens it. The agent subdirectory is then `0700` owned by
`eumemic-review` — inside a directory the agent has **no `x` bit on**. Unix permission
checks walk the path, so the dropped uid cannot open *anything* under `agent/`: not
`gitconfig`, not `harness-spec.json`, and it cannot create `last-message.md`.

Both halves of the round die on this, in order:

1. **`_verify_dropped_diff` (`:444`) fails** because `GIT_CONFIG_GLOBAL` (`:455`) is
   unreadable, so `safe.directory` never applies and git refuses the runner-owned
   checkout. Driving the real `_run_harness`:

   ```
   DRIVER: launcher evidence lines=15 sha=d0d90a12...
   FATAL: dropped user eumemic-review cannot git diff daefd674...37807905:
   warning: Not a git repository. Use --no-index to compare two paths outside a working tree
   DRIVER: SystemExit(1)
   ```

   (git 2.47 renders the cross-uid refusal as "Not a git repository" here rather than
   "dubious ownership"; same check, same cause.)

2. **The harness never starts even if the check is bypassed** — the trampoline cannot read
   its own spec:

   ```
   PermissionError: [Errno 13] Permission denied:
     '/tmp/eumemic-probe2-o1178s0f/agent/harness-spec.json'
   harness rc=1
   ```

So **F1 is not closed**. The failure mode changed — an early, loud FATAL exit 1 instead of
a NO_EVIDENCE exit 3 — but the outcome TASK cares about is the same: *every* review
produces no artifact. It is also not runner-specific and not race-dependent; `mkdtemp` is
`0700` everywhere.

The previous tip got traverse access by accident (`os.chmod(temp, 0o755)` on the root it
also chowned) — which is exactly what caused F2. This tip removed that line and moved the
`0700`/chown down to `agent/`, but never gave the agent a way *through* the root.

**Fix shape (one line, verified):** `os.chmod(root, 0o711)` immediately after
`root = Path(temp)` — traverse-only, still launcher-owned so teardown is unaffected, and
still not listable. With only that line added to a copy of the script, the same two drivers
give:

```
===== GOOD PATH =====
HARNESS: uid=998 cwd=/tmp/e2e/repo lines=15 sha=d0d90a1287ef
HARNESS: git rc=0 stderr=''
DRIVER: RETURNED ARTIFACT:
### Code review
...
<!-- inspected: lines=15 sha256=d0d90a1287ef4ae3d02bafead0327e06afd17e76d0188cdac1735826e043c35d -->
EXIT=0
===== HEADING-ONLY PATH =====
... NO EVIDENCE OF INSPECTION ...
EXIT=3
```

with a clean teardown and no leftover temp directory on either path. Everything else in the
round — the `safe.directory` gitconfig, the dropped-uid self-check, the narrowed chown, the
`sudo rm -rf` teardown, the artifact readback — is correct and works. This is the only
thing standing between the tip and a PASS. (Equivalent alternatives: put `agent/` under a
`0711` launcher-owned dir, or `sudo install -d -o <user> -m 0700` it somewhere already
traversable. `0755` on the *root* would also work but re-opens nothing, since the root is
never chowned — `0711` is the tighter choice.)

## Secondary finding

### B2 — `ignore_cleanup_errors=True` is not the belt the comment claims

`:917-919` states: *"ignore_cleanup_errors is belt: even if sudo-rm of the agent subdir
fails, PermissionError from TemporaryDirectory must not supersede SystemExit."* That is
false for CPython 3.13. `TemporaryDirectory._rmtree.onexc` (`/usr/lib/python3.13/tempfile.py:1086-1099`)
calls `_resetperms(path)` **before** any `ignore_errors` branch; `chmod` on a
foreign-owned directory raises `EPERM`, which propagates straight out of `__exit__`. The
`if repeated and path == name: if ignore_errors: return` early-out only covers the
top-level directory on a *second* pass, which is never reached. Reproduced against the real
code path:

```
File ".../tempfile.py", line 1141, in __exit__ -> cleanup -> _rmtree -> onexc -> _resetperms
PermissionError: [Errno 1] Operation not permitted: '/tmp/eumemic-probe-hvzwryol/agent'
```

— and the directory leaked. In the normal flow the `finally: _rmtree_maybe_foreign(agent_home)`
(`:966`) removes `agent/` first, so this does not fire; it only matters when the `sudo rm`
itself fails, which is precisely the case the comment says it covers. Not blocking on its
own, but it is the exact traceback shape TASK's live-CI evidence cites, so the claim should
not stand unqualified. Either make the teardown the real guarantee (`mkdtemp` +
`_rmtree_maybe_foreign(root)` in a `finally`, no `TemporaryDirectory`), or downgrade the
comment to say the sudo-rm is the only guarantee.

## Why the tests still did not catch it

Same gap as last round, unaddressed. Every new isolation test stubs the boundary:
`test_drop_chowns_only_the_path_it_is_given` monkeypatches `_sudo` (so no chown happens),
`test_verify_dropped_diff_*` monkeypatch `subprocess.run` (so no drop happens), and
`test_no_evidence_still_cleans_agent_home_and_keeps_exit_3` replaces
`_drop_into_agent_user` with identity. `test_drop_wraps_the_harness_with_setpriv_no_new_privs`
now asserts the gitconfig exists and names `cwd` — true, and still useless, because nothing
asserts the agent uid can *open* it. The assertions are about argv and file content; the
defect is about reachability, which no test touches.

A single test that does a real `setpriv` drop when a suitable non-sudo user exists (skip
otherwise) and asserts the dropped uid can `cat` its own `harness-spec.json` would have
caught both prior blockers and this one. Given the launcher now has `_verify_dropped_diff`,
an even cheaper version is to assert `_verify_dropped_diff` succeeds against a real
foreign-owned repo in that same skip-if-unavailable test.

## Non-blocking notes

1. **The self-check can silently no-op.** `run_agent`/`_run_harness` take `base_sha`/`head_sha`
   as defaulted `""` and gate on `if base_sha and head_sha and "setpriv" in command` (`:928`).
   A caller that forgets the kwargs skips the fail-closed check with no signal, and
   `"setpriv" in command` is a bare list-membership test on a string. Making the shas
   required parameters and gating on `_drop_into_agent_user` having actually wrapped
   (e.g. returning a flag) would keep the check from quietly disappearing.
2. **`_agent_gitconfig_text` does not escape git-config value syntax.** `#`, `;`, a
   trailing space, or a backslash in the workspace path would be mis-parsed. GitHub
   workspace paths are safe, so this is prose-level; writing the entry with
   `git config --file <path> --add safe.directory <dir>` would remove the class.
3. **Digest posture asymmetry.** `diff_evidence` runs git with the launcher's inherited
   env and global config; `_verify_dropped_diff` runs it under `env -i ... LANG=C` with a
   different `GIT_CONFIG_GLOBAL`. Any global-config difference that changes diff bytes
   (`diff.noprefix`, `core.abbrev`) would make the new check fail closed on a healthy
   runner. Low risk on ubuntu-latest, but the two digests should be computed under the
   same config posture.
4. **`_make_tree_readable` widens the agent's HOME, not just the artifact.**
   `chmod -R a+rX` (`:424`) now lands on the agent's home directory, so anything the
   harness wrote there becomes world-readable to every user on the runner. Correct for the
   artifact; worth scoping to `last-message.md`.

## Verdict

**FAIL.** High and Medium are intact and genuinely closed, the keep list is untouched, and
both F1 and F2 are attacked with the right shapes — `safe.directory` via `GIT_CONFIG_GLOBAL`
plus a dropped-uid self-check for F1, a narrowed chown plus `sudo rm` teardown for F2. But
the launcher's temp root is `0700` and launcher-owned, so the dropped uid cannot traverse
into the home it was just given: `GIT_CONFIG_GLOBAL` and `harness-spec.json` are both
unreadable and the harness never starts. F1 is therefore not closed — every review still
ends with no artifact, now as FATAL exit 1 rather than NO_EVIDENCE exit 3. Adding
`os.chmod(root, 0o711)` makes the whole round pass end to end, which I verified on both the
success and refusal paths.

## Leftover applied at open_pr (Shepherd)

Applied B1 on the implementer tip before push: `os.chmod(root, 0o711)` immediately after
`root = Path(temp)` in `_run_harness`, docs note on the `0711` traverse bit, and unit pin
`test_temp_root_gets_traverse_bit_before_drop`. Reviewer-verified shape; maker≠checker
kept (no second product invent beyond the named leftover).

