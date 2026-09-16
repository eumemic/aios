# Uncorrelated review — aios#2410 tip `84aa0759`

- **Round**: `botpost2410u` (implementer grok-4.6, worktree `aios-botpost2410u`)
- **Checker**: claude-opus-5, worktree `aios-botpost2410urev`, branch `botpost2410urev` (maker ≠ checker)
- **Tip reviewed**: `84aa075961fa370debec3d45e98195a086d21b19` "fix(ci): let dropped review uid enter a 0700 runner home"
- **Verdict**: **PASS** — the live FATAL is closed, and I reproduced both halves of the
  A/B on a real `setpriv` drop to a real unprivileged uid. F2, the High privilege
  boundary, the Medium digest gate and the keep list are all intact.

State: `origin/gvisorgrn = 5766de7d`; HEAD is one product commit ahead and **unpushed**.
Not pushed, not merged, no PR opened. `TASK.md` is dirty (Shepherd's round file) and
untouched. The tip touches four files and no `src/`: `DONE.md`,
`docs/eumemic-bot-review.md`, `scripts/eumemic_bot_review.py`,
`tests/unit/test_eumemic_bot_review.py`.

## Method

Same posture as the previous two rounds: the committed tests still stub the boundary, so
nothing in-tree can prove reachability. I drove the **real** `_run_harness` through a
**real** `sudo setpriv` drop to the **real** unprivileged `eumemic-review` user (uid 998,
no sudo) against a **real** git repo sitting under a **0700** `$HOME` — the ubuntu-latest
shape from the live Action. Every claim below is a reproduction, not a reading.

## F1 — live FATAL: closed (A/B proof)

`_ensure_dropped_uid_can_enter` (`scripts/eumemic_bot_review.py:441`) is called from
`_run_harness` (`:986`) on the dropped path, before `_verify_dropped_diff`. It adds
other-**execute** on every ancestor of the checkout and `a+rX`-equivalent bits on the
checkout itself.

The same driver, same repo, same 0700 `$HOME`, with the new function neutered vs. as
committed:

```
===== WITHOUT the new opener (fix neutered) =====
DRIVER launcher evidence: (19, 'eb5f3889...67b7')
FATAL: dropped user eumemic-review cannot git diff 8be94c73...b78ed901:
  fatal: cannot change to '/tmp/rev2410u/home/runner/work/aios/aios': Permission denied
DRIVER SystemExit: 1

===== WITH the opener (as committed) =====
HARNESS uid=998 cwd=/tmp/rev2410u/home/runner/work/aios/aios HOME=/tmp/eumemic-review-7ohpca52/agent
HARNESS git rc=0 stderr=
HARNESS lines=19 sha=eb5f38899a4cf17046a0c18463f56e9e6f27182076f623766d69fac3e57e67b7
DRIVER RETURNED ARTIFACT:
### Code review
...
<!-- inspected: lines=19 sha256=eb5f3889...67b7 -->
EXIT=0
```

The neutered run reproduces TASK's live error **verbatim** (`cannot change to '...':
Permission denied`); the committed code makes the dropped uid enter the checkout, read
`.git`, and emit a **non-empty digest that matches the launcher's**. The shape is also
diagnosis-independent: it opens *every* ancestor rather than betting on which component
was restrictive, so it closes the class, not just the observed instance.

Three properties I checked because they could have turned this fix into a different
failure:

- **It does not perturb the digest.** Files get `+r` only, never `+x`, so no tracked file
  flips `100644`→`100755` between `diff_evidence` (computed before the walk) and the
  agent's own hash. Verified with a tracked `0755` script and a tracked `0600` file in the
  tree: digest byte-identical before/after, `git status --porcelain` clean, `run.sh` still
  `-rwxr-xr-x`.
- **It does not open `$HOME` for reading.** `0700` → `0711`, traverse-only. Live, as the
  dropped uid: `LIST_HOME=denied`, `READ_PRIVATE=denied` (a `0600` sibling), while
  `ENTER_CHECKOUT=ok`.
- **It is not a cost.** The walk on the real aios checkout (fetch-depth-0, 68M `.git`) is
  605 dirs / 4940 files, **1** of which needs a chmod at all, at 0.11s. No sudo fallback
  fires in the CI shape, where the runner owns the whole tree.

## F2 — cleanup: still closed

`os.chmod(root, 0o711)` (`:978`, the leftover applied from last round) plus the chown
narrowed to `agent/` plus `_rmtree_maybe_foreign` in the `finally` (`:1023`). Across three
real dropped runs in this review — success, heading-only refusal, and the FATAL path —
**zero** `/tmp/eumemic-review-*` directories leaked and no `PermissionError` traceback
appeared. (The three leftovers on this box timestamp 19:18–19:21, i.e. the implementer's
own probes before the fix, not my runs at 20:05–20:07.) The NO_EVIDENCE exit is not
swallowed: exit **3** with the banner and the `::error` annotation, nothing written.

## High — privilege boundary: still closed

Untouched by this tip, and re-verified live rather than assumed. As the dropped uid:
`SUDO=denied`, `READ_OTHER_ENVIRON=denied`, harness `uid=998` while the launcher stayed
uid 1000. `_require_agent_user` (`:342`) still fails closed on uid 0, a shared euid, and a
user it cannot prove is outside `sudo`/`admin`/`wheel`. The reusable key stays in the
launcher's `_ProxyBroker`; the spec crossing the boundary carries only the loopback token;
`REVIEW_PROXY_KEY_FILE` is read and unlinked in `_proxy_key` **before** anything is
widened, so the staged key is gone from the filesystem by the time the agent can traverse
`RUNNER_TEMP`. Nothing in prose passes prctl+broker off as the boundary. The new walk adds
only `r`/`x`, never `w`, so the agent still cannot tamper with the workspace or the
artifact path.

## Medium — artifact gate: still closed

`require_inspection_evidence` (`:611`) is untouched, still called inside `_run_harness`
(`:1010`) and again in `run_agent_phase` before the write, still full-64-hex-only. Live
heading-only harness, through the real drop:

```
FATAL: NO EVIDENCE OF INSPECTION — refusing to publish a verdict: ... no well-formed
`<!-- inspected: lines=<N> sha256=<HEX> -->` line ...
DRIVER SystemExit: 3
```

Note the gate is now *reachable on its merits* for the first time: previously every run
died before the harness, so the digest channel had never actually accepted a real dropped
agent's evidence. It does now.

## Keep list — intact

The tip touches no `src/`. Hosts-first `resolve_ipv4` / `build_resolve_ipv4_fn` with
`ResolveScope` and operator-controlled `operator_hosts` (`src/aios/sandbox/setup.py:490`,
`:532`, `:567`), the runsc gateway bake (`.github/workflows/gvisor-validation.yml`), and
proxy-key-out-of-harness-env (`REVIEW_PROXY_KEY_FILE` staged in its own step, read and
unlinked, plus `_STRIPPED_ENV`) are all unchanged.

## Checks run

Focused only, per the ops constraint — no full suite, no `-n`:

- `uv run pytest tests/unit/test_eumemic_bot_review.py -q` → **69 passed**
- `uv run ruff check` / `ruff format --check` on the two changed Python files → clean
- `uv run mypy scripts/eumemic_bot_review.py` → clean

## Non-blocking notes

1. **The next dir of the same class is the agent home, not the checkout.** The opener is
   applied to `os.getcwd()` only. `agent_home` lives under `tempfile.gettempdir()`, which
   is reachable today purely because ubuntu-latest leaves `TMPDIR` unset and `/tmp` is
   `1777`. Set `TMPDIR` to anything `0700` (or point it at `$HOME`) and the dropped uid
   loses its own `gitconfig`/`harness-spec.json` again — the exact defect of round
   `botpost2410t`, in a new disguise. It would fail closed and loudly, so it is not a
   blocker, but calling `_ensure_dropped_uid_can_enter(root.parent)` (or asserting
   traversability of the temp root's ancestors) would retire the class rather than the
   instance.
2. **The walk widens `0600` files inside the checkout to world-readable.** Verified:
   a tracked `0600` `cfg.ini` became readable to the dropped agent. Harmless in this
   workflow — the only thing in `$GITHUB_WORKSPACE` before the agent step is
   `actions/checkout` output with `persist-credentials: false` — but it means any future
   step that stages a secret into the workspace hands it to an untrusted agent. Scoping
   the read-widening to `.git` plus `git ls-files` output would keep the property the
   digest needs without the sharp edge.
3. **Two mechanisms for one job.** `_make_tree_readable` (`:421`, one `sudo chmod -R
   a+rX`) and the new per-entry Python walk do the same thing by different means.
   Per CLAUDE.md's "compose, don't accrete", one of them should be the primitive.
4. **`_chmod_add` `_die`s when `stat` fails mid-walk** (`:434`), turning a benign
   file-vanished race into a FATAL. `os.walk` over a live tree can hand out entries that
   are already gone; skipping `FileNotFoundError` would be strictly better.
5. **The new unit test mutates directories outside `tmp_path`.** Because the opener walks
   to `/`, `test_ensure_dropped_uid_can_enter_opens_a_0700_home` left `/tmp/pytest-of-box`
   and `/tmp/pytest-of-box/pytest-17{2,3}` at `0711` on this box (earlier `pytest-171` is
   still `0700`). Harmless, but a unit test should not chmod its way up the filesystem —
   an explicit stop boundary on the ancestor walk would fix both this and note 1.
6. **Still no test that performs a real drop.** `test_run_agent_opens_the_checkout_before_
   the_dropped_diff` stubs `_drop_into_agent_user`, `_ensure_dropped_uid_can_enter` and
   `_verify_dropped_diff` and asserts call *order* — useful, and still not reachability.
   `test_ensure_dropped_uid_can_enter_opens_a_0700_home` is the first test in this family
   that asserts real modes, which is the right direction. A skip-if-unavailable test that
   does a real `setpriv` drop and asserts the dropped uid can `cat` its own
   `harness-spec.json` and reproduce the digest would have caught all three blockers of
   the last three rounds; this review has had to supply that out-of-tree every time.
7. Prior-round notes 1–3 stand: the self-check still gates on the stringly
   `"setpriv" in command` (`:986`) with `base_sha`/`head_sha` defaulted to `""`, so a
   caller that forgets the kwargs silently skips the fail-closed check;
   `_agent_gitconfig_text` still does not escape git-config value syntax; and
   `diff_evidence` (launcher env/global config) vs `_verify_dropped_diff`
   (`env -i … LANG=C`, different `GIT_CONFIG_GLOBAL`) still compute the digest under
   different config postures. All three matched live here.

## Verdict

**PASS.** This is the first tip in the series where a review can actually complete: the
dropped `eumemic-review` uid enters a `0700`-ancestor checkout, reproduces the launcher's
digest byte-for-byte, and the artifact comes back with exit 0 — while the heading-only
harness still exits 3 with no artifact, `$HOME` stays unlistable, the agent still cannot
sudo or read the key-holder's `/proc`, the temp tree tears down cleanly, and the digest is
provably unperturbed by the chmod walk. The fix is also the right *shape*: it opens every
ancestor instead of betting on which one was `0700`. The remaining notes are hardening,
not blockers.
