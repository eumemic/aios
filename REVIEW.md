# Uncorrelated review — aios#2410 tip `caee7e8d`

- **Round**: `botpost2410s` (implementer grok-4.6, worktree `aios-botpost2410s`)
- **Checker**: claude-opus-5, worktree `aios-botpost2410srev`, branch `botpost2410srev` (maker ≠ checker)
- **Tip reviewed**: `caee7e8d` "fix(ci): isolate the review agent from the proxy-key holder"
- **Verdict**: **FAIL** — the privilege split is the right *shape*, but as written the
  review launcher cannot complete a single review on a GitHub-hosted runner. Two
  independent blocking defects, both reproduced end-to-end against the real code.

State: `origin/gvisorgrn = 39dda7d7`; HEAD is one product commit ahead and **unpushed**,
as TASK says. Not pushed, not merged, no PR opened by this review. `TASK.md` is dirty in
the worktree (Shepherd's round file); untouched here.

The tip touches five files and no `src/`:
`.github/workflows/eumemic-bot-review.yml`, `DONE.md`, `docs/eumemic-bot-review.md`,
`scripts/eumemic_bot_review.py`, `tests/unit/test_eumemic_bot_review.py`.

## What passes

**High — privilege boundary (design).** The claim in `DONE.md` is now the real shape, not
prctl theatre. `run_agent_phase` calls `_require_agent_user()` (`scripts/eumemic_bot_review.py:898`)
*before* the key is read, and `_run_harness` unconditionally routes the harness through
`_drop_into_agent_user` (`:377`, called at `:791`) — there is no un-dropped path left.
The gate fails closed in the right direction: `_require_agent_user` (`:341`) dies if the
user shares this euid, is uid 0, or can sudo, and `_agent_user_may_sudo` returns `True`
(refuse) when it cannot prove the contrary. `setpriv --reuid/--regid/--clear-groups
--no-new-privs --inh-caps=-all` is a genuine boundary: a different uid cannot read the
launcher's `/proc`, cannot `ptrace` it, and `no_new_privs` neutralises setuid `sudo` even
if sudoers changed under it. The reusable key stays in the launcher's `_ProxyBroker`; the
spec file handed across the boundary carries only the loopback token. Prose in the script
header, `docs/eumemic-bot-review.md` and the workflow comments no longer claims prctl
seals a GH runner — the honesty ask in TASK is met, and `test_comments_do_not_claim_prctl_seals_github_runners`
pins it.

**Medium — artifact gate (design).** `require_inspection_evidence` (`:450`) refuses to
return an artifact unless it carries `<!-- inspected: lines=N sha256=HEX -->` whose full
64-hex digest equals the launcher's own `git diff base...head` digest (`diff_evidence`, `:420`),
and it is invoked both inside `_run_harness` (`:824`) and again in `run_agent_phase` before
the write. Exit status alone no longer authorises anything; heading-only output is refused
with a distinct banner and exit code. The line count is parsed for diagnostics only and
cannot accept — correct, since it is public at the `.diff` URL.

**Keep list.** Intact by construction (the diff touches no `src/`): hosts-first
`resolve_ipv4` / `build_resolve_ipv4_fn` with `ResolveScope` and operator-controlled
`operator_hosts` (`src/aios/sandbox/setup.py:565`, `:535`), runsc gateway work, and
proxy-key-out-of-harness-env (`REVIEW_PROXY_KEY_FILE` staged in a separate step, read and
unlinked in `_proxy_key`).

**Checks run** (focused only, per ops constraint — no full suite, no `-n`):
- `uv run pytest tests/unit/test_eumemic_bot_review.py -q` → **56 passed**
- `uv run ruff check` / `ruff format --check` on the two changed Python files → clean
- `uv run mypy scripts/eumemic_bot_review.py` → clean

## Blocking findings

### F1 — the dropped agent cannot run the mandated proof commands: git refuses the runner-owned checkout cross-uid

`_drop_into_agent_user` runs the harness as `eumemic-review` with `cwd = os.getcwd()` — the
`actions/checkout` workspace, owned by `runner` — and with `HOME` redirected to the temp dir
(`:388`). Git's `safe.directory` ownership check is per-uid and read from the *current user's*
config; `actions/checkout`'s `set-safe-directory` writes it to `runner`'s `~/.gitconfig`, which
the agent user never reads. Nothing in the tip configures `safe.directory`, `GIT_CONFIG_GLOBAL`,
or re-owns the tree for the agent.

So the exact commands the prompt mandates (`:759-760`) fail for the agent:

```
$ sudo -n -- setpriv --reuid=eumemic-review --regid=eumemic-review --clear-groups \
      --no-new-privs --inh-caps=-all -- env HOME=/tmp/agenthome PATH=/usr/bin:/bin \
      sh -c 'cd /tmp/exp-repo && git status'
fatal: detected dubious ownership in repository at '/tmp/exp-repo'
```

and `git --no-pager diff BASE...HEAD | sha256sum` therefore hashes **empty output**
(`e3b0c442…`, the sha256 of the empty string). Driving the real `_run_harness` with a
"perfect" harness that does exactly what the prompt says:

```
::error title=NO EVIDENCE OF INSPECTION — refusing to publish a verdict::inspection evidence
does not match the diff: agent claimed lines=0 sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,
launcher computed lines=7 sha256=6083b1ee…
```

The prompt explicitly instructs the agent, when its shell cannot run those commands, to emit
no evidence line and render no verdict. The two halves of this round therefore compose into a
system that refuses **every** review, not just ungrounded ones: the Medium gate is fail-closed
against the High fix. An agent might stumble onto `git -c safe.directory=…` on its own, but the
prompt tells it not to deviate ("Run exactly"), so that is luck, not design.

Fix shape: give the agent a legitimate view of the tree — e.g. write a gitconfig into the
agent HOME and pass `GIT_CONFIG_GLOBAL` in `dropped_env` with `[safe] directory = <cwd>`, or
hand the agent an agent-owned copy of the checkout (`cp -a` + `chown`), which also stops the
untrusted agent from writing to the launcher's tree. Whichever is chosen, it needs a check
that the drop can actually reproduce `diff_evidence` — the launcher already knows the expected
digest, so it can verify the boundary before spending a review.

### F2 — `TemporaryDirectory` cleanup raises `PermissionError` after the chown; no artifact is ever written

`_drop_into_agent_user` does `chmod 0755` then `sudo chown -R eumemic-review <temp>` (`:394-396`)
on the dir that `_run_harness` holds open as `with tempfile.TemporaryDirectory(...) as temp`
(`:789`). The launcher is unprivileged and no longer owns that directory, so on `with`-exit it
can neither unlink the files inside (needs write on a 0755 dir owned by another uid) nor
`chmod` it in `tempfile`'s recovery path (needs ownership). The later `sudo chmod -R a+rX`
(`:819`) adds read/execute, not write, so it does not help.

Reproduced end-to-end through the real `_run_harness`, with git made to work so the digest
matched (i.e. the *success* path):

```
PermissionError: [Errno 1] Operation not permitted: '/tmp/eumemic-review-_7zxtqb9'
  File ".../tempfile.py", line 1145, in cleanup
  File ".../shutil.py", line 707, in _rmtree_safe_fd
  File ".../tempfile.py", line 477, in _resetperms
```

`_run_harness` never returns, so `run_agent_phase` never reaches `artifact_path.write_text` —
a fully valid, fully evidenced review is thrown away with a traceback. The same bug also
destroys the Medium fix's "distinct, loud state" property: on the refusal path the
`SystemExit(NO_EVIDENCE_EXIT_CODE)` is superseded by the cleanup `PermissionError`, so the
step exits 1 with a traceback instead of 3 with the banner.

This is unconditional on any real drop — it is not runner-specific and not race-dependent.

Fix shape: do not hand the launcher's own `TemporaryDirectory` to another uid. Create an
agent-owned subdirectory (`sudo install -d -o <user> -m 0700 <temp>/agent`), keep the parent
owned by the launcher, copy/read the artifact out, and `sudo rm -rf` the subdirectory before
the `with` exits (belt: `ignore_cleanup_errors=True`).

## Why the tests did not catch either

Every isolation test monkeypatches the boundary: `test_drop_wraps_the_harness_with_setpriv_no_new_privs`
stubs `_sudo` to a success `CompletedProcess` (so no chown happens), and
`test_run_agent_actually_drops_before_exec` / the `passthrough_drop` fixture replace
`_drop_into_agent_user` with identity — the temp dir stays launcher-owned and cleanup
succeeds. They assert the argv shape, which is real value, but nothing exercises a drop.
A test that performs an actual `setpriv` drop when a suitable non-sudo user exists (skip
otherwise), or a launcher self-check that the dropped uid can reproduce `diff_evidence`,
would have caught both findings in one pass.

## Non-blocking notes

1. **`_die` diagnostics lost the harness name.** After the wrap, `command[0]` is `"sudo"`, so
   `_die(f"{command[0]} exited with status …")` (`:813`) and the `FileNotFoundError` branch
   (`:806`) now say "sudo is not installed" / "sudo exited with status N" for a codex/claude/pi
   failure. Cheap fix: keep the harness argv for messages.
2. **"the digest is the only accepting channel" is slightly overstated for public repos.**
   A PR's `.diff` URL is itself a three-dot diff; where it is byte-identical, a networked agent
   could hash the fetched bytes without inspecting the checkout. The gate still does what TASK
   asks (heading-only is refused) and `eumemic/aios` is private, so this is a prose caveat, not
   a defect.
3. The harness spec file is world-readable (`chmod 0644`, `:395`) — correct as written, since it
   carries only the loopback token, but it is worth a comment that anything added to
   `dropped_env` becomes readable to every user on the runner.

## Verdict

**FAIL.** The High's security property is genuinely achieved in design (separate uid, no sudo,
`no_new_privs`, key never crosses) and the Medium's gate is correctly shaped and fail-closed —
but the two together cannot produce a published review on a GH-hosted runner: F1 makes the
evidence unobtainable and F2 crashes the launcher before the artifact is written, on both the
success and refusal paths. Both are reproducible in one run and both have small, local fixes.
