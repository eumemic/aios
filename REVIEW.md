# Uncorrelated review — Actions control-plane strip + unit rewrite (`487ae3c2`)

Reviewer: Claude Opus 5, branch `eumbotfcrev` (worktree `/workspace/aios-eumbotfcrev`).
Commits under review: `f66485fb` ("fix: strip Actions runner file command paths") and
`487ae3c2` ("fix: cover all Actions control env vars"), on top of `4ea7eefd`.
Fixes committed locally as `2fd5e84f`. Nothing pushed, no PR opened.

## Verdict

**The diagnosis is right and the fix works — I reproduced the false-fail on the
pre-fix tree and its absence after — but it shipped a formatting violation that
CI's own lint job would have failed on the next push, and the requirement it was
written to satisfy (the two new names in `_STRIPPED_ENV`) was not actually pinned
by any test.** Both are fixed here, along with four smaller items. Ready for PR
after `2fd5e84f`.

All six TASK.md / prompt checks pass on the reviewed tip; findings 1 and 2 are
about how the change would have fared *next*, not about the behaviour it claims.

## Verification of the required points

| # | Requirement | Result |
|---|---|---|
| 1 | `_STRIPPED_ENV` gains `GITHUB_STEP_SUMMARY`, `GITHUB_STATE` | ✅ `scripts/eumemic_bot_review.py:124-129` |
| 2 | Child env unreachable for `file_commands` paths under any key, name-strip not weakened | ✅ value strip added; all five names still in the list |
| 3 | Test uses an explicit runner-env dict, no ambient substring assertion | ✅ `os.environ` replaced wholesale |
| 4 | Launcher still reads `GITHUB_OUTPUT` from parent | ✅ `_record_published` (`:510`) reads its own `os.environ`; test green |
| 5 | `origin/master` is an ancestor; prior #2404 harness commits retained | ✅ all nine (`0e727e02`…`487ae3c2`) present |
| 6 | DONE.md claims match reality | ✅ root cause and pytest count confirmed; validation section incomplete (finding 1) |

**Root cause reproduced, not assumed.** Checked the pre-fix tree (`4ea7eefd`) out
into the worktree and ran the test under a simulated hosted-runner environment
carrying `_runner_file_commands` paths under `GITHUB_STEP_SUMMARY`,
`GITHUB_STATE`, and one unrelated key:

```
3 failed, 50 deselected      # 4ea7eefd, same ambient env
58 passed                    # this tip + fixes, same ambient env
```

DONE.md's "4 passed" for the named `-k` selection is accurate.

## Findings

### Blocking

**1. `ruff format --check` fails on the rewritten test — CI's lint job would have
gone red immediately.** `code-validation.yml:243` runs
`ruff format --check src tests …`, which covers this file. The new set literal
was written hand-wrapped:

```python
control_names = {
    "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STEP_SUMMARY", "GITHUB_STATE"
}
```

`ruff format` wants one element per line with a magic trailing comma, so
`--check` reported `Would reformat: tests/unit/test_eumemic_bot_review.py` on the
reviewed tip. This is the same failure class the task exists to close — green
`pytest` locally, red CI — one job over. DONE.md's validation section lists only
the pytest run; CLAUDE.md requires mypy, ruff check *and* ruff format before
every commit. Fixed, and `ruff check`/`ruff format --check` are now clean over
all of `src tests`.

### Serious

**2. The name strip of `GITHUB_STEP_SUMMARY` / `GITHUB_STATE` — TASK.md item 1 —
was not pinned by any test.** Deleting both names from `_STRIPPED_ENV` left the
entire file green:

```
3 passed, 50 deselected      # with both names deleted, before this fix
```

The reason is that the test's values for those two keys contain `file_commands`,
so the *new value strip* removed them regardless of the name list. The two
mechanisms were entangled, and the one the task was filed for was the one not
under test. That matters because the value strip keys off `_runner_file_commands`
— an undocumented internal of the runner's temp layout, not a contract. If GitHub
renames that directory, the name list is the only cover left, and nothing would
have caught its removal.

Fixed by splitting the mechanisms across two tests. New
`test_control_variables_are_stripped_by_name_not_only_by_path` is parametrized
over all five names and plants a value the marker cannot match
(`/runner/_temp/control-plane-abc`), so only the name list can remove it. Both
tests were mutation-checked: deleting the two names now fails 2 cases; deleting
the value-strip clause fails 3.

### Minor

**3. The stronger value assertion was dropped when it no longer had to be.** The
rewrite replaced `assert not [v for v in env.values() if "file_commands" in v]`
with a single `assert "RUNNER_TEMP_SUMMARY" not in env`. That assertion was only
unsafe because it ran over the *ambient* environment; once `os.environ` is
replaced wholesale with an explicit dict it is both ambient-proof and strictly
stronger than naming one key. Restored, with a comment saying why it is safe here.

**4. Nothing pinned the filter's blast radius.** No test asserted that an ordinary
inherited variable *survives* `_agent_command`. `PATH` is load-bearing — the child
needs it to find `codex` / `claude` / `pi` at all — and a value filter that
over-matched would be invisible to the unit suite while breaking every real run.
Added `PATH` to the runner env and asserted it comes through unchanged.

**5. `_CONTROL_PATH_MARKERS` carried a dead element.**
`("file_commands", "_runner_file_commands")` — the second can never match without
the first, which is a substring of it. Collapsed to a single
`_CONTROL_PATH_MARKER` and dropped the `any()`, per CLAUDE.md's extreme-simplicity
line.

**6. The comment overclaimed the strip as containment.** "deny those paths
wherever they occur" reads as though the agent can no longer reach the control
files. It can: `RUNNER_TEMP` is still inherited, `_runner_file_commands` sits
directly beneath it, and hosted runners default it to `/home/runner/work/_temp`,
so `ls $RUNNER_TEMP/_runner_file_commands/` finds the UUID-named files without
guessing. Left as defence in depth — stripping `RUNNER_TEMP` buys nothing against
a predictable path — but the comment now says so, and points at the actual
containment (`contents: read`, `persist-credentials: false`, both already in
place). The rest of this file argues its security properties carefully; this line
should not be the one that oversells.

**7. Documentation gaps around the two new variables.** The `_STRIPPED_ENV`
comment block gives a per-variable rationale for `GITHUB_OUTPUT` / `GITHUB_ENV` /
`GITHUB_PATH` and nothing for the two additions; the test docstring likewise. Both
now cover `GITHUB_STATE` (mutates later steps of this job) and
`GITHUB_STEP_SUMMARY` (the "did not post" net writes the operator's account of
the run there, `eumemic-bot-review.yml:137-145` — an agent holding that path
writes the second half of the same forgery). Also fixed a stale cross-reference
to `test_stripping_github_output_does_not_break_the_signal`, which has been
`…_the_publication_signal` since `4ea7eefd`.

## Checked and deliberately not changed

- **`monkeypatch.setattr(reviewer.os, "environ", runner_env)`** swaps the real
  `os.environ` for a plain dict process-wide for the test's duration — broader
  than it looks. Kept: `_agent_command` is a pure call, monkeypatch restores it,
  and full replacement is the only way to be genuinely ambient-proof, which is
  the whole point of the rewrite.
- **Stripping `GITHUB_STEP_SUMMARY` from the child does not break the safety
  net's summary.** That step runs in its own runner shell with its own
  environment (`eumemic-bot-review.yml:137`), not in the agent's — the same
  argument that makes the `GITHUB_OUTPUT` strip free.
- **`scripts/` is outside CI's ruff and mypy paths** (`code-validation.yml:242-246`
  covers `src tests packages/… connectors/…`), so the launcher itself is
  unlinted either way. Ran both against it by hand — clean. Widening the CI paths
  to include `scripts/` is a real gap but belongs to its own change.

## Commands run

```
uv run pytest -q tests/unit/test_eumemic_bot_review.py            # 53 -> 58 passed
uv run mypy tests/unit/test_eumemic_bot_review.py                 # clean
uv run ruff check src tests && uv run ruff format --check src tests
uv run bash scripts/verify_eumemic_bot_review_gate.sh             # ALL CHECKS PASSED
```

The gate script is the sanctioned one-command re-verification from the previous
round, and it still passes end to end: 58 unit tests, the GITHUB_OUTPUT mutant
killed, the forged-`published=true` attack refused with the safety net still
firing, and the honest agent still publishing.

Plus the pre-fix reproduction, the simulated-runner ambient run, and the three
mutation checks quoted above.

**Full unit suite — two runs, and they do not agree.** `uv run pytest tests/unit
-q -n 4` gave `10 failed, 6178 passed` and then `2 failed, 6186 passed`, with
*disjoint* failure sets (`test_attachment_staging`, `test_host_dir_reaper`,
`test_image_resize`, `test_revocation_kinds_coverage` in the first;
`test_invoke_session_tools`, `test_litellm_param_validation` in the second).
Every one of them passes when its file is run on its own, so these are
pre-existing ordering/parallelism flakes under xdist, not regressions: this
branch changes no `src/` file, and none of the failing test files differ from
`origin/master`. Reporting it because the numbers are real, not because it
blocks this PR — but a suite whose failure set changes run to run is worth its
own issue.
