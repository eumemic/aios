# Uncorrelated review — `botpost2410f` tip `a8da0ea8` (aios#2410)

**Verdict: changes requested — 1 High, 1 Medium, both fixed on this review
branch (`botpost2410frev`, tip `a3a7612e`).**

The design TASK.md asks for is right, and it is the right design: the
chicken/egg is genuinely broken (there is no base-SHA Python left to execute),
and the `mktemp` race is closed by a boundary rather than by a better temp
name. Both properties are structural now, not argued.

What it does not do is post. `actions/upload-artifact` has excluded
dot-prefixed files by default since v4.4, and the one file crossing the new job
boundary is `.eumemic-bot-review.md` — so the upload matches nothing,
`if-no-files-found: error` fails it, and `publish` downloads nothing. The
isolation is real and the review still never reaches the PR. That is the same
silent-miss shape this workflow has now failed on three ways running, and it
would have shipped green.

## Scope

Verified against TASK.md items 1–5. Not reviewed: Track G (excluded). Nothing
was pushed, merged, or opened as a PR.

---

## F1 (High) — the review artifact is a hidden file, so it is never uploaded

`.github/workflows/eumemic-bot-review.yml` at `a8da0ea8`:

```yaml
          REVIEW_ARTIFACT_PATH: ${{ github.workspace }}/.eumemic-bot-review.md
...
      - uses: actions/upload-artifact@v4
        with:
          path: .eumemic-bot-review.md
          if-no-files-found: error
```

From the action's own v4 README: *"With `v4.4` and later, hidden files are
excluded by default"*, and *"Hidden files are defined as any file beginning
with `.`"*. `@v4` is a floating major tag, so every run resolves to v4.4+.

The chain then fails end to end, quietly at each link:

1. the upload pattern matches zero files;
2. `if-no-files-found: error` fails the step — which is `continue-on-error`,
   so the job stays green;
3. `publish` runs, `download-artifact` finds nothing, the mint is skipped;
4. the run summary says "did not post" and the check is green.

This was invisible to every gate. The single-job predecessor read the file off
the same runner's disk, where the dot prefix meant nothing; the property only
became load-bearing when the file started crossing a job boundary in this
commit. `tests/unit/test_eumemic_bot_review.py` asserted
`upload["with"]["path"] == ".eumemic-bot-review.md"` — pinning the defect. It
is the only dot-prefixed upload in the repository, so the sibling in
`code-validation.yml` offered no counter-example either.

**Fixed** in `a3a7612e`: `include-hidden-files: true`. Chosen over renaming the
file because the input states the intent where a reader of the upload step will
look, and the dot prefix is what the launcher, the docs and the publisher's
read path already agree on. The action's warning about that input concerns
directory sweeps picking up `.env`-shaped files; this is a single explicit
file, so there is nothing to sweep.

`test_upload_opts_into_the_hidden_artifact_filename` pins the launcher's write
path, the upload path and the publisher's read path as **one chain** — both
halves are silent when wrong, and the second (`review/` + basename) had no
coverage at all — and requires the input only while the basename starts with
`.`, so a later rename to a non-hidden name stays valid without it. Confirmed
the test fails against `a8da0ea8`'s workflow, not just passes against the fix.

## F2 (Medium) — `always()` fires the "did not post" summary on superseded runs

```yaml
  publish:
    needs: agent
    if: ${{ always() && !github.event.pull_request.draft }}
```

`always()` is true when the run is **cancelled**, and this workflow sets
`cancel-in-progress: true`. So every push that supersedes an in-flight run
starts `publish` on a fresh runner, finds no artifact, and writes
`### eumemic-bot review did not post` to that run's summary.

Not a security issue — a signal-quality one, and the signal is the whole point
of the step. The workflow's own comment says a run that published nothing
"says so where a reader sees it without opening logs"; a marker that also fires
for every superseded push is one a reader learns to skip. This is a regression
from the single-job shape, where a cancelled run simply took the summary step
with it.

**Fixed** in `a3a7612e`: `!cancelled()`, which is GitHub's documented form for
"run unless the run was cancelled" and still overrides the default
`needs`-failed skip — so a *failed* agent job (including a `timeout-minutes`
kill, which is a job failure, not a run cancellation) still reaches the
summary. Pinned in `test_workflow_never_fails_the_pr_check_on_an_ops_miss`
alongside the existing step-level `always()`, which is correct and unchanged.

---

## Checklist verdicts

| # | Requirement | Verdict |
|---|---|---|
| 1 | Job A `agent` has no App token / `GH_TOKEN`, uploads only the markdown; Job B `publish` `needs: agent`, fresh runner, downloads only that, mints there, runs no PR-head/base Python | **Holds.** No `EUMEMIC_BOT_PRIVATE_KEY`, `create-github-app-token` or `GH_TOKEN` anywhere in `jobs.agent` (asserted over *every* step, not just the agent step); `jobs.publish` has no `actions/checkout` and its only `run:` is inline shell. The launcher additionally refuses to start the agent phase if `GH_TOKEN` is set, and still drops the checkout's persisted `http.*.extraheader` before the agent runs. |
| 2 | Publisher is not "base SHA Python"; works without master having `publish`; POSTs as eumemic-bot, verifies, keeps continue-on-error + summary | **Holds.** Inline `gh api --method POST` with the markdown passed as a jq-built JSON body — never through the shell, so agent-authored markdown cannot inject. Verification is stricter than the Python it replaced: `html_url` **and** the run-specific marker **and** `user.login == "eumemic-bot[bot]"` (the old path did not check the author). All four steps stay `continue-on-error`. Nothing in the trust path depends on master's script. |
| 3 | Broken `git show $BASE_SHA:…` + same-runner `mktemp` removed; docs and tests updated | **Holds.** `assert "git show" not in text` and `assert "mktemp" not in text` cover the whole workflow file, so neither can come back. `docs/eumemic-bot-review.md` is rewritten to the two-job flow and states the boundary in the terms that make it true (filesystem, PATH, processes and `/proc` do not cross runners). |
| 4 | Coding-agent routing, mint-after-agent, gVisor fixes, chroot/BusyBox work retained; `origin/master` an ancestor | **Holds.** All three prefixes still route and install singly, and only the routed proxy secret reaches the agent step. `4c359ef0` (chroot before the privileged loader), `f23cea3a` (`docker/sandbox-resolv.conf`) and the corrected operator `PATH` are all present and unmodified. `origin/master` (`01f76bb8`) is an ancestor of `HEAD`; no rebase needed. |
| 5 | DONE.md claims match reality | **Accurate.** `99466c28`, `01f76bb8`, `a9aa9e53` all exist and are what it says they are; the ancestry claim checks out; `32 passed` reproduced exactly at `a8da0ea8`. It is now 33, and DONE.md predates F1/F2. |

## Notes, not findings

- **Job rename is safe.** `review` became `agent` + `publish`, which changes the
  reported check names. `master`'s required contexts are
  `detect, lint, unit, integration, e2e (docker), e2e (non-docker)` — this
  workflow is not among them, as its "must never FAIL the PR check" stance
  requires. Checked against the live branch-protection settings, not assumed.
- **The agent still controls the artifact's contents,** and always did; that is
  the design. What changed is that the bytes are now only ever *data* to the
  token-holding process. Worth stating plainly because it is the one thing the
  job split does **not** buy: an agent that can reach `ACTIONS_RUNTIME_TOKEN`
  off the runner could upload the artifact itself, and would gain nothing —
  same markdown, same JSON body, no reachable token.
- **`run_publish_phase` in `scripts/eumemic_bot_review.py` is now dead in the
  Action path** and labelled "legacy … for manual compatibility". TASK.md
  explicitly permits leaving it unused, so I left it; against CLAUDE.md's
  "don't deprecate, delete" it is a shim, and it keeps a second, differently
  behaved implementation of the publish contract (it does not check the comment
  author) one wiring mistake away from a token-holding step. Worth deleting in a
  follow-up, not on a review branch.
- **Comment growth is unchanged:** every `synchronize` posts a new comment
  rather than updating the previous one. Pre-existing, out of scope, and
  arguably correct given the per-SHA marker.

## Local results

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py
33 passed

$ uv run pytest tests/unit -q -n 4
6163 passed, 10 warnings in 92.34s

$ uv run ruff check src tests && uv run ruff format --check src tests
All checks passed! / 1091 files already formatted

$ uv run mypy src tests
Success: no issues found in 1091 source files
```

Not pushed, not merged, no PR opened.
