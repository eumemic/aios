# Review — publish `### Code review` after a `/wait` read timeout (`2fa61ffe`)

Uncorrelated review of the aios#2410 comment-post fix on `botpost2410rev`
(same tip as implementer branch `botpost2410`). Scope per the brief:
comment-post / review-finish only in `scripts/eumemic_bot_review.py` plus its
unit tests. The gVisor/runsc product fix (`789174bc`, `ffef992c`) was not
touched; Track G was not touched; nothing was pushed, merged, or opened as a PR.

(The previous round's gVisor review lives in git history at `ffef992c:REVIEW.md`
— this file is replaced per review round, as it was at `ffef992c` and `b334ae63`.)

## Verdict

**The launcher fix is correct and complete.** All four behavioral items of
TASK.md hold as written, and I found no defect in `scripts/eumemic_bot_review.py`
— I did not change a line of it (`git diff scripts/` is empty).

Two things were wrong around it, both now fixed on this branch:

1. **`2fa61ffe` introduced a mypy error that CI runs and DONE.md never checked.**
   The parent commit was clean. This alone would have failed the pipeline the
   commit exists to unblock.
2. **The delivered tests did not pin the delivered fix.** Three separate
   mutations that revert the core of the change kept all 15 committed tests
   green — including deleting the `retry_timeout=True` opt-in, i.e. the fix
   itself. Coverage now catches all three.

| TASK.md item | Verdict |
|---|---|
| 1. `/wait` transport timeout retries; HTTP 4xx/5xx stay fatal | **Correct** — both urllib timeout shapes handled; verified empirically |
| 2. Deadline exit still reads events and publishes an existing artifact | **Correct** — `_review_from_events` runs before the status check |
| 3. Archive timeout must not undo a successful publish | **Correct** — already held; was untested (F4) |
| 4. Rebased on `origin/master`, gVisor commits kept | **Correct** — 0 behind / 3 ahead, both runsc commits intact |
| 5. Unit test pins the retry | **Insufficient as delivered** — F2/F3 |

DONE.md's factual claims check out: commit sha matches `HEAD`, and
`uv run pytest -q tests/unit/test_eumemic_bot_review.py` really does report
`15 passed`; the two ruff claims also reproduce on the committed tree. What
DONE.md omits is that `uv run mypy src tests` — mandated by CLAUDE.md before
every commit — was not run and did not pass.

## What I verified

**Item 1.** `_request` now splits the two exception shapes urllib actually
produces. I confirmed both against a live socket rather than from memory:

```
response/read phase (server accepts, never replies) -> builtins.TimeoutError('timed out')
connect phase (SYN blackholed)                      -> urllib.error.URLError(TimeoutError('timed out'))
```

The first is the incident's `The read operation timed out`; the second is why
the `isinstance(exc.reason, TimeoutError)` branch is not speculative. Both are
needed, and both are correct. `HTTPError` is caught first and still `_die`s, so
4xx/5xx remain fatal even under `retry_timeout=True`; a non-timeout `URLError`
(connection refused) also stays fatal. The retry does not advance `after`, which
is right — a timed-out poll observed no events, so re-asking from a bumped
cursor could skip the artifact turn itself.

**Item 2.** `_wait_until_working_stops` returns `"active"` on deadline instead
of dying, and `_ask_for_review_artifact` calls `_review_from_events` *before*
inspecting the status. A session that produced the artifact and then hung
publishes. The `"still working … without a `### Code review` artifact"` death is
reached only when the event log genuinely lacks it, and it correctly does *not*
spend a corrective turn on a budget that is already exhausted.

**Item 3.** In `main`, publication and its GitHub round-trip verification happen
inside `try:`; `_archive` runs in `finally` and swallows its own `SystemExit`.
A timed-out archive after a successful post leaves exit 0, so the workflow's
`steps.review.outcome == 'failure'` guard does not fire the "did not post"
summary for a review that did post. Correct, and now pinned.

**Item 4.** After `git fetch origin`: `git rev-list --left-right --count
origin/master...HEAD` = `0  3`. The three commits ahead are `789174bc` and
`ffef992c` (runsc egress + operator-binary) and `2fa61ffe`. Nothing dropped.

**Item 5.** Mutation-tested, see F2–F4.

## Findings

### F1 — `2fa61ffe` introduced a mypy error (must-fix; **fixed**)

`tests/unit/test_eumemic_bot_review.py:115` in the committed test:

```
error: Incompatible return value type (got "object", expected "dict[str, Any]")  [return-value]
```

`polls = iter([TimeoutError(...), {...}])` infers `Iterator[object]`, so the
narrowed `return result` does not type. Confirmed introduced, not pre-existing:

```
mypy on tests/unit/test_eumemic_bot_review.py @ 2fa61ffe (this commit)  -> 1 error
mypy on tests/unit/test_eumemic_bot_review.py @ ffef992c (its parent)   -> Success
```

CLAUDE.md requires `uv run mypy src tests` before every commit and CI runs it.
Shipping this would have red-checked the branch whose entire purpose is to make
the review pipeline work. Fixed by giving the list an explicit
`list[dict[str, Any] | BaseException]` annotation.

### F2 — the retry opt-in was not pinned by any test (**fixed**)

`test_wait_retries_a_read_timeout` monkeypatches `_request` wholesale, so it
never exercises the `retry_timeout` plumbing that makes the retry reachable.
Deleting `retry_timeout=True` from the `/wait` call site — reverting the fix to
the exact behavior that killed run 34243703484, where `_request` `_die`s and the
loop's `except` can never run — left **all 15 committed tests passing**.

Added `test_request_reraises_a_transport_timeout_only_when_asked` (both timeout
shapes, and that the default still fails hard),
`test_request_keeps_non_timeout_failures_fatal_under_retry_timeout` (4xx, 5xx,
refused), and `test_wait_survives_timeouts_through_the_real_request_path`, which
stubs only `urlopen` so the whole chain runs. That last one fails under the
mutation.

I also strengthened the committed `test_wait_retries_a_read_timeout` rather than
leaving it redundant: it now asserts the cursor does **not** advance across a
timeout (`after=0` twice), an invariant nothing else covers.

### F3 — the deadline-returns-`"active"` path was not pinned (**fixed**)

Restoring the old `_die` at the end of `_wait_until_working_stops` — i.e.
undoing TASK item 2 at its source — also left all 15 tests green, because
`test_deadline_still_publishes_an_existing_artifact` stubs the whole function.
Added `test_wait_returns_active_when_the_deadline_passes` (asserting no poll is
even issued past the deadline) and
`test_deadline_without_an_artifact_is_fatal_and_skips_the_corrective_turn`.

### F4 — TASK item 3 had no coverage at all (**fixed**)

Nothing asserted that a failing archive leaves a published review intact; the
existing `main` tests all archive successfully. Added
`test_a_failed_archive_does_not_undo_a_published_review`, which fails if
`_archive` stops swallowing its `SystemExit`.

### F5 — residual: the post-deadline `GET /events` is itself un-retried (**not fixed**)

After the wait loop gives up, publication now hinges on a single un-retried
30-second `GET /v1/sessions/{id}/events`. If *that* request times out,
`_request` `_die`s and the run ends in exactly the incident's outcome — green
check, zero `### Code review` comments — from the same transport fault class the
commit set out to survive.

I did not change it. It is outside the four items, the pathology observed was
specific to the long-poll (a 30s server-held socket against a 60s client
budget) rather than to a short request, and retrying it needs deadline plumbing
that would widen a deliberately narrow fix. Flagging for the Shepherd rather
than deciding it here.

### F6 — `_archive` prints `FATAL:` for a survivable condition (**not fixed**)

`_archive` catches the `SystemExit` and warns, but `_die` has already written
`FATAL: POST …/archive failed: …` to stderr. The run then exits 0. This is
cosmetic — the workflow keys the "did not post" summary off the step outcome,
not off log text — but it is not harmless: TASK.md's own incident narrative
quotes that `FATAL` line as part of the failure, which is precisely the
misreading it invites. TASK.md item 3 explicitly blesses the current archive
behavior ("already warns; keep that"), and every fix I could see costs either
duplicated request code or a second flag on `_request`, so I left it. Worth a
one-line cleanup if `_request`'s error handling is ever revisited.

### F7 — considered and rejected: no backoff in the retry loop

The `except TimeoutError: … continue` has no sleep or attempt cap, which reads
like a hot-loop risk against a 2700s deadline. It is not one: only `TimeoutError`
retries, and a socket timeout costs the full `_WAIT_HTTP_TIMEOUT` (60s) of wall
clock before it raises — fast failures (refused, 4xx/5xx) are fatal and exit
immediately. That bounds the loop at ~45 iterations over the whole budget. No
change needed; adding backoff would be the kind of defensive guard CLAUDE.md
argues against.

## Tests

```text
uv run pytest -q tests/unit/test_eumemic_bot_review.py     -> 24 passed
uv run ruff check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py    -> All checks passed!
uv run ruff format --check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py -> 2 files already formatted
uv run ruff check src tests scripts                        -> All checks passed!
uv run mypy tests/unit/test_eumemic_bot_review.py          -> Success: no issues found
```

Mutation results (each mutation reverts one piece of the fix; each is caught by
exactly one **new** test, and by none of the 15 committed ones):

| Mutation | Committed suite | With this branch's tests |
|---|---|---|
| drop `retry_timeout=True` at the `/wait` call site | 15 passed | `test_wait_survives_timeouts_through_the_real_request_path` FAILED |
| restore `_die` on deadline in `_wait_until_working_stops` | 15 passed | `test_wait_returns_active_when_the_deadline_passes` FAILED |
| make `_archive` propagate its failure | 15 passed | `test_a_failed_archive_does_not_undo_a_published_review` FAILED |
| advance `after` on a timed-out poll | 15 passed | `test_wait_retries_a_read_timeout` FAILED |

The launcher script was restored bit-identical after each mutation
(`git diff scripts/eumemic_bot_review.py` empty).

**Caveat:** the full `uv run mypy src tests` could not be completed in this
worktree — it is killed by the OOM killer (exit 137, no output) on this box, on
three attempts. The file-scoped run above is clean and is what surfaced F1; the
repo-wide run still needs to pass in CI or on a larger machine before merge.

## Changes made on this branch

`tests/unit/test_eumemic_bot_review.py` only — F1's type fix, the cursor
assertion added to the existing retry test, and six new tests. No production
code changed.
