# Uncorrelated review — dev-review verification bound (`9b6a9286`)

Reviewer: Claude Opus 5, branch `reviewspdrev` (worktree `/workspace/aios-reviewspdrev`).
Commit under review: `9b6a9286` ("fix(ci): bound dev-review verification scope").
Fixes committed locally as `8485af35` and `0a95da3e`. Nothing pushed, no PR opened.

## Verdict

**The diagnosis is sound and the fix is the right shape — a prompt bound, not a
timeout raise or a machinery rewrite — but it shipped with one material hole and
a test that could not fail.** Both are fixed on this branch. Land after those two
commits.

The hole is not in what the bound forbids; it is in what the bound now
*sanctions*. By elevating "focused tests for affected behavior" to the reviewer's
principal form of verification, the change makes it load-bearing that the tree
the reviewer tests is the PR. It is not: `/mnt/review` is a clone of the
repository's **default branch**. That was tolerable while verification was
unbounded and diffuse; it is not tolerable once focused tests are the whole
verification budget, because a focused test run against master exercises the
unchanged code and passes for the wrong reason. A fast review that silently
verifies the wrong tree is a worse outcome than the 30-minute review it replaced.

## Issues found

### Fatal

None. Publication, soft-fail, archive, timeout ordering, tool grants, and clone
access are untouched by `9b6a9286` — verified against the diff. The change cannot
regress the "green Action, no comment" class the launcher exists to prevent.

### Serious

1. **The reviewer's clone is on the default branch, not the PR — and the prompt
   implied otherwise.** `GithubRepositoryResource`
   (`src/aios/models/github_repositories.py:41`) has **no ref/branch/sha field**,
   and `attach_session_repo` (`src/aios/sandbox/github_clone.py:290`) issues a
   plain `git clone --reference <cache> --dissociate <url> <dest>` — default
   branch HEAD, no checkout of anything else anywhere in the provisioning path
   (`grep -rn "head_sha\|checkout" src/aios/sandbox/` finds only a docstring).
   The launcher passes `CLONE_URL = head.repo.clone_url`, which for the ordinary
   same-repo PR is `eumemic/aios` — i.e. **master**. The prompt said only "The
   repository is cloned at /mnt/review", which any reader takes to mean the PR is
   checked out there.

   Corroboration that this is live, not theoretical: the DONE's own evidence for
   PR #2362 reports the reviewer running "a base-code mutation run" — base code
   is exactly what a default-branch clone hands it.

   Fixed in `8485af35`: the prompt now states the clone is on the default branch,
   names `head_sha` as the commit to reach, and gives the reviewer a check it can
   run itself (`git -C /mnt/review rev-parse HEAD`). Fetch mechanics are left to
   the model — `origin` is already the per-session git proxy, so `git fetch` works
   from inside the sandbox, and per CLAUDE.md the model handles that failure
   itself rather than the launcher scripting it.

2. **The new test asserts the constant's own words, so it cannot fail.**
   `test_review_scope_avoids_repeating_ci_and_exhaustive_work` read
   `reviewer.REVIEW_SCOPE` and asserted substrings of the literal it was written
   from. Delete `{REVIEW_SCOPE}` from the f-string in `main()` and the bound stops
   existing while the test stays green — a constant nothing sends is not a bound.
   Nothing pinned the `infra/agents/dev-review.json` half either, and that half is
   the *only* instruction a workflow child ever sees, so dropping it silently
   relocates the expensive tool loop to the other caller rather than removing it.

   Fixed in `0a95da3e`: `_Api` now records the `POST /v1/sessions` body, one test
   asserts the bound and the head-checkout instruction against the
   `initial_message` the launcher actually sends, and a second holds the same
   bound in the committed manifest.

### Minor (not fixed — flagged for the implementer's call)

3. **The "unchanged substantive diff" clause is unactionable on the launcher
   path.** It tells the reviewer not to repeat expensive checks "reported by an
   earlier eumemic-bot review", but the launcher prompt passes **no comments**.
   The manifest's request contract names `{repo, pr_number, head_sha, comments}`;
   the launcher supplies repo/pr/sha and nothing else. The clause therefore only
   binds if the model volunteers a `GET /repos/{repo}/issues/{n}/comments` — which
   the http_server allowlist permits, but nothing directs. If it *does* volunteer
   it, it pulls prior full review artifacts into context, which is itself a
   non-trivial token cost. Either pass the comments or drop the clause; leaving it
   inert is the one option that buys nothing. I did not change it because both
   directions are product calls, not defects.

4. **Repo-wide lint/type-check is forbidden; scoped lint/type-check is not
   explicitly permitted.** The bound says "focused tests" but offers no scoped
   counterpart for mypy/ruff, so a literal reader drops type-checking entirely.
   Low impact in practice — this repo's mypy is invoked whole-package
   (`uv run mypy src tests packages/...`), so a genuinely "scoped" run is not
   really on offer — but the asymmetry is worth a word if the prompt is revised.

5. **`uv sync --dev` is the floor under "focused tests".** The bound removes the
   repo-wide *suites*, not the dependency install that running any test at all in
   a fresh sandbox requires. Expect that fixed cost to survive. This is context
   for reading the first post-fix run, not a defect.

6. **~30s of tail slop in the launcher's poll (pre-existing, immaterial).**
   `wait_for_events` (`src/aios/api/routers/sessions.py:1130`) returns the moment
   events past `after` exist, so the DONE is right that the 30s is a long-poll
   maximum and not a sleep. One wrinkle: `session_status` is read from the same
   response, so if the final assistant event lands a beat before the step flips
   the session out of `active`, one further poll can burn its full 30s. Bounded
   and irrelevant against 10–30 minutes; noted only so it is not mistaken for a
   regression when the post-fix timings come in.

## Fixes applied

| SHA | Commit | Files |
|---|---|---|
| `8485af35` | `fix(ci): point the reviewer's clone at the PR head` | `scripts/eumemic_bot_review.py`, `docs/eumemic-bot-review.md` |
| `0a95da3e` | `test(ci): pin the review bound to the prompt and the manifest` | `tests/unit/test_eumemic_bot_review.py` |

Checks after both: `uv run pytest tests/unit/test_eumemic_bot_review.py -q` — 13
passed; full `uv run pytest tests/unit -q -n 4` — 6073 passed; `ruff check` /
`ruff format --check` clean on the touched paths; `mypy tests/unit/...` clean.
(`mypy scripts/` reports pre-existing bare-`dict` generics also present on
`origin/master`; `scripts/` is not in CI's mypy target, so it is out of scope.)

## Do the DONE's claims hold?

**Root cause — holds, with one caveat about provenance.** "The dominant
wall-clock cost is the review model's self-directed tool loop" is consistent with
everything I can check in-repo: the launcher prompt genuinely placed no bound on
verification, the manifest genuinely encouraged deeper inspection via the clone,
and the reviewer genuinely has `bash` plus a full working tree. I could **not**
independently re-verify the GitHub run timings (#2371/#2380/#2362) from this
checkout — no network to the Actions API, and the DONE itself notes the older
logs have expired. I take the timing evidence as reported. The mechanism stands
on its own, and the "base-code mutation run" detail in the cited artifact turned
out to be an independent tell for issue 1 above.

**"Checkout, token mint, publication, archive, and the long-poll are not material"
— holds.** The long-poll half I verified directly in the endpoint code (see
minor 6). The publication path is a single POST plus a marker round-trip.

**"Publication, soft-fail behavior, timeouts, clone access, tools, and targeted
bug-catching verification are unchanged" — holds** for the first five, verified
against `git diff origin/master...HEAD`. The sixth ("targeted bug-catching
verification unchanged") is the claim that did **not** hold as landed: targeted
verification against a master tree is not targeted verification of the PR. It
holds after `8485af35`.

**"53 passed" and "`git diff --check` clean" — reproduced** at `9b6a9286`.

**"No post-fix live timing exists; do not claim a precise old/new number" —
holds, and is the right call.** Nothing in this branch licenses a speedup figure
before the first live run. Read that run for two things, not one: the elapsed
time, and whether the artifact shows the reviewer actually reached `head_sha` in
`/mnt/review`.

## What I did not verify

- Live behaviour of the reviewer under the new prompt. Prompt bounds are
  probabilistic; only a real run shows whether the model honours them, and
  whether it honours the checkout instruction in particular.
- That `git fetch origin pull/<n>/head` specifically succeeds through the
  per-session git proxy. The proxy is documented to forward smart-HTTP fetch with
  auth injected, and the prompt deliberately does not prescribe the mechanics, so
  a model that finds one route blocked can take another — but this is the one
  step of `8485af35` that wants confirmation from the first live run.
- Fork PRs. `CLONE_URL` is the *head* repo, so on a fork the clone is the fork's
  default branch and `pull/<n>/head` does not exist there; `head_sha` does. The
  prompt asks for the SHA rather than a ref, which is the right shape for both
  cases, but no fork PR has exercised it.
