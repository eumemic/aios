# Uncorrelated review — eumemic-bot review comment fix (`1bb1cacc`)

Reviewer: Claude Opus 5, branch `botreviewrev`. Commit under review: `1bb1cacc`
("fix: require eumemic-bot review comment before archive"). Fixes committed
locally as `0d1c1b83`. Nothing pushed, no PR opened.

## Verdict

The commit's **shape is right** — publication belongs to the launcher, not to the
review session, and `archive_when_idle: false` is the correct consequence. But
the implementation could not post a comment under any circumstances: three
independently fatal defects each guarantee zero output on the real API. The
reported symptom ("archives without posting `### Code review`") would have
reproduced unchanged after this commit.

## Issues found

### Fatal (each alone means no comment is ever posted)

1. **Wrong list envelope.** `_review_from_events` iterated
   `payload.get("items", [])`. `ListResponse[T]` serializes its rows under
   **`data`** (`src/aios/models/common.py`, confirmed against the committed
   `openapi.json`). The artifact scan always saw an empty list, so every run
   took the "no artifact" branch regardless of what the model wrote.

2. **Wrong wait primitive.** `_await_turn` used `GET /v1/sessions/{id}/await`,
   which resolves on `last_reacted_seq >= watermark`. `reacting_to` advances on
   **every** assistant message, including the model's very first *tool-call*
   turn — so the launcher declared the review finished seconds after it started,
   found no artifact (correctly — the review had not happened yet), spent its one
   corrective turn immediately, and then failed. Replaced with a cursor-threaded
   `GET /v1/sessions/{id}/wait` long-poll that blocks until
   `session_status != "active"`. That status is derived as
   `last_stimulus_seq > last_reacted_seq OR open_tool_call_count > 0`, and tool
   results are stimuli, so there is no false-idle window mid-turn.

3. **Client socket timeout shorter than the server long-poll it requested.** The
   request asked the server to hold for up to 60s while the socket deadline was
   30s. Worse, a urllib **read** timeout raises a bare `TimeoutError`, not
   `urllib.error.URLError`, so the handler did not catch it and the launcher died
   on an unhandled traceback. Now `_WAIT_SECONDS = 30` with a socket deadline of
   `2 * _WAIT_SECONDS`, and `TimeoutError` is translated alongside `URLError`.

### Serious

4. **Session leak on every failure path.** `archive_when_idle: false` with no
   archive on the `_die` paths stranded a session forever on each failed run —
   which, given #1–#3, was every run. `main()` now archives in a `finally`, and
   the archive helper swallows only its own failure (warns on stderr) so it can
   never mask the original error.

5. **Timeout budget could not fit the job.** Two independent
   `REVIEW_TIMEOUT_SECONDS` deadlines (2 × 1200s = 40 min) inside
   `timeout-minutes: 30` meant the corrective turn was frequently unreachable;
   the job would be killed mid-wait with no summary. Both waits now share a
   single deadline.

6. **Verification was too strict to be true.** The check compared the posted body
   for exact equality with the artifact; any server-side normalization (line
   endings, trailing whitespace) would report a false failure *after* a comment
   had in fact been published. Replaced with a
   `<!-- eumemic-bot-review:<head_sha> -->` marker round-trip plus a non-empty
   `html_url` — which is what actually proves GitHub stored *this run's*
   artifact, and additionally ties the comment to the reviewed SHA.

### Minor

7. **Stale `skip-token-revoke: true`.** It existed because the *session* used to
   post after the job ended. The publisher now runs inside the job, so the
   action's post-step revoke can no longer 401 it; leaving the flag would keep an
   installation token alive for its full hour for no reason. Removed, with a
   comment explaining why.

8. **A silently green run is indistinguishable from a successful one.**
   `continue-on-error: true` is correct (an ops miss must not block merges), but
   it is also exactly how the original misses went unnoticed. Added a
   `$GITHUB_STEP_SUMMARY` note that fires when the token mint or the review step
   fails, so a reader sees "did not post" without opening logs.

9. **The regression test was a source grep.** The committed test asserted
   `'"archive_when_idle": False' in source` — it passes on code that cannot run.
   Replaced with ten behavioral tests.

## Fixes applied (`0d1c1b83`)

- `scripts/eumemic_bot_review.py` — `data` envelope; `/wait` long-poll with
  cursor threading and a shared deadline; socket deadline outliving the
  long-poll; `TimeoutError` translation; content-part block handling; leading
  whitespace tolerated on the heading line; marker-based verification; archive in
  a `finally` on every exit path; `REVIEW_TIMEOUT_SECONDS` documented.
- `.github/workflows/eumemic-bot-review.yml` — dropped `skip-token-revoke`;
  named/ID'd the review step; `REVIEW_TIMEOUT_SECONDS: "1200"` tied by comment to
  `timeout-minutes: 30`; added the unpublished-review job-summary step.
- `tests/unit/test_eumemic_bot_review.py` — rewritten: envelope, content blocks,
  user-message filtering, wait-cursor threading, corrective turn, second-miss
  fatality, exact publish→verify→archive call order, archive-on-failure, GitHub
  non-confirmation, plus a contract test pinning paths, query params and response
  fields against the committed `openapi.json`.
- `docs/eumemic-bot-review.md` — matches the implemented behavior.

`infra/agents/dev-review.json` was reviewed and **not** changed: its dual
contract (workflow child POSTs and calls `return`; foreground session must not
POST and must not `return`, emitting the artifact as a plain assistant message)
is correct, and consistent with the launcher prompt. The `return` tool is only
injected when a session owes an open request (`harness/step_context.py`), so a
foreground launcher session genuinely lacks it — which is the most likely reason
real sessions ended without an artifact even before the launcher bugs.

## Verification

- `uv run pytest -q tests/unit/test_eumemic_bot_review.py` → **10 passed**.
- With `tests/unit/test_reconcile_agents.py` → **51 passed**.
- Adjacent CI-script suites (`test_onboarding_docs_drift`,
  `test_phantom_ref_canary_workflow`, `test_ci_queue_watchdog`) → **19 passed**.
- `ruff check`, `ruff format --check`, `mypy` clean on the touched files.
- Mutation check: temporarily reintroducing the `items` envelope bug fails three
  tests; restored afterwards.
- Workflow YAML parses (`yaml.safe_load`).
- Diff touches only `scripts/eumemic_bot_review.py`, the workflow, its test, and
  the doc. **No product or aios#2384/SMS code is touched.**

## Leftover risk

- **No live proof from this worktree.** There are no aios credentials here (only
  `.env.example`; no `AIOS_URL`/`AIOS_API_KEY`), and `gh` is authenticated as the
  user `eumemic`, not as the eumemic-bot App. Real proof requires the repo
  secrets (`AIOS_API_KEY`, `EUMEMIC_BOT_PRIVATE_KEY`, `DEV_REVIEW_AGENT_ID`) and
  a PR-triggered Action run — i.e. it can only be obtained after this branch is
  pushed and a PR opened, from that PR's own `eumemic-bot review` run. The proof
  to look for in the step log is the line
  `posted and verified ### Code review: <comment URL>`.
- **The live agent's prompt lags the merge.** `infra/agents/dev-review.json` only
  reaches the live agent via `reconcile-agents` after merge to `master`, so the
  first PR run exercises the new launcher against the *old* agent contract. The
  launcher tolerates that: a session that ends without the artifact gets one
  corrective turn asking for it as a plain message.
- **`continue-on-error: true` still keeps the check green** when nothing is
  published. That is deliberate (ops misses must not block merges); the job
  summary is the compensating signal, and it is a summary, not an alert — nobody
  is paged.
- **Model compliance is not enforceable.** If the model never emits a
  `### Code review` heading even after the corrective turn, the launcher fails
  loudly and posts nothing. That is the intended failure mode, not a regression.
- **`DEV_REVIEW_AGENT_ID` is read from `secrets`** while `EUMEMIC_BOT_APP_ID` is
  a variable; if that secret is unset the launcher skips rather than failing.
  Unchanged from the reviewed commit and out of scope, but worth confirming it is
  actually populated before expecting a run to post.

REVIEW_DONE
