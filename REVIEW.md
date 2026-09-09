# Uncorrelated review — aios#2410 App-token isolation (`botpost2410d` tip `5f5fe433`)

**Verdict: approve with two fixes applied.** The blocking finding is genuinely
closed: the eumemic-bot installation token is minted only after the coding agent
has exited, so it is not in the launcher's process tree — and therefore not in
`/proc/$PPID/environ` — while the agent runs. Two leftovers are fixed on this
review branch; one residual is documented rather than fixed, because closing it
needs the separate identity/container that TASK.md deprioritizes. Nothing blocks
Shepherd's push.

## Requirement verification

**1. No pre-agent mint — PASS.** `.github/workflows/eumemic-bot-review.yml`
orders the steps `checkout → harness → agent → app (mint) → publish`. The
`actions/create-github-app-token` step is gated
`if: steps.harness.outcome == 'success' && steps.agent.outcome == 'success'`, so
the token is not minted at all unless the agent step has already finished
successfully. A failed agent leaves no token on the runner.

**2. Phase split — PASS.**
- `run_agent_phase` (`scripts/eumemic_bot_review.py:331`) `_die`s if `GH_TOKEN`
  is set, pins the checkout, drops the persisted `http.*.extraheader`
  credential, runs the harness through `subprocess.run` (which waits for exit),
  and only then writes `REVIEW_ARTIFACT_PATH`. The write is unconditionally
  after the child has been reaped — `run_agent` returns a string, so there is no
  path that publishes before the agent has terminated.
- The agent step's `env:` contains no `GH_TOKEN` and no `GITHUB_TOKEN`; GitHub
  does not export `GITHUB_TOKEN` to `run:` steps implicitly, so the agent phase
  has no repo-write credential in env or in `.git/config`.
- `run_publish_phase` reads the completed artifact and POSTs it. It contains no
  call into `run_agent`/`_agent_command`; the publish step is a separate OS
  process that starts after the mint.
- Tests cover the ordering, not just the endpoints:
  `test_workflow_mints_only_after_agent_exits_and_never_gives_agent_gh_token`
  (step index `agent < app < publish`, argv suffixes, `GH_TOKEN` absent from the
  agent env / bound to `steps.app.outputs.token` in publish),
  `test_agent_phase_rejects_gh_token_before_launch` (fails the test if the agent
  is launched at all), `test_agent_phase_writes_artifact_after_agent_returns`
  (asserts the artifact does not exist while the agent runs), and
  `test_main_scrubs_the_git_credential_before_handing_the_tree_to_the_agent`
  (records call order).

**3. Routing / head pin / ops behaviour — PASS.** `model_kind` still routes
`gpt-*→codex`, `claude-*→claude`, `grok-*→pi`; the codex argv still carries
`--sandbox danger-full-access` with the explicit `model_providers.*` table
(pinned positionally by
`test_codex_routes_through_an_explicit_provider_not_openai_base_url`).
`_pin_checkout` still refuses a tree that is not `HEAD_SHA` and still fetches a
missing `BASE_SHA`. `continue-on-error` is on all four ids and the `always()`
job-summary step names all four outcomes — asserted by
`test_workflow_never_fails_the_pr_check_on_an_ops_miss`. `REVIEW_TIMEOUT_SECONDS
900 < timeout-minutes 20` is still asserted.

**4. Base and gVisor — PASS.** `origin/master` is an ancestor of `HEAD` (no
rebase outstanding). `d1da0cd9` (the #2410 tip) is an ancestor, and the gVisor
product commits `789174bc` / `ffef992c` remain in history.
`git diff ffef992c HEAD -- src/ tests/unit/sandbox/` is **empty** — the product
fix and its tests are untouched by this branch's CI work. No Track G.

**5. DONE.md — accurate.** `27faf31a` and the listed ancestors all resolve, the
described enforcement matches the code, and the quoted results reproduce (29
passed / 6156 passed / ruff clean at `5f5fe433`). Counts move by one with this
review's added test; DONE.md is updated to the post-review numbers.

## The blocking finding, re-tested

The original finding was that scrubbing the *child* environment does not isolate
anything, because the agent can read the parent's environment. That mechanism is
worth stating precisely, because it decides where a fix has to live: `unsetenv`
does **not** rewrite `/proc/<pid>/environ`, which is the exec-time stack mapping.
Confirmed locally:

```text
$ SECRETTEST=leakvalue python3 -c "
import os; os.environ.pop('SECRETTEST', None)
print(b'leakvalue' in open('/proc/self/environ','rb').read())"
True
```

So a secret is withheld from the agent only if the *step* never receives it —
removing it later, in either process, is cosmetic. Mint-after-agent satisfies
that for `GH_TOKEN`: at agent time the token does not exist anywhere on the
runner. This is the right shape of fix, and the implementer took it.

## Findings

**F1 — the agent step still received all three proxy secrets (medium; fixed).**
`_STRIPPED_ENV` removes the two non-routed proxy keys from the agent's own
environment "so each harness gets back exactly the one proxy key it needs" — but
the workflow passed `OAI_`, `ANT_` and `XAI_PROXY_API_KEY` into the agent step
unconditionally, so all three stayed readable off `/proc/$PPID/environ` for the
whole run. That is the same defeat as the blocking finding, one severity down
(model-proxy budget, not repo write), and it made the stripping list read as a
guarantee it could not provide. TASK.md §2(a) asks for the routed key(s) only.

Fixed: the harness-install step now emits a `family` output (`oai`/`ant`/`xai`)
alongside the package it installs, and each proxy secret is passed as
`${{ steps.harness.outputs.family == '<fam>' && secrets.<FAM>_PROXY_API_KEY || '' }}`.
The unrouted two arrive as empty strings, which `_proxy_key` already treats as
unset. `test_workflow_gives_the_agent_step_only_the_routed_proxy_secret` pins
both halves so the routing and the gate cannot drift apart.

**F2 — docs described the superseded single-phase flow (low; fixed).**
`docs/eumemic-bot-review.md` still said the launcher "posts it with a short-lived
installation token", and presented environment stripping as the protection — the
claim this whole change exists to retire. Rewritten: the three-step
agent → mint → publish sequence, a "Why the token is minted after the agent"
section that states the `/proc` mechanism explicitly, and the accepted residual
below. The `danger-full-access` bullet no longer claims env stripping as its
mitigation. Added the same reasoning as comments at the workflow's agent step and
above `_STRIPPED_ENV`, so a future "simplification" back to one phase has to read
why it cannot be.

**F3 — a hostile agent can still tamper with the later publish step (residual;
accepted, not fixed).** `--sandbox danger-full-access` gives the agent an
unsandboxed shell as the runner user (with passwordless sudo on GitHub-hosted
runners). It can therefore modify anything the publish step later touches —
including `scripts/eumemic_bot_review.py` in the workspace, which the publish
step executes *with* `GH_TOKEN` — or scrape the runner worker for
`EUMEMIC_BOT_PRIVATE_KEY`. No filesystem-side mitigation holds against that
(copying the launcher to `RUNNER_TEMP` first is writable by the same agent), so
adding one would buy false assurance. The sound fix is a separate UID or
container, which TASK.md explicitly deprioritizes.

This does **not** reopen the required property, which is about the token existing
during the agent run: it does not. What remains is an active-tampering path, not
a credential sitting in readable memory, and it is bounded — the job is
ephemeral, `pull_request` from forks gets no secrets (so the agent's input is
collaborator-authored), and the workflow token is `contents: read`. Recorded in
the docs as accepted rather than silently carried.

## Verification

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py
30 passed

$ uv run pytest tests/unit -q -n 4
6157 passed, 9 warnings in 52.90s

$ uv run mypy scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
Success: no issues found in 2 source files

$ uv run ruff check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
All checks passed!
$ uv run ruff format --check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
2 files already formatted

$ git diff ffef992c HEAD -- src/ tests/unit/sandbox/    # gVisor fix untouched
(empty)
```

No push, no merge, no PR. Branch `botpost2410drev`.
