# Uncorrelated review — aios#2410 harness port (`botpost2410c` tip `30504a81`)

**Verdict: approve with one fix applied.** The port is faithful, the four
TASK.md requirements are met, and the gVisor product fix is untouched. One real
gap opened by the sandbox change (a credential the env stripping cannot reach)
is fixed on this review branch in `798fb985`. Nothing blocks Shepherd's push.

## Requirement verification

**1. Harness files present, gVisor commits intact — PASS.**
All four files are on the branch and are byte-identical to `origin/eumbotca`
except for the deliberate sandbox change and its doc/test follow-ups:

| File | vs `origin/eumbotca` |
|---|---|
| `scripts/eumemic_bot_review.py` | sandbox hunk only |
| `.github/workflows/eumemic-bot-review.yml` | identical |
| `docs/eumemic-bot-review.md` | sandbox paragraph + "read-only" → "instruct the agent not to modify the checkout" |
| `tests/unit/test_eumemic_bot_review.py` | one added argv assertion |

The `read-only` → `instruct the agent not to modify` doc rewording is correct
rather than cosmetic: the sandbox no longer enforces it, so the claim had to be
downgraded to what the prompt asks. `REVIEW_SCOPE` already carried "Do not
modify the checkout".

All four gVisor commits are ancestors of `HEAD` (`789174bc`, `ffef992c`,
`2fa61ffe`, `bf061a6c`), and `git diff ffef992c HEAD -- src/` is empty — the
product fix was not touched by the port. `tests/unit/sandbox`: 531 passed.

`2fa61ffe`/`bf061a6c` (the wait-read-timeout retry and its tests) were correctly
*superseded* rather than merged: `e7f73af7` deletes the aios-session polling
those commits hardened, and deletes their tests with it. No orphan references —
`grep` for `dev-review`/`DEV_REVIEW` finds only `infra/agents/dev-review.json`
and `reconcile_agents.py`, which the docs explicitly scope out.

**2. Codex sandbox — PASS.** `_agent_command` emits
`codex exec --model <m> --sandbox danger-full-access --ephemeral -c … -`.
No bwrap loopback is required. Claude (`--print`, `ANTHROPIC_BASE_URL`) and Pi
(generated `models.json` + `--provider xai-proxy`) are unchanged from #2404.
`test_codex_routes_through_an_explicit_provider_not_openai_base_url` pins the
argv positionally (`command[command.index("--sandbox") + 1]`), so a silent
revert to `read-only` fails the suite.

**3. Fail-closed publishing — PASS.** `main` POSTs, then requires both an
`html_url` and the run-specific `<!-- eumemic-bot-review:<sha> -->` marker
echoed back, else `_die`. `_github_request` `_die`s on any HTTPError/URLError.
Budget is #2404's: `REVIEW_TIMEOUT_SECONDS: "900"` inside `timeout-minutes: 20`
— no regression to 45m aios sessions. `continue-on-error` on
harness/app/review plus the `always()` job-summary step is intact, and
`test_workflow_pins_head_and_base_and_keeps_no_aios_session_config` asserts
`REVIEW_TIMEOUT_SECONDS < timeout-minutes * 60` and `timeout-minutes <= 20`, so
the ordering that keeps the FATAL (not a runner kill) is pinned.

**4. DONE.md accuracy — PASS.** All eight shas resolve and are ancestors of
`HEAD`. The sandbox-argv explanation matches the code. `uv run pytest -q
tests/unit/test_eumemic_bot_review.py` → **24 passed** as claimed (26 after this
review's two added tests). `ruff check`/`format --check` clean on both touched
files; `mypy tests/unit/test_eumemic_bot_review.py` clean.

## Findings

### F1 — `danger-full-access` exposes the checkout's persisted push credential (fixed, `798fb985`)

`_STRIPPED_ENV` removes `GH_TOKEN`, `GITHUB_TOKEN` and the Actions runtime
tokens from the agent's *environment*, and both the docstring and
`docs/eumemic-bot-review.md` state the agent "must not hold a credential that
can write". But `actions/checkout@v4` defaults to `persist-credentials: true`
and the workflow does not override it, so the workflow `GITHUB_TOKEN` is written
into `.git/config` as `http.https://github.com/.extraheader`. That is a file,
not an env var, and nothing in the launcher touched it.

Under `--sandbox read-only` this was contained on the codex route. Removing the
sandbox gives the default route an unrestricted shell *and* an open network on
the very tree it is reading PR-authored files (`AGENTS.md`, `CLAUDE.md`, source)
from — so the credential became both readable and usable. The claude and pi
routes never had a sandbox, so they carried this already; the sandbox change
just moved it onto the route that actually runs.

Severity is bounded, and worth stating plainly: top-level `permissions:
contents: read` caps the token at read, it dies with the job, and fork PRs get
no secrets, so `actions/create-github-app-token` fails and the review step is
skipped — the path fails closed for untrusted authors. So this is hardening, not
an open hole. It is still a gap between a documented invariant and the code.

Fixed in the launcher rather than the workflow: `_pin_checkout` is the only
consumer of that credential (its fallback fetch of a missing PR base), so
`main` now calls `_drop_persisted_git_credentials()` immediately after pinning
and before `run_agent`. Doing it there instead of `persist-credentials: false`
keeps the base fetch working on a private repo and covers all three routes in
one place. Verified against a real repository (extraheader removed, unrelated
config preserved) plus a mocked ordering assertion.

### F2 — `_github_request` failed the repo's own mypy config (fixed, `798fb985`)

`body: dict | None = None) -> dict` is a `[type-arg]` error under the strict
`disallow_any_generics`. It never failed CI because `code-validation.yml` runs
`mypy src tests packages/… connectors/…` and does not include `scripts/`.
Inherited from #2404; typed as `dict[str, Any]`. `uv run mypy
scripts/eumemic_bot_review.py` is now clean.

## Observations (no action)

- **Harness CLI flags are unverified locally** — `codex --ephemeral`,
  `claude --no-session-persistence`, `pi --no-session`. None of these harnesses
  is installed here. The `bwrap: loopback` failure quoted in TASK.md is itself
  evidence that Codex parsed the full argv (including `--ephemeral` and the two
  `-c` overrides) and got as far as sandbox setup, so the codex route's argv is
  at least accepted. The claude/pi flags are inherited from #2404 and are not
  exercised by the default model.
- **Proxy routing is argv-verified, not end-to-end.** The tests assert the
  provider table (`base_url`, `wire_api="responses"`, `env_key`), which is the
  right thing to unit-test, but no run has yet proven a `gpt-5.6-sol` completion
  through oai-proxy — the previous attempt died at sandbox setup before
  reaching the model. That is what Shepherd's push will actually establish.
- **`_artifact_in` scans for the last heading**, which is correct for the
  stdout-scraping claude/pi routes and harmless for codex's
  `--output-last-message` file. Well covered by
  `test_run_agent_takes_the_final_heading_not_an_echoed_one`.
- A review body over GitHub's 65536-character comment limit would 422 and
  `_die` into the "did not post" summary. Acceptable fail-closed behaviour; not
  worth pre-empting.

## Commands run

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py
26 passed in 3.41s          # 24 before this review's two added tests

$ uv run pytest -q tests/unit/sandbox
531 passed, 2 warnings in 25.83s

$ uv run ruff check scripts tests/unit/test_eumemic_bot_review.py
All checks passed!
$ uv run ruff format --check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
2 files already formatted
$ uv run mypy scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
Success: no issues found
```

Not run: the full `tests/unit` suite and any e2e (no Docker). The change is
confined to `scripts/` and its own test module; `grep` confirms nothing else in
the tree imports the launcher.

## Scope honoured

No Track G. gVisor product fix untouched. Nothing merged, nothing pushed, no PR
opened. Branch stays `botpost2410crev`.
