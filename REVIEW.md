# Uncorrelated review — eumemic-bot → local coding-agent harness (`98ccd9d5`)

Reviewer: Claude Opus 5, branch `eumbotcarev` (worktree `/workspace/aios-eumbotcarev`).
Commit under review: `98ccd9d5` ("feat(ci): run eumemic-bot PR reviews via local coding-agent harness"),
implemented by gpt-5.6-sol / Codex on `eumbotca`.
Fixes committed locally as `71ef58c5`. Nothing pushed, no PR opened.

## Verdict

**The architecture is right and every structural requirement in TASK.md is met —
but the default path did not work.** Codex, the harness behind the default
`gpt-5.6-sol`, was pointed at the eumemic proxy with `OPENAI_BASE_URL`, which
Codex ignores. Every review on the default model would have 401'd against the
real OpenAI and landed in the "did not post" summary — the exact silent-miss
failure this Action exists to prevent, now on the happy path instead of the edge.

That is fixed, along with four smaller but real defects. **Ready for PR after
`71ef58c5`.**

I verified the harness invocations empirically rather than by reading: all three
CLIs (codex-cli 0.152.1, claude 2.1.263, pi 0.73.1) are installed in this
environment, so each command was run against a local stand-in proxy that logs the
request line, the `Authorization` header, and the body.

## Issues found

### Fatal

**1. Codex never reaches oai-proxy — the default model is dead on arrival.**
`_agent_command` set `OPENAI_API_KEY` + `OPENAI_BASE_URL` and relied on Codex's
built-in `openai` provider honouring the latter. It does not. Running the exact
argv the launcher built:

```
ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket:
      HTTP error: 401 Unauthorized, url: wss://api.openai.com/v1/responses
ERROR: unexpected status 401 Unauthorized: Missing bearer or basic authentication
       in header, url: https://api.openai.com/v1/responses
```

Note the second line: the built-in provider does not even send the key from
`OPENAI_API_KEY`, because it expects `codex login` credentials under
`CODEX_HOME`. The proxy host is never contacted. This is not a "might not work in
CI" — it is reproducible offline, and it takes out the documented default.

Fix: declare the proxy as a provider and select it —
`-c model_provider=eumemic_oai_proxy` plus a `model_providers.eumemic_oai_proxy`
table carrying `base_url`, `env_key="OPENAI_API_KEY"`, `wire_api="responses"`.
Re-running the launcher-built argv against the stand-in proxy after the fix:

```
POST /v1/responses AUTH='Bearer <OAI_PROXY_API_KEY>'
```

The Claude Code and Pi paths were checked the same way and were **correct as
written** — `ANTHROPIC_BASE_URL` is honoured, the generated `models.json` under
`PI_CODING_AGENT_DIR` is picked up, `pi`'s `read,grep,find,ls,bash` are all real
tool names, and both harnesses read the prompt from stdin. `codex exec`'s
`--ephemeral`, `--output-last-message`, and trailing `-` are all real and behave
as assumed. Nothing here was guessed wrong except the base-URL mechanism.

### Serious

**2. The installation token was handed to the agent.** `_agent_command` built the
child environment with `os.environ.copy()`, so the eumemic-bot installation
token in `GH_TOKEN` — which can comment and push as the bot — plus the Actions
runtime tokens and all three proxy keys were inherited by a process that reads
PR-authored files (`AGENTS.md`, `CLAUDE.md`, source) and runs shell commands. The
`claude-*` and `grok-*` harnesses have unrestricted network. Fixed by
constructing the child env as a filtered copy and handing back exactly the one
proxy key the routed harness needs.

**3. The agent was never told what the PR base is.** The prompt said "Review the
changes against the PR base using local git history" without naming a base ref or
SHA. Nothing in the checkout identifies it: `ref` is the head SHA, HEAD is
detached, and the repo's default branch is `master` while an agent guessing will
reach for `origin/main`. The agent would have diffed against a guess or reviewed
the whole tree — which quietly undoes the point of the change, since the previous
review path's central defect (`b334ae63`, `REVIEW.md` before this rewrite) was
also "the reviewer is looking at the wrong tree." Fixed: `BASE_SHA` comes from
`pull_request.base.sha`, the prompt names an explicit `git diff <base>...<head>`
range, and `_pin_checkout` requires the base commit locally, fetching it once if
absent rather than letting the agent silently review nothing.

**4. A timed-out review logs nothing at all.** `subprocess.run(capture_output=True)`
buffers for the full 15 minutes; on `TimeoutExpired` the original code discarded
`exc.stdout`/`exc.stderr` and died. The single likeliest failure mode was the one
that left an operator with an empty step log. Now the partial output is emitted
before the `FATAL`.

**5. Artifact extraction took the first heading, not the last.** `### Code review`
is a contract on the agent's *final message*. Codex satisfies that through its own
`--output-last-message` file, but the Claude Code and Pi paths scrape stdout,
which also carries tool activity — a `grep` for the heading, or a quoted earlier
review, would have become the comment body from that point on. Switched to the
last occurrence, which is equivalent for Codex and correct for the other two.

### Nits (fixed)

- All three harnesses were `npm install --global`-ed on every run regardless of
  the routed model: ~3× the install time, and an unrelated publisher hiccup would
  block every review. Now a `case` on `$REVIEW_MODEL` installs one package, and
  an unroutable model fails the install step with a clear message instead of
  reaching the launcher.
- `config_dir.mkdir()` → `exist_ok=True`.
- The workflow lost every explanatory comment in the rewrite, including the
  "must never FAIL the PR check" rationale that explains why `continue-on-error`
  and the summary step are load-bearing rather than sloppy. Restored and updated.

### Confirmed correct, no change needed

- The aios session path is genuinely gone: no `POST /v1/sessions`, no
  `AIOS_API_KEY` / `AIOS_URL` / `DEV_REVIEW_AGENT_ID` / environment resolution
  anywhere in the launcher or workflow. `infra/agents/dev-review.json` is left
  alone, and a repo-wide grep shows no other caller was disturbed.
- Prefix routing matches the Herdr contract exactly, `gpt-5.6-sol` is the default
  in both the workflow (`vars.EUMEMIC_BOT_REVIEW_MODEL || 'gpt-5.6-sol'`) and
  `DEFAULT_MODEL`.
- Head pinning: checkout uses `pull_request.head.sha`, not the synthetic merge
  commit, and the launcher independently re-verifies `HEAD`.
- `continue-on-error` on all three steps, `always()` summary step keyed on all
  three outcomes, `<!-- eumemic-bot-review:<sha> -->` marker appended and
  verified against GitHub's echoed body.
- Budget: 900 s launcher inside a 20 min job, comfortably under the ≤20 min the
  task asked for, and ordered so the launcher's own FATAL fires before a runner
  kill would strip `continue-on-error` and the summary.

## What I fixed

| SHA | Subject |
|---|---|
| `71ef58c5` | `fix(ci): route Codex through oai-proxy via provider config, not OPENAI_BASE_URL` |

Touching `scripts/eumemic_bot_review.py`, `.github/workflows/eumemic-bot-review.yml`,
`docs/eumemic-bot-review.md`, `tests/unit/test_eumemic_bot_review.py`.

## On the tests

The original 11 were not tautologies — they exercised real behaviour — but they
were shaped to the implementation and so could not have caught any of the five
defects above. In particular `test_codex_command_uses_responses_proxy` asserted
`env["OPENAI_BASE_URL"] == "https://oai-proxy.eumemic.ai/v1"`, which is exactly
the assertion that passes while the feature is broken: it pins the variable
Codex ignores.

The suite is now 24 tests, and I mutation-checked the ones that matter rather
than trusting green. Each of these reintroduced defects fails at least one test:
reverting Codex to `OPENAI_BASE_URL`; `env = dict(os.environ)`; first-occurrence
artifact extraction; deleting `BASE_SHA` from the workflow; swallowing timeout
output; installing all three harnesses unconditionally; dropping
`continue-on-error` from the review step.

Also added: a marker-verification-failure case (the silent miss the launcher
exists to catch), missing-key-for-routed-family, missing-artifact, and
`_pin_checkout` covering wrong-tree / fetch-the-base / base-unfetchable.

Two tests were rewritten to patch a new `_git` helper rather than monkeypatching
the shared `subprocess.run` global, which the old `test_main_posts_and_verifies_marker`
did (with a hand-rolled save/restore around it).

## Verification run

- `uv run --frozen pytest -q tests/unit/test_eumemic_bot_review.py` — **24 passed**.
- `uv run ruff check src tests scripts` — clean; `ruff format --check` clean.
- `uv run mypy tests/unit/test_eumemic_bot_review.py` — clean (`scripts/` is
  outside the repo's mypy scope; the launcher is `py_compile`-clean).
- Live harness smokes against a local stand-in proxy: Codex, Claude Code and Pi
  each driven with the launcher's own generated argv and environment; confirmed
  the request reaches the configured base URL with the right bearer, the prompt
  arrives as the user message, and `GH_TOKEN` is absent from the child env.

Not verifiable here, and left for the first real run: that the eumemic proxies
accept these exact wire shapes (Responses API for oai-proxy/xai-proxy,
`x-api-key` for ant-proxy) and that `gpt-5.6-sol` is served under that name.

## Ready for PR

Yes, with `71ef58c5` included. `TASK.md` and `DONE.md` remain untracked.
