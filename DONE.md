# Done

The eumemic-bot review workflow now mints its GitHub App installation token only after the coding agent has exited.

## Commits

- `27faf31a` — split the review launcher and workflow into agent and publish phases, with token minting between them

The existing gVisor and coding-harness commits remain in history, including `789174bc`, `ffef992c`, `e7f73af7`, `a3a969ef`, `a65dcbb2`, and `2a008bdb`.

## Mint-after-agent enforcement

The workflow runs `eumemic_bot_review.py agent` with routed proxy keys and no `GH_TOKEN`. This mode rejects a set `GH_TOKEN`, pins the checkout, removes the persisted checkout credential, launches the existing coding harness (including Codex with `--sandbox danger-full-access`), waits for it to exit, and only then writes the `### Code review` artifact into the workspace.

After that step has completed successfully, `actions/create-github-app-token` mints the eumemic-bot installation token. A separate `eumemic_bot_review.py publish` process receives `GH_TOKEN`, reads the completed artifact, posts it, and verifies the run-specific marker. The publish process never launches a coding agent.

## Verification

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py
29 passed

$ uv run pytest tests/unit -q
6156 passed, 4 warnings in 133.33s

$ uv run ruff check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
All checks passed!

$ uv run ruff format --check scripts/eumemic_bot_review.py tests/unit/test_eumemic_bot_review.py
2 files already formatted
```

No push was performed.
