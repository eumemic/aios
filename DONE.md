# Done

Ported the #2404 local coding-agent review harness onto the existing gVisor branch without changing or dropping its product commits.

## Commits

- `e7f73af7` — port the local coding-agent harness, workflow, documentation, and unit tests
- `a3a969ef` — route Codex through oai-proxy using an explicit Responses API provider
- `a65dcbb2` — strip `OPENAI_BASE_URL` from the agent subprocess environment
- `2a008bdb` — run Codex with `--sandbox danger-full-access` on GitHub Actions

The original gVisor commits remain in history: `789174bc`, `ffef992c`, `2fa61ffe`, and `bf061a6c`.

## Sandbox argv

The Codex command changed from `codex exec ... --sandbox read-only ...` to `codex exec ... --sandbox danger-full-access ...`. This avoids the bubblewrap loopback setup that GitHub-hosted runners reject. The ephemeral publisher job continues to strip GitHub and Actions credentials from the coding-agent subprocess; Claude and Pi routing is unchanged.

## Verification

```text
$ uv run pytest -q tests/unit/test_eumemic_bot_review.py
........................                                                 [100%]
24 passed in 7.90s
```

After Shepherd force-pushes this branch to `gvisorgrn`, the review Action should run the new harness on the new HEAD and post a verified `### Code review` comment as `eumemic-bot`.
