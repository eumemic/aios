# Done

Root cause: GitHub Actions places `_runner_file_commands` paths (including
`step_summary` and `artifacts`) in environment variables other than
`GITHUB_OUTPUT`, `GITHUB_ENV`, and `GITHUB_PATH`. The test therefore observed a
runner-owned control path even though the named variables were stripped.

Fix: `_agent_command` now removes inherited variables whose values contain
`file_commands` (case-insensitive), in addition to stripping the well-known
control-variable names. The unit test covers an unrelated env key carrying a
runner path. The launcher continues reading `GITHUB_OUTPUT` from its parent
environment.

Validation: `uv run pytest -q tests/unit/test_eumemic_bot_review.py -k
'agent_cannot_reach_the_actions_control_files or
stripping_github_output_does_not_break_the_publication_signal'` — 4 passed.

Branch rebased onto `origin/master`; nothing pushed.
