# Done

Root cause: GitHub Actions places `_runner_file_commands` paths (including
`step_summary` and `artifacts`) in environment variables other than
`GITHUB_OUTPUT`, `GITHUB_ENV`, and `GITHUB_PATH`. The test therefore observed a
runner-owned control path even though the named variables were stripped.

Fix: `_agent_command` now strips all five Actions control variables
(`GITHUB_OUTPUT`, `GITHUB_ENV`, `GITHUB_PATH`, `GITHUB_STEP_SUMMARY`, and
`GITHUB_STATE`) and removes inherited variables whose values contain
`file_commands` (case-insensitive). The unit test uses an explicit runner-env
dict, including an unrelated key carrying a runner path, with no ambient-env
substring assertion. The launcher continues reading `GITHUB_OUTPUT` from its
parent environment.

Validation: `uv run pytest -q tests/unit/test_eumemic_bot_review.py -k
'agent_cannot_reach_the_actions_control_files or
stripping_github_output_does_not_break_the_publication_signal'` — 4 passed.

Branch is up to date with `origin/master`; nothing pushed.

Root cause: trigger fires provision the owning session through the same registry path, but the old credential interception was keyed to the IPv4 addresses returned by one `getent ahostsv4` sample. `api.github.com` rotates addresses and can return an unsampled A record on the curl re-resolution. Limited mode dropped that flow; Unrestricted mode allowed it to bypass the proxy, so the recorder stayed empty. This is not an IPv6-only failure.

Evidence: `src/aios/sandbox/setup.py` documented the live #2042 residual, including the exact Limited/Unrestricted behavior. The trigger runner calls `get_or_provision(..., pool=pool)`, and session provisioning builds the credential proxy from the bound vaults; the missing guarantee was name-based interception in the sandbox DNS path.

Fix: integrated the #2042 product implementation. Each credential proxy now owns a worker-controlled DNS resolver that answers credential names with sentinel `169.254.53.53`; sandbox DNS is redirected to it, and the sentinel is DNATed to the secret-egress proxy. This removes A-record sampling/re-resolution bypasses for both Limited and Unrestricted trigger fires and fails closed if resolver/DNAT setup or verification is incomplete. Added the compatibility `is_run_owner_id` helper required by the current tree.

Verification: `uv run pytest -q tests/unit/sandbox/test_credential_dns.py tests/unit/test_networking.py` — **130 passed**. Docker trigger e2e was not run locally.

No push performed.
