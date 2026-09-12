# Done

## Status: NOT fixed — the landed change is inert (see REVIEW.md)

The first pass on this branch diagnosed the empty recorder as curl preferring an
IPv6 (AAAA) answer for `api.github.com` and bypassing the IPv4-only
credential-host DNAT, and forced `curl -4`. Uncorrelated review disproved that
diagnosis; the `-4` flag changes nothing about the observed failure. The MASTER
RED is therefore still open.

- Disproof (three independent facts, any one sufficient):
  - `api.github.com` publishes **no AAAA record** — `getent ahosts
    api.github.com` returns only IPv4, while the `google.com` control on the
    same resolver returns a real AAAA. There is no v6 answer for curl to prefer.
  - The `aios-sandbox` bridge is created `--ipv6=false`
    (`src/aios/sandbox/network.py:61`), so the container holds no global IPv6
    address and no v6 route out.
  - The Limited path additionally applies `ip6tables -P OUTPUT DROP`
    (`_IP6TABLES_LOCKDOWN_LINES`, `src/aios/sandbox/setup.py`), so a v6 bypass
    could not produce the red Limited leg even if a v6 route existed.
  - Corroborating: the run-origin sibling `test_run_env_var_placeholder.py`
    runs the identical curl to the identical host **without** `-4` and is green.

- Actual root cause (evidence-backed, not yet fixed): the #2042 DNS-sampling
  residual, documented in-tree above `_nat_dnat_lines`
  (`src/aios/sandbox/setup.py`) and behaviourally pinned by
  `TestCredentialHostEgressVerdict` (`tests/unit/test_networking.py`). The
  lockdown sidecar installs one `-d <ip>` DNAT per address a single `getent
  ahostsv4` happened to return; curl re-resolves at fire time and a rotating
  pool can hand it an address no DNAT covers. Unrestricted then egresses
  DIRECT to the real host and Limited falls through to `-P OUTPUT DROP` — both
  leave `recorder.requests == []`, which is exactly the reported signature on
  both legs, and explains why a different leg is red on each CI run.

- What is retained: `curl -4` in `_SWAP_COMMAND`, with its comment rewritten to
  state honestly that it is a standing guard (the DNAT is IPv4-only by design,
  so a future AAAA rollout on the credential host would route the placeholder
  around the proxy under Unrestricted), **not** the fix for the observed flake.

- Commits: `4d2d6661` (the `-4` flag), `a57622ac` (superseded DONE.md), plus the
  review commits on `trigswaprev` correcting the comment and this file.
- Branch is 2 ahead / 0 behind `origin/master` @ `0b07495e`; not pushed.
- Verified locally: `uv run ruff check src tests`, `uv run ruff format --check`,
  `uv run mypy src tests`, and `uv run pytest tests/unit/test_networking.py` all
  pass. Docker is genuinely absent here, so the e2e legs remain CI-only; but
  contrary to the earlier note, `uv run pytest` **does** work in this
  environment and the unit suite was run.
