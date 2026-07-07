"""Guard onboarding docs / local check-runner against known drift (issue #1712).

These files are the fresh-contributor onramp; when they drift from the runtime
config or from ``.github/workflows/code-validation.yml`` they silently mislead:

* ``.env.example`` must declare every *required* (no-default) setting, or a
  contributor who copies it still hits a pydantic ``ValidationError`` on first
  run. ``egress_ca_key`` is required and deliberately separate from
  ``vault_key``, so it must be present.
* ``CLAUDE.md`` must not carry the stale ``:8090`` port or a hardcoded test-file
  count that re-rots.
* ``scripts/run-checks.sh`` claims lock-step with the CI lint job, so its lint
  targets and connector test suites must cover the same slack / aios-sdk trees
  CI does.

Modeled on the other ``*_drift.py`` guards: read the source-of-truth files and
assert structural coverage, no network / DB / Docker.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_env_example_declares_required_egress_ca_key() -> None:
    """.env.example must declare the required, no-default AIOS_EGRESS_CA_KEY."""
    env_example = _read(".env.example")
    assert "AIOS_EGRESS_CA_KEY=" in env_example, (
        "config.py declares egress_ca_key as required (no default), but "
        ".env.example omits AIOS_EGRESS_CA_KEY — a contributor copying it hits "
        "a pydantic ValidationError on first run. Add it near AIOS_VAULT_KEY."
    )


def test_claude_md_has_no_stale_port_or_test_count() -> None:
    """CLAUDE.md must not carry the stale :8090 port or a hardcoded test count."""
    claude = _read("CLAUDE.md")
    assert ":8090" not in claude, (
        "CLAUDE.md still references the stale :8090 port; api_port default is 8080."
    )
    assert "160 test" not in claude, (
        "CLAUDE.md hardcodes a test-file count that re-rots; use an "
        "order-of-magnitude phrase instead."
    )


def test_run_checks_matches_ci_lint_and_test_targets() -> None:
    """run-checks.sh must cover the same slack/aios-sdk trees CI's lint+test jobs do."""
    checks = _read("scripts/run-checks.sh")
    # Lint targets: CI lints connectors/slack/src.
    assert "connectors/slack/src" in checks, (
        "run-checks.sh LINT_TARGETS omit connectors/slack/src, but "
        "code-validation.yml lints it — the 'lock-step' comment misleads."
    )
    # Test suites: CI runs the slack and aios-sdk suites.
    assert "connectors/slack/tests" in checks, (
        "run-checks.sh omits the connectors/slack/tests suite that CI runs."
    )
    assert "packages/aios-sdk/tests" in checks, (
        "run-checks.sh omits the packages/aios-sdk/tests suite that CI runs."
    )
