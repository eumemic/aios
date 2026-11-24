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


# WP-25 X7: ``AIOS_*`` keys in .env.example that are NOT server Settings
# fields. Each value is a mandatory reason naming the consumer.
_X7_ENV_KEYS_OUTSIDE_SETTINGS: dict[str, str] = {
    "AIOS_API_KEY": (
        "client-side key read by the aios CLI (src/aios/cli/runtime.py) and "
        "scripts/dev-bootstrap.sh, not by the server Settings model"
    ),
}


def _x7_env_example_keys() -> list[str]:
    """Every non-comment ``AIOS_*=`` key declared in .env.example."""
    keys: list[str] = []
    for line in _read(".env.example").splitlines():
        stripped = line.strip()
        if stripped.startswith("AIOS_") and "=" in stripped:
            keys.append(stripped.split("=", 1)[0])
    return keys


def _x7_env_example_value(key: str) -> str | None:
    for line in _read(".env.example").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1]
    return None


def _x7_settings_env_names() -> set[str]:
    """Env names accepted by ``Settings`` (prefix + field name, plus aliases)."""
    from aios.config import Settings

    prefix = str(Settings.model_config.get("env_prefix", ""))
    names: set[str] = set()
    for field_name, info in Settings.model_fields.items():
        names.add(f"{prefix}{field_name}".upper())
        for alias in (info.alias, info.validation_alias):
            if isinstance(alias, str):
                names.add(alias.upper())
            elif alias is not None:
                for choice in getattr(alias, "choices", []):
                    if isinstance(choice, str):
                        names.add(choice.upper())
    return names


def test_env_example_keys_map_to_settings_fields() -> None:
    """WP-25 X7: every ``AIOS_*`` key in .env.example maps to a Settings field."""
    keys = _x7_env_example_keys()
    accepted = _x7_settings_env_names()
    unknown = sorted(
        k for k in keys if k.upper() not in accepted and k not in _X7_ENV_KEYS_OUTSIDE_SETTINGS
    )
    assert not unknown, (
        f".env.example declares AIOS_* keys with no Settings field: {unknown}. "
        "add a Settings field, delete the key, or allowlist it with a reason "
        "in _X7_ENV_KEYS_OUTSIDE_SETTINGS."
    )
    for key, reason in _X7_ENV_KEYS_OUTSIDE_SETTINGS.items():
        assert reason.strip(), f"allowlist entry {key} needs a non-empty reason"
    stale = sorted(k for k in _X7_ENV_KEYS_OUTSIDE_SETTINGS if k not in keys)
    assert not stale, f"_X7_ENV_KEYS_OUTSIDE_SETTINGS has stale entries: {stale}"


def test_env_example_declares_instance_id() -> None:
    """WP-25 X7: AIOS_INSTANCE_ID is documented with its Settings default."""
    from aios.config import Settings

    assert (
        _x7_env_example_value("AIOS_INSTANCE_ID")
        == Settings.model_fields["instance_id"].get_default()
    )


def test_env_example_docker_image_matches_settings_default() -> None:
    """WP-25 X7: .env.example's AIOS_DOCKER_IMAGE equals the Settings default."""
    from aios.config import Settings

    default = Settings.model_fields["docker_image"].get_default()
    assert _x7_env_example_value("AIOS_DOCKER_IMAGE") == default, (
        f".env.example AIOS_DOCKER_IMAGE drifted from the config.py default {default!r}"
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


def _x5_unit_pytest_line(text: str) -> str:
    """Return the first line invoking ``pytest tests/unit`` in *text*."""
    for line in text.splitlines():
        if "pytest tests/unit" in line:
            return line.strip()
    return ""


def test_run_checks_pytest_workers_match_ci() -> None:
    """WP-25 X5: run-checks.sh runs the unit suite with CI's ``-n 2``."""
    local = _x5_unit_pytest_line(_read("scripts/run-checks.sh"))
    ci = _x5_unit_pytest_line(_read(".github/workflows/code-validation.yml"))
    msg = (
        "unit pytest worker count drifted between scripts/run-checks.sh "
        f"({local!r}) and .github/workflows/code-validation.yml ({ci!r}); "
        "both must carry -n 2."
    )
    assert " -n 2" in f" {local} ", msg
    assert " -n 2" in f" {ci} ", msg
