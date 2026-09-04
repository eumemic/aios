"""Unit tests for ``aios.config.Settings`` validators.

Tests live in their own file because ``test_cli_dev.py`` covers the dev
bootstrap surface; this file covers process-load invariants enforced at
``Settings()`` construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_external_byok_rejects_legacy_inference_env_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_TENANCY_POSTURE", "external_byok")
    monkeypatch.setenv("AIOS_INFERENCE_CREDENTIAL_POLICY", "legacy_env")

    with pytest.raises(ValidationError, match="external_byok requires account_only"):
        Settings(_env_file=(str(secrets),))


def test_inference_credential_policy_defaults_account_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_INFERENCE_CREDENTIAL_POLICY", raising=False)
    settings = Settings(_env_file=(str(secrets),))
    assert settings.inference_credential_policy == "account_only"


def test_dead_pipeline_max_setting_is_not_exposed() -> None:
    from aios.config import Settings

    assert "sandbox_pipeline_max_seconds" not in Settings.model_fields


def test_workspace_root_must_be_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AIOS_WORKSPACE_ROOT=./relative`` fails fast at process load.

    Without enforcement, the API and worker processes can resolve the path
    differently depending on each process's CWD, producing CWD-drift bugs
    that surface much later as ``ForbiddenError`` on every tool call once a
    sandbox recycles.
    """
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "./workspaces")

    with pytest.raises(ValidationError, match="must be an absolute path"):
        Settings(_env_file=(str(secrets),))


def test_workspace_root_error_mentions_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error message mentions that ``~`` is not expanded.

    A common operator mistake is ``AIOS_WORKSPACE_ROOT=~/aios/workspaces``,
    which pathlib stores verbatim — ``Path("~/aios/workspaces").is_absolute()``
    is False. The message names this explicitly so the operator doesn't have
    to relearn it.
    """
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "~/aios/workspaces")

    with pytest.raises(ValidationError, match=r"does not expand '~'"):
        Settings(_env_file=(str(secrets),))


def test_workspace_root_accepts_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absolute paths pass through unchanged."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/var/lib/test")

    s = Settings(_env_file=(str(secrets),))
    assert s.workspace_root == Path("/var/lib/test")


def test_workspace_root_default_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-coded default in ``Settings`` must satisfy its own validator."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_WORKSPACE_ROOT", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.workspace_root.is_absolute()


def test_github_clone_session_timeout_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-session ``git clone --reference --dissociate`` budget defaults to 30s.

    Must be small enough that the harness step timeout (300s) is never the
    instrument that fires on a hung clone — see issue #697.
    """
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_session_timeout_seconds == 30.0


def test_github_clone_cache_timeout_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare-cache clone/fetch budget defaults to 300s — cold-case clones
    of large repos can legitimately take minutes, and the cache lives
    off the per-session critical path."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_GITHUB_CLONE_CACHE_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_cache_timeout_seconds == 300.0


def test_model_call_deadline_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_MODEL_CALL_DEADLINE_S", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.model_call_deadline_s == 900.0


def test_model_call_deadline_rejects_step_budget_or_above(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import ValidationError

    from aios.config import HARNESS_STEP_TIMEOUT_S, Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_MODEL_CALL_DEADLINE_S", str(HARNESS_STEP_TIMEOUT_S))

    with pytest.raises(ValidationError, match="AIOS_MODEL_CALL_DEADLINE_S"):
        Settings(_env_file=(str(secrets),))


def test_github_clone_session_timeout_below_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of issue #697: the per-session clone budget must
    fit strictly inside the harness step budget so a hung clone doesn't
    burn the full harness turn.
    """
    from aios.config import HARNESS_STEP_TIMEOUT_S, Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_session_timeout_seconds < HARNESS_STEP_TIMEOUT_S


def test_github_clone_session_timeout_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS`` overrides the default."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", "7")

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_session_timeout_seconds == 7.0


def test_github_clone_session_timeout_rejects_above_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-session clone budget >= the harness step budget would silently
    defeat issue #697's fix (a hung clone would still burn a whole user
    turn before the step-level cap fires). Settings construction must
    reject the misconfiguration loudly at startup.
    """
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", "960")

    with pytest.raises(ValidationError, match="must be strictly less than"):
        Settings(_env_file=(str(secrets),))


def test_step_timeout_single_source() -> None:
    """The harness step budget is defined once in config and consumed by the
    Settings validators; no second copy exists to drift out of sync."""
    import aios.config as config_mod
    import aios.harness.loop as loop_mod

    assert config_mod.HARNESS_STEP_TIMEOUT_S == 960.0
    # loop imports the same object, not a private duplicate.
    assert not hasattr(loop_mod, "_JOB_TIMEOUT_S")
    assert loop_mod.HARNESS_STEP_TIMEOUT_S is config_mod.HARNESS_STEP_TIMEOUT_S


def test_sandbox_snapshot_budget_bytes_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-session snapshot budget (durable session sandboxes) defaults to
    4 GiB — over budget at teardown triggers flatten, never a refusal."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_snapshot_budget_bytes == 4 * 1024 * 1024 * 1024


def test_sandbox_snapshot_budget_bytes_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES`` sets the global per-session budget."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES", str(8 * 1024 * 1024 * 1024))

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_snapshot_budget_bytes == 8 * 1024 * 1024 * 1024


def test_sandbox_snapshot_budget_bytes_rejects_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the 10 MiB floor the budget can't fit the image's base, so
    Settings construction rejects it loudly."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES", "1024")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))


def test_container_idle_timeout_default_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable session sandboxes raise the idle default 300 → 1800 (§5.10):
    teardown now costs a commit, so keeping an idle container alive is cheap."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_CONTAINER_IDLE_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.container_idle_timeout_seconds == 1800


def test_workflow_max_inflight_children_per_run_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-run wave admission cap (#784) defaults to 8: a single run admits at most
    8 concurrently in-flight ``agent()`` children per step, journaling the rest."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.workflow_max_inflight_children_per_run == 8


def test_workflow_max_inflight_children_per_run_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN`` overrides the default."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN", "3")

    s = Settings(_env_file=(str(secrets),))
    assert s.workflow_max_inflight_children_per_run == 3


def test_workflow_max_inflight_children_per_run_rejects_below_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-run wave cap below 1 would admit nothing and strand every frontier;
    Settings construction rejects it loudly (``ge=1``)."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))


def test_docker_cli_timeout_default_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_SANDBOX_DOCKER_CLI_TIMEOUT_SECONDS", raising=False)
    assert Settings(_env_file=(str(secrets),)).sandbox_docker_cli_timeout_seconds == 30.0
    monkeypatch.setenv("AIOS_SANDBOX_DOCKER_CLI_TIMEOUT_SECONDS", "45")
    assert Settings(_env_file=(str(secrets),)).sandbox_docker_cli_timeout_seconds == 45.0


def _browser_secrets(tmp_path: Path) -> Path:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    return secrets


def test_browser_call_timeout_default_covers_cold_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped default call_timeout must clear the documented cold-open
    floor against the shipped inner-budget defaults, so every deployment
    running defaults never trips the invariant (the bug is dormant otherwise)."""
    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    monkeypatch.delenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AIOS_SANDBOX_BROWSER_PROVISION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AIOS_SANDBOX_BROWSER_TAKEOVER_OPEN_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    named_floor = (
        s.sandbox_browser_provision_timeout_seconds
        + s.sandbox_browser_takeover_open_timeout_seconds
    )
    assert s.sandbox_browser_call_timeout_seconds == 210.0
    assert named_floor == 120.0 + 45
    assert s.sandbox_browser_call_timeout_seconds > named_floor


def test_browser_call_timeout_rejects_below_named_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tightening the 504 timeout below the cold-open floor 504s the caller
    while the worker still completes the open and inserts a viewerless grant
    that wedges the account's browser plane — the documented invariant the
    field description commits to. Settings construction must reject it loudly
    at startup rather than letting the misconfiguration ship to the worker."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    monkeypatch.setenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", "60")

    with pytest.raises(ValidationError, match="AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS"):
        Settings(_env_file=(str(secrets),))


def test_browser_call_timeout_rejects_equal_to_named_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "plus margin" the description appends is deliberately unspecified,
    so a value exactly equal to the named floor is rejected (``must exceed``
    the floor); the boundary case stays operator judgment."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    # Default inner budgets: 120 + 45 = 165. Exactly the floor must be rejected.
    monkeypatch.setenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", "165")

    with pytest.raises(ValidationError, match="must exceed the cold-open floor"):
        Settings(_env_file=(str(secrets),))


def test_browser_call_timeout_accepts_just_above_named_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One second above the named floor passes the validator and is stored —
    the unspecified margin is left as operator judgment, exactly as the
    field description hedges it."""
    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    monkeypatch.setenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", "166")

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_browser_call_timeout_seconds == 166.0


def test_browser_call_timeout_rejects_when_sibling_budgets_raise_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validator reacts to the sibling budgets too, not just the
    call_timeout field: the default 210s is valid against the default
    inner budgets (floor 165s), but raising ``takeover_open`` to 130 makes
    the named floor 120 + 130 = 250s and rejects the same 210s default."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    monkeypatch.delenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("AIOS_SANDBOX_BROWSER_TAKEOVER_OPEN_TIMEOUT_SECONDS", "130")

    with pytest.raises(ValidationError, match="AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS"):
        Settings(_env_file=(str(secrets),))


def test_browser_call_timeout_error_names_env_var_and_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rejection message co-locates the env-var name with the constraint
    (the only operator-facing surface that does — grep across ``*.md`` finds
    zero references to the var), so the message must name both the env var
    and the floor's named components an operator would need to reconcile it."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = _browser_secrets(tmp_path)
    monkeypatch.setenv("AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS", "60")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=(str(secrets),))

    msg = str(exc_info.value)
    assert "AIOS_SANDBOX_BROWSER_CALL_TIMEOUT_SECONDS=60.0" in msg
    assert "provision 120.0s" in msg
    assert "takeover_open 45s" in msg
    assert "165.0s" in msg
    assert "viewerless grant" in msg
