"""Unit tests for ``aios.config.Settings`` validators.

Tests live in their own file because ``test_cli_dev.py`` covers the dev
bootstrap surface; this file covers process-load invariants enforced at
``Settings()`` construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest


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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "~/aios/workspaces")

    with pytest.raises(ValidationError, match=r"does not expand '~'"):
        Settings(_env_file=(str(secrets),))


def test_workspace_root_accepts_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absolute paths pass through unchanged."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/var/lib/test")

    s = Settings(_env_file=(str(secrets),))
    assert s.workspace_root == Path("/var/lib/test")


def test_workspace_root_default_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-coded default in ``Settings`` must satisfy its own validator."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_GITHUB_CLONE_CACHE_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_cache_timeout_seconds == 300.0


def test_model_call_deadline_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_MODEL_CALL_DEADLINE_S", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.model_call_deadline_s == 900.0


def test_model_call_deadline_rejects_step_budget_or_above(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import ValidationError

    from aios.config import Settings
    from aios.harness.loop import _JOB_TIMEOUT_S

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_MODEL_CALL_DEADLINE_S", str(_JOB_TIMEOUT_S))

    with pytest.raises(ValidationError, match="AIOS_MODEL_CALL_DEADLINE_S"):
        Settings(_env_file=(str(secrets),))


def test_github_clone_session_timeout_below_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of issue #697: the per-session clone budget must
    fit strictly inside the harness step budget so a hung clone doesn't
    burn the full harness turn.
    """
    from aios.config import Settings
    from aios.harness.loop import _JOB_TIMEOUT_S

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.github_clone_session_timeout_seconds < _JOB_TIMEOUT_S


def test_github_clone_session_timeout_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS`` overrides the default."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", "960")

    with pytest.raises(ValidationError, match="must be strictly less than"):
        Settings(_env_file=(str(secrets),))


def test_github_clone_session_timeout_mirror_matches_harness_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config-module mirror of ``_JOB_TIMEOUT_S`` must stay in sync with
    the harness's own constant — otherwise the validator above starts
    rejecting (or admitting) misconfigurations using a stale bound.
    """
    from aios.config import _HARNESS_STEP_TIMEOUT_S
    from aios.harness.loop import _JOB_TIMEOUT_S

    assert _HARNESS_STEP_TIMEOUT_S == _JOB_TIMEOUT_S


def test_sandbox_snapshot_budget_bytes_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-session snapshot budget (durable session sandboxes) defaults to
    4 GiB — over budget at teardown triggers flatten, never a refusal."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_snapshot_budget_bytes == 4 * 1024 * 1024 * 1024


def test_sandbox_snapshot_budget_bytes_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_SANDBOX_SNAPSHOT_BUDGET_BYTES`` sets the global per-session budget."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.workflow_max_inflight_children_per_run == 8


def test_workflow_max_inflight_children_per_run_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN`` overrides the default."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
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
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))
