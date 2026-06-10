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


def test_github_clone_session_timeout_below_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of issue #697: the per-session clone budget must
    fit strictly inside the harness step budget so a hung clone doesn't
    burn the full 5-minute turn.
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
    monkeypatch.setenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", "400")

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


def test_sandbox_disk_bytes_default_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk cap (issue #725) defaults to ``None`` so current behavior
    (host default, unbounded) is unchanged unless the operator opts in."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_SANDBOX_DISK_BYTES", raising=False)

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_disk_bytes is None


def test_sandbox_disk_bytes_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``AIOS_SANDBOX_DISK_BYTES`` sets the global writable-layer cap."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_SANDBOX_DISK_BYTES", str(4 * 1024 * 1024 * 1024))

    s = Settings(_env_file=(str(secrets),))
    assert s.sandbox_disk_bytes == 4 * 1024 * 1024 * 1024


def test_sandbox_disk_bytes_rejects_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the 10 MiB floor the cap can't fit the image's own base size,
    so Settings construction rejects it loudly rather than provisioning a
    container that can't start."""
    from pydantic import ValidationError

    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_SANDBOX_DISK_BYTES", "1024")

    with pytest.raises(ValidationError):
        Settings(_env_file=(str(secrets),))
