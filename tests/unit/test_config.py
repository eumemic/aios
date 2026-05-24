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
        Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]


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
        Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]


def test_workspace_root_accepts_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absolute paths pass through unchanged."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/var/lib/test")

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.workspace_root == Path("/var/lib/test")


def test_workspace_root_default_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-coded default in ``Settings`` must satisfy its own validator."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_WORKSPACE_ROOT", raising=False)

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
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

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
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

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
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

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.github_clone_session_timeout_seconds < _JOB_TIMEOUT_S


def test_github_clone_session_timeout_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS`` overrides the default."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS", "7")

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.github_clone_session_timeout_seconds == 7.0
