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
    secrets.write_text("AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "./workspaces")

    with pytest.raises(ValidationError, match="absolute"):
        Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]


def test_workspace_root_accepts_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absolute paths pass through unchanged."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/var/lib/test")

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.workspace_root == Path("/var/lib/test")


def test_workspace_root_default_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-coded default in ``Settings`` must satisfy its own validator."""
    from aios.config import Settings

    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_API_KEY=k\nAIOS_VAULT_KEY=v\nAIOS_DB_URL=postgresql://x/y\n")
    monkeypatch.delenv("AIOS_WORKSPACE_ROOT", raising=False)

    s = Settings(_env_file=(str(secrets),))  # type: ignore[call-arg]
    assert s.workspace_root.is_absolute()
