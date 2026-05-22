"""Unit coverage for the absolute-path invariant on ``Settings.workspace_root``.

The rest of the codebase calls ``settings.workspace_root.resolve()`` at use
time, which silently produces CWD-dependent paths if the configured value is
relative. With API and worker processes running from different working
directories (different systemd units, different ``uv run`` invocations), this
can desynchronize the path each process computes — surfacing as a
``ForbiddenError`` on the bind-mount boundary rather than a startup failure.

These tests pin the validator that closes the gap at config load time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_workspace_root_rejects_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "./workspaces")
    with pytest.raises(ValidationError, match=r"workspace_root must be an absolute path"):
        Settings()


def test_workspace_root_rejects_parent_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "../workspaces")
    with pytest.raises(ValidationError, match=r"workspace_root must be an absolute path"):
        Settings()


def test_workspace_root_rejects_bare_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "workspaces")
    with pytest.raises(ValidationError, match=r"workspace_root must be an absolute path"):
        Settings()


def test_workspace_root_accepts_absolute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/var/lib/aios/workspaces")
    s = Settings()
    assert s.workspace_root == Path("/var/lib/aios/workspaces")


def test_workspace_root_default_is_absolute() -> None:
    """The hardcoded default must satisfy the new validator."""
    from aios.config import Settings

    s = Settings()
    assert s.workspace_root.is_absolute()
