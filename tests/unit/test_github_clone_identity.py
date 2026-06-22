"""Sandbox clone configures the per-resource git identity (#207).

When a ``github_repository`` attachment carries ``git_user_name`` /
``git_user_email``, the per-session working tree's ``.git/config`` is
stamped via ``git config user.name`` / ``git config user.email`` after
the clone completes.  Subsequent ``git commit`` calls from inside
the sandbox attribute commits to that identity without the agent
seeing or self-correcting from git's "Please tell me who you are"
error.

These tests pin the sandbox-side wiring (``ensure_session_working_tree``)
without spinning up a real container — ``_run_git`` is patched so each
``git`` call appears as a captured argv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aios.sandbox.github_clone import ensure_session_working_tree


@pytest.fixture
def _stub_volumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> Path:
    """Redirect the working-tree dirs into ``tmp_path`` so the test
    doesn't touch the operator's real workspace."""
    repos_root = tmp_path / "session-repos"

    def fake_repos_root(_session_id: str, **kwargs: Any) -> Path:
        repos_root.mkdir(parents=True, exist_ok=True)
        return repos_root

    def fake_working_tree_dir(_session_id: str, resource_id: str, **kwargs: Any) -> Path:
        return repos_root / resource_id

    monkeypatch.setattr("aios.sandbox.github_clone.session_repos_root", fake_repos_root)
    monkeypatch.setattr(
        "aios.sandbox.github_clone.session_repo_working_tree_dir", fake_working_tree_dir
    )
    return repos_root


async def _run_with_captured_git(
    *,
    git_user_name: str | None,
    git_user_email: str | None,
    repos_root: Path,
) -> list[list[str]]:
    """Invoke ``ensure_session_working_tree`` with patched ``_run_git``;
    return the captured argv list (one entry per ``git`` call).
    """
    captured: list[list[str]] = []

    async def fake_run_git(
        argv: list[str], *, cwd: Path | None = None, op: str = "git", **kwargs: Any
    ) -> tuple[int, bytes, bytes]:
        full_argv = ["git", *argv] if cwd is None else ["git", "-C", str(cwd), *argv]
        captured.append(full_argv)
        return 0, b"", b""

    with patch("aios.sandbox.github_clone._run_git", fake_run_git):
        await ensure_session_working_tree(
            session_id="sess_01TEST",
            resource_id="ghrepo_01ABC",
            repo_url="https://github.com/octocat/Hello-World",
            token="ghp_secret",
            cache_dir=repos_root / "cache",
            proxy_url="http://aios-worker:9090/o/r",
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
    return captured


class TestGitIdentityConfig:
    @pytest.mark.asyncio
    async def test_both_fields_set_runs_git_config_for_each(self, _stub_volumes: Path) -> None:
        captured = await _run_with_captured_git(
            git_user_name="Agent JN",
            git_user_email="agent+jn@example.com",
            repos_root=_stub_volumes,
        )

        config_calls = [c for c in captured if "config" in c]
        assert any("user.name" in c and "Agent JN" in c for c in config_calls), (
            f"expected user.name=Agent JN, got {config_calls!r}"
        )
        assert any("user.email" in c and "agent+jn@example.com" in c for c in config_calls), (
            f"expected user.email=agent+jn@example.com, got {config_calls!r}"
        )

    @pytest.mark.asyncio
    async def test_neither_field_set_skips_git_config(self, _stub_volumes: Path) -> None:
        captured = await _run_with_captured_git(
            git_user_name=None,
            git_user_email=None,
            repos_root=_stub_volumes,
        )

        # The remote set-url call uses 'remote', not 'config' — only the
        # identity stamps would emit user.name / user.email tokens.
        all_args = [arg for c in captured for arg in c]
        assert "user.name" not in all_args
        assert "user.email" not in all_args

    @pytest.mark.asyncio
    async def test_only_name_set_stamps_only_name(self, _stub_volumes: Path) -> None:
        captured = await _run_with_captured_git(
            git_user_name="Agent JN",
            git_user_email=None,
            repos_root=_stub_volumes,
        )
        all_args = [arg for c in captured for arg in c]
        assert "user.name" in all_args
        assert "user.email" not in all_args

    @pytest.mark.asyncio
    async def test_config_runs_after_clone_and_remote_set_url(self, _stub_volumes: Path) -> None:
        """Identity stamps must come after the clone (so the working tree
        exists) and after the remote URL scrub (the existing post-clone
        step).  Order matters because clone + scrub can fail and abort
        the whole call before any identity is written.
        """
        captured = await _run_with_captured_git(
            git_user_name="Agent JN",
            git_user_email="agent+jn@example.com",
            repos_root=_stub_volumes,
        )

        labels: list[str] = []
        for c in captured:
            if "clone" in c:
                labels.append("clone")
            elif "remote" in c and "set-url" in c:
                labels.append("remote-set-url")
            elif "config" in c and "user.name" in c:
                labels.append("config-name")
            elif "config" in c and "user.email" in c:
                labels.append("config-email")

        clone_idx = labels.index("clone")
        remote_idx = labels.index("remote-set-url")
        name_idx = labels.index("config-name")
        email_idx = labels.index("config-email")

        assert clone_idx < remote_idx < name_idx
        assert clone_idx < remote_idx < email_idx
