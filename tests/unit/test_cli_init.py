"""Tests for ``aios init`` — scaffold a typed agent-authoring project.

Deterministic, offline, no Docker: drives the command through
``typer.testing.CliRunner`` against a tmp_path.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_init_writes_scaffold_into_empty_dir(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output

    agent_py = target / "agent.py"
    readme = target / "README.md"
    pyproject = target / "pyproject.toml"
    assert agent_py.exists()
    assert readme.exists()
    assert pyproject.exists()
    assert "from aios_sdk.authoring import" in agent_py.read_text(encoding="utf-8")


def test_init_refuses_non_empty_dir(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    target.mkdir()
    (target / "existing.txt").write_text("keep me", encoding="utf-8")
    before = {p.name for p in target.iterdir()}

    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code != 0

    after = {p.name for p in target.iterdir()}
    assert after == before  # nothing new written
    assert (target / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_init_refuses_symlinked_target(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    result = runner.invoke(app, ["init", str(link)])
    assert result.exit_code != 0

    # Nothing written through the symlink into the real dir.
    assert list(real.iterdir()) == []
