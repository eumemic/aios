"""Tests for the CLI's payload loaders."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aios.cli.files import PayloadError, load_payload, walk_skill_dir


def test_load_from_file(tmp_path: Path):
    p = tmp_path / "body.json"
    p.write_text('{"name": "X", "n": 1}')
    out = load_payload(p, False, None)
    assert out == {"name": "X", "n": 1}


def test_load_from_data_string():
    out = load_payload(None, False, '{"k": true}')
    assert out == {"k": True}


def test_load_from_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"via": "stdin"}'))
    out = load_payload(None, True, None)
    assert out == {"via": "stdin"}


def test_load_no_source_raises():
    with pytest.raises(PayloadError, match="no payload"):
        load_payload(None, False, None)


def test_load_multiple_sources_raises(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(PayloadError, match="only one of"):
        load_payload(p, False, "{}")


def test_load_non_object_raises():
    with pytest.raises(PayloadError, match="must be a JSON object"):
        load_payload(None, False, "[1,2,3]")


def test_load_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(PayloadError, match="invalid JSON"):
        load_payload(p, False, None)


def test_walk_skill_dir(tmp_path: Path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\nhello")
    (skill / "inner").mkdir()
    (skill / "inner" / "helper.py").write_text("print(1)")

    files = walk_skill_dir(skill)
    assert "SKILL.md" in files
    assert files["SKILL.md"].startswith("---")
    assert "inner/helper.py" in files
    assert files["inner/helper.py"] == "print(1)"


def test_walk_skill_dir_missing_skill_md(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "other.md").write_text("x")
    with pytest.raises(PayloadError, match=r"missing SKILL\.md"):
        walk_skill_dir(d)


def test_walk_skill_dir_not_a_directory(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(PayloadError, match="not a directory"):
        walk_skill_dir(f)
