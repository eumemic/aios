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
    """Keys are prefixed with the root directory's basename.

    The server's ``_extract_skill_metadata`` parses the directory name
    from the ``SKILL.md`` key and rejects bare ``SKILL.md``. Prepending
    ``root.name`` gives it the ``my-skill/SKILL.md`` shape it requires.
    """
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\nhello")
    (skill / "inner").mkdir()
    (skill / "inner" / "helper.py").write_text("print(1)")

    files = walk_skill_dir(skill)
    assert "my-skill/SKILL.md" in files
    assert files["my-skill/SKILL.md"].startswith("---")
    assert "my-skill/inner/helper.py" in files
    assert files["my-skill/inner/helper.py"] == "print(1)"
    # A bare ``SKILL.md`` key would be rejected by the server.
    assert "SKILL.md" not in files


def test_walk_skill_dir_matches_server_contract(tmp_path: Path):
    """End-to-end shape check: the dict returned by ``walk_skill_dir``
    round-trips through the server's ``_extract_skill_metadata`` without
    the *"SKILL.md must be inside a directory"* failure that motivated
    the fix."""
    from aios.services.skills import _extract_skill_metadata

    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: does a thing\n---\nbody")
    (skill / "helper.py").write_text("print(1)")

    files = walk_skill_dir(skill)
    directory, name, description, normalized = _extract_skill_metadata(files)
    assert directory == "my-skill"
    assert name == "my-skill"
    assert description == "does a thing"
    # Normalized keys have the directory prefix stripped.
    assert "SKILL.md" in normalized
    assert "helper.py" in normalized


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
