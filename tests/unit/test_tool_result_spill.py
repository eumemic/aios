"""Unit coverage for :func:`aios.sandbox.tool_result_spill.cap_tool_result_content`.

Deterministic, no Docker: a ``tmp_path`` workspace root is monkeypatched onto
the cached settings so the spill file lands somewhere inspectable (mirrors the
``monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)`` pattern in
``test_volumes_attachments.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.sandbox.tool_result_spill import cap_tool_result_content
from aios.sandbox.volumes import ensure_session_attachments_dir

_SESSION_ID = "sess_spill_unit"
_TOOL_CALL_ID = "tc_unit_1"


def _spill_path(session_id: str, tool_call_id: str) -> Path:
    return ensure_session_attachments_dir(session_id) / "tool_results" / f"{tool_call_id}.txt"


async def test_within_cap_returns_unchanged_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "z" * 500

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result == content
    assert not _spill_path(_SESSION_ID, _TOOL_CALL_ID).exists()


async def test_exactly_at_cap_returns_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: ``len(content) == max_chars`` is within the cap (``<=``)."""
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "a" * 1_000

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result == content
    assert not _spill_path(_SESSION_ID, _TOOL_CALL_ID).exists()


async def test_over_cap_returns_stub_and_writes_full_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "b" * 1_500

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result.startswith("[Tool result truncated:")
    assert "/mnt/attachments/tool_results/tc_unit_1.txt" in result
    assert "1,500 characters" in result
    assert "read tool" in result

    spill = _spill_path(_SESSION_ID, _TOOL_CALL_ID)
    assert spill.exists()
    assert spill.read_text(encoding="utf-8") == content


async def test_no_part_temp_left_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The atomic-rename write must leave only the final file, not its
    ``.part`` staging sibling."""
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "c" * 2_000

    await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    spill_dir = _spill_path(_SESSION_ID, _TOOL_CALL_ID).parent
    names = sorted(p.name for p in spill_dir.iterdir())
    assert names == [f"{_TOOL_CALL_ID}.txt"]
