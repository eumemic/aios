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
from aios.sandbox.tool_result_spill import (
    cap_tool_result_content,
    record_spill_attachment,
)
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

    assert result.content == content
    # No spill → no attachment record to register with the GC.
    assert result.attachment is None
    assert not _spill_path(_SESSION_ID, _TOOL_CALL_ID).exists()


async def test_exactly_at_cap_returns_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: ``len(content) == max_chars`` is within the cap (``<=``)."""
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "a" * 1_000

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result.content == content
    assert result.attachment is None
    assert not _spill_path(_SESSION_ID, _TOOL_CALL_ID).exists()


async def test_over_cap_returns_stub_and_writes_full_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "b" * 1_500

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result.content.startswith("[Tool result truncated:")
    assert "/mnt/attachments/tool_results/tc_unit_1.txt" in result.content
    assert "1,500 characters" in result.content
    assert "read tool" in result.content

    spill = _spill_path(_SESSION_ID, _TOOL_CALL_ID)
    assert spill.exists()
    assert spill.read_text(encoding="utf-8") == content


async def test_over_cap_returns_attachment_record_keyed_for_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spilled result must carry a ``metadata.attachments`` record whose
    ``in_sandbox_path`` matches the path the GC walk reconstructs for the
    file on disk (``/mnt/attachments/tool_results/<id>.txt``). Without this
    the spill file's only reference lives in the content stub, invisible to
    ``list_attachment_paths_for_sessions`` → reaped on the next worker boot
    (#1093)."""
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "b" * 1_500

    result = await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    assert result.attachment is not None
    assert result.attachment["in_sandbox_path"] == "/mnt/attachments/tool_results/tc_unit_1.txt"
    assert result.attachment["size"] == 1_500
    assert result.attachment["filename"] == "tc_unit_1.txt"
    assert result.attachment["source"] == "tool_result_spill"


async def test_no_part_temp_left_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The atomic-rename write must leave only the final file, not its
    ``.part`` staging sibling."""
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    content = "c" * 2_000

    await cap_tool_result_content(_SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000)

    spill_dir = _spill_path(_SESSION_ID, _TOOL_CALL_ID).parent
    names = sorted(p.name for p in spill_dir.iterdir())
    assert names == [f"{_TOOL_CALL_ID}.txt"]


class TestRecordSpillAttachment:
    """``record_spill_attachment`` is the single seam both tool-result append
    sinks use to register a spill file under ``metadata.attachments`` — the
    same convention staged inbounds use — so the existing GC referenced-set
    query protects it (#1093)."""

    def test_none_is_noop(self) -> None:
        data: dict[str, object] = {"role": "tool", "content": "ok"}
        record_spill_attachment(data, None)
        assert "metadata" not in data

    def test_creates_metadata_and_attachments_list(self) -> None:
        data: dict[str, object] = {"role": "tool", "content": "stub"}
        att = {"in_sandbox_path": "/mnt/attachments/tool_results/x.txt"}
        record_spill_attachment(data, att)
        assert data["metadata"] == {"attachments": [att]}

    def test_appends_to_existing_attachments_list(self) -> None:
        prior = {"in_sandbox_path": "/mnt/attachments/echo/prior.png"}
        data: dict[str, object] = {
            "role": "tool",
            "content": "stub",
            "metadata": {"attachments": [prior]},
        }
        att = {"in_sandbox_path": "/mnt/attachments/tool_results/x.txt"}
        record_spill_attachment(data, att)
        assert data["metadata"] == {"attachments": [prior, att]}
