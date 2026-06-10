"""Tests for the memory-stores system-prompt block builder."""

from __future__ import annotations

from aios.harness.memory_stores import (
    augment_with_memory_stores,
    build_memory_stores_block,
)
from aios.models.memory_stores import MemoryStoreResourceEcho


def _echo(
    name: str,
    *,
    access: str = "read_write",
    description: str = "",
    instructions: str = "",
) -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=f"memstore_{name}",
        access=access,
        instructions=instructions,
        name=name,
        description=description,
        mount_path=f"/mnt/memory/{name}",
    )


def test_empty_returns_empty_string() -> None:
    assert build_memory_stores_block([]) == ""


def test_augment_unchanged_when_empty() -> None:
    assert augment_with_memory_stores("base", []) == "base"


def test_block_includes_mount_path_and_access() -> None:
    block = build_memory_stores_block([_echo("scratch")])
    assert "## Memory stores" in block
    assert "### scratch" in block
    assert "/mnt/memory/scratch" in block
    assert "read_write" in block


def test_block_includes_description_and_instructions() -> None:
    block = build_memory_stores_block(
        [_echo("scratch", description="A desc", instructions="check first")]
    )
    assert "A desc" in block
    assert "check first" in block


def test_block_omits_empty_description() -> None:
    block = build_memory_stores_block([_echo("scratch")])
    assert "description:" not in block
    assert "instructions:" not in block


def test_augment_appends_with_double_newline() -> None:
    out = augment_with_memory_stores("base", [_echo("a")])
    assert out.startswith("base\n\n")
    assert "## Memory stores" in out


def test_augment_with_empty_base() -> None:
    out = augment_with_memory_stores("", [_echo("a")])
    assert out.startswith("## Memory stores")


def test_playbook_content_when_stores_attached() -> None:
    block = build_memory_stores_block([_echo("scratch")])
    # Only pin the two substrings that cross-reference code constants — a drift
    # between the model-facing copy and what the harness actually enforces/raises
    # is a real bug: "100 KiB" derives from MAX_CONTENT_BYTES // 1024, and
    # "memory_precondition_failed_error" from MemoryPreconditionFailedError.error_type.
    # The rest of the playbook is prose; its structure/ordering is guarded by
    # test_playbook_appears_after_per_mount_sections, so we don't golden-pin wording here.
    assert "100 KiB" in block
    assert "memory_precondition_failed_error" in block


def test_playbook_appears_after_per_mount_sections() -> None:
    block = build_memory_stores_block([_echo("scratch"), _echo("notes")])
    last_mount = block.rindex("/mnt/memory/notes")
    playbook_start = block.index("Check memory first")
    assert last_mount < playbook_start
