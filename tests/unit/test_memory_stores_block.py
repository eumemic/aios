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
        access=access,  # type: ignore[arg-type]
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
    assert "Check memory first" in block
    assert "Write early, write often" in block
    assert "What not to bother saving" in block
    assert "Never save" in block
    assert "memory_precondition_failed_error" in block
    assert "content_sha256" in block
    assert "100 KiB" in block
    assert "Secrets" in block
    assert "credentials" in block
    assert "`write`" in block
    assert "`edit`" in block
    assert "version log" in block
    assert "Re-check memory when you get stuck" in block
    assert "instructions on those points" in block


def test_playbook_appears_after_per_mount_sections() -> None:
    block = build_memory_stores_block([_echo("scratch"), _echo("notes")])
    last_mount = block.rindex("/mnt/memory/notes")
    playbook_start = block.index("Check memory first")
    assert last_mount < playbook_start
