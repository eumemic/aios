"""Tests for ``resolve_memory_target`` against a populated runtime cache.

Pure in-memory: we install a synthetic mount cache via
``runtime.set_session_memory_mounts`` and exercise the resolver.
"""

from __future__ import annotations

import pytest

from aios.harness import runtime
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.tools.memory_intercept import resolve_memory_target

SESSION_ID = "sesn_01ABCDEFGHJKMNPQRSTVWXYZ12"


def _echo(name: str, access: str = "read_write") -> MemoryStoreResourceEcho:
    return MemoryStoreResourceEcho(
        memory_store_id=f"memstore_{name}",
        access=access,  # type: ignore[arg-type]
        instructions="",
        name=name,
        description="",
        mount_path=f"/mnt/memory/{name}",
    )


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    runtime.clear_session_memory_mounts(SESSION_ID)
    yield
    runtime.clear_session_memory_mounts(SESSION_ID)


class TestResolveMemoryTarget:
    def test_path_outside_any_mount(self) -> None:
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("scratch")])
        assert resolve_memory_target(SESSION_ID, "/workspace/foo.md") is None

    def test_path_under_mount_resolves(self) -> None:
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("scratch")])
        target = resolve_memory_target(SESSION_ID, "/mnt/memory/scratch/notes.md")
        assert target is not None
        assert target.store_name == "scratch"
        assert target.store_path == "/notes.md"
        assert target.access == "read_write"

    def test_nested_path_under_mount(self) -> None:
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("scratch")])
        target = resolve_memory_target(SESSION_ID, "/mnt/memory/scratch/a/b/c.md")
        assert target is not None
        assert target.store_path == "/a/b/c.md"

    def test_read_only_propagates(self) -> None:
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("ref", access="read_only")])
        target = resolve_memory_target(SESSION_ID, "/mnt/memory/ref/x")
        assert target is not None
        assert target.access == "read_only"

    def test_no_mounts_for_session(self) -> None:
        # cache empty for this session; even mount-shaped paths return None
        assert resolve_memory_target(SESSION_ID, "/mnt/memory/scratch/x") is None

    def test_path_that_only_shares_prefix(self) -> None:
        # /mnt/memory/scratch_v2/... must NOT match the "scratch" mount.
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("scratch")])
        assert resolve_memory_target(SESSION_ID, "/mnt/memory/scratch_v2/x.md") is None

    def test_picks_correct_mount_when_multiple(self) -> None:
        runtime.set_session_memory_mounts(SESSION_ID, [_echo("a"), _echo("b", access="read_only")])
        ta = resolve_memory_target(SESSION_ID, "/mnt/memory/a/x.md")
        tb = resolve_memory_target(SESSION_ID, "/mnt/memory/b/x.md")
        assert ta is not None and ta.store_name == "a"
        assert tb is not None and tb.store_name == "b"
        assert tb.access == "read_only"
