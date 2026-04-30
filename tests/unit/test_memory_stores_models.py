"""Pydantic validation for memory store models.

Pure in-memory, no Postgres, no Docker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.memory_stores import (
    MAX_CONTENT_BYTES,
    MAX_INSTRUCTIONS_CHARS,
    MAX_STORES_PER_SESSION,
    MemoryCreate,
    MemoryStoreResource,
    MemoryUpdate,
    MemoryUpdatePrecondition,
    validate_resources,
)


class TestMemoryCreatePath:
    def test_accepts_simple_path(self) -> None:
        m = MemoryCreate(path="/foo.md", content="x")
        assert m.path == "/foo.md"

    def test_accepts_nested_path(self) -> None:
        MemoryCreate(path="/a/b/c.md", content="x")

    def test_accepts_path_with_spaces(self) -> None:
        MemoryCreate(path="/has spaces.md", content="x")

    def test_rejects_no_leading_slash(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="foo.md", content="x")

    def test_rejects_empty_segment_double_slash(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="/foo//bar", content="x")

    def test_rejects_trailing_slash(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="/foo/", content="x")

    def test_rejects_root_only(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="/", content="x")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="/foo\x00bar", content="x")

    @pytest.mark.parametrize(
        "path",
        [
            "/..",
            "/../foo",
            "/foo/..",
            "/foo/../bar",
            "/.",
            "/./foo",
            "/foo/.",
            "/foo/./bar",
            "/a/b/../c",
        ],
    )
    def test_rejects_traversal_segments(self, path: str) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path=path, content="x")

    def test_accepts_dotted_filenames(self) -> None:
        # `.foo`, `..foo`, `...` are valid filenames — only `.` and `..` as
        # full segments are forbidden.
        MemoryCreate(path="/.hidden", content="x")
        MemoryCreate(path="/..foo", content="x")
        MemoryCreate(path="/...", content="x")
        MemoryCreate(path="/a/.b/c..d", content="x")


class TestMemoryCreateContent:
    def test_at_cap_is_ok(self) -> None:
        MemoryCreate(path="/x", content="x" * MAX_CONTENT_BYTES)

    def test_above_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCreate(path="/x", content="x" * (MAX_CONTENT_BYTES + 1))

    def test_unicode_byte_count(self) -> None:
        # 'é' encodes as 2 bytes; should count by bytes, not chars.
        boundary = "é" * (MAX_CONTENT_BYTES // 2)
        MemoryCreate(path="/x", content=boundary)
        with pytest.raises(ValidationError):
            MemoryCreate(path="/x", content=boundary + "é")


class TestMemoryUpdate:
    def test_requires_at_least_one_field(self) -> None:
        with pytest.raises(ValidationError):
            MemoryUpdate()

    def test_content_only(self) -> None:
        MemoryUpdate(content="hello")

    def test_path_only(self) -> None:
        MemoryUpdate(path="/new.md")

    def test_with_precondition(self) -> None:
        MemoryUpdate(
            content="hello",
            precondition=MemoryUpdatePrecondition(type="content_sha256", content_sha256="0" * 64),
        )

    def test_precondition_bad_sha(self) -> None:
        with pytest.raises(ValidationError):
            MemoryUpdatePrecondition(type="content_sha256", content_sha256="too short")


class TestSessionResourceCap:
    def _resource(self, suffix: str) -> MemoryStoreResource:
        return MemoryStoreResource(type="memory_store", memory_store_id=f"memstore_{suffix}")

    def test_at_cap_is_ok(self) -> None:
        resources = [self._resource(f"{i:024d}") for i in range(MAX_STORES_PER_SESSION)]
        validate_resources(resources)

    def test_above_cap_rejected(self) -> None:
        resources = [self._resource(f"{i:024d}") for i in range(MAX_STORES_PER_SESSION + 1)]
        with pytest.raises(ValueError):
            validate_resources(resources)

    def test_duplicate_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_resources([self._resource("a" * 24), self._resource("a" * 24)])

    def test_instructions_cap(self) -> None:
        with pytest.raises(ValidationError):
            MemoryStoreResource(
                type="memory_store",
                memory_store_id="memstore_x",
                instructions="x" * (MAX_INSTRUCTIONS_CHARS + 1),
            )
