"""Unit tests for the SandboxPath resolver + Attachment validation.

Resolution covers the three legitimate prefixes (``/workspace/``,
``/mnt/attachments/``, the bare directories), traversal escapes
(``..``), and the dispatch-time integration: a tool with
``attachments: list[SandboxPath]`` receives :class:`Path` objects and
out-of-tree paths surface as a tool error result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aios_connector_http import (
    Attachment,
    AttachmentError,
    HttpConnector,
    SandboxPath,
    tool,
)
from aios_connector_http.sandbox import resolve_sandbox_path, workspace_root


class TestResolveSandboxPath:
    def test_workspace_path_resolves_to_session_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "sess_1").mkdir()
        (tmp_path / "sess_1" / "photo.jpg").write_bytes(b"x")
        out = resolve_sandbox_path(
            session_id="sess_1",
            sandbox_path="/workspace/photo.jpg",
            root=tmp_path,
        )
        assert out == (tmp_path / "sess_1" / "photo.jpg").resolve()

    def test_attachments_path_resolves_under_attachments_root(self, tmp_path: Path) -> None:
        (tmp_path / "_attachments" / "sess_1").mkdir(parents=True)
        out = resolve_sandbox_path(
            session_id="sess_1",
            sandbox_path="/mnt/attachments/inbound.png",
            root=tmp_path,
        )
        assert out == (tmp_path / "_attachments" / "sess_1" / "inbound.png").resolve()

    def test_traversal_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "sess_1").mkdir()
        out = resolve_sandbox_path(
            session_id="sess_1",
            sandbox_path="/workspace/../../../etc/passwd",
            root=tmp_path,
        )
        assert out is None

    def test_unmapped_prefix_returns_none(self, tmp_path: Path) -> None:
        out = resolve_sandbox_path(
            session_id="sess_1",
            sandbox_path="/etc/passwd",
            root=tmp_path,
        )
        assert out is None

    def test_workspace_prefix_collision_isolated(self, tmp_path: Path) -> None:
        """``/workspaces/foo`` (note plural) must NOT match ``/workspace/``."""
        out = resolve_sandbox_path(
            session_id="sess_1",
            sandbox_path="/workspaces/foo",
            root=tmp_path,
        )
        assert out is None

    def test_workspace_root_env_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIOS_WORKSPACE_ROOT", raising=False)
        assert workspace_root() == Path("/var/lib/aios/workspaces")

    def test_workspace_root_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_WORKSPACE_ROOT", "/custom/path")
        assert workspace_root() == Path("/custom/path")


class TestAttachmentParams:
    def test_as_params_returns_size_and_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"x" * 100)
        att = Attachment(host_path=str(path), filename="doc.pdf", content_type="application/pdf")
        out = att.as_params()
        assert out == {
            "host_path": str(path),
            "filename": "doc.pdf",
            "content_type": "application/pdf",
            "size": 100,
        }

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        att = Attachment(
            host_path=str(tmp_path / "missing"),
            filename="x",
            content_type="text/plain",
        )
        with pytest.raises(AttachmentError, match="does not exist"):
            att.as_params()

    def test_directory_rejected(self, tmp_path: Path) -> None:
        att = Attachment(
            host_path=str(tmp_path),
            filename="x",
            content_type="text/plain",
        )
        with pytest.raises(AttachmentError, match="not a regular file"):
            att.as_params()

    def test_oversize_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "big.bin"
        path.write_bytes(b"\0" * (5 * 1024 * 1024 + 1))
        att = Attachment(
            host_path=str(path), filename="big.bin", content_type="application/octet-stream"
        )
        with pytest.raises(AttachmentError, match="5242880 bytes"):
            att.as_params()


class _PathConsumer(HttpConnector):
    """Tools that take SandboxPath args — exercises dispatcher resolution."""

    connector = "test"

    def __init__(self, root: Path) -> None:
        super().__init__(base_url="http://x", token="aios_runtime_x")
        self._root = root
        self.calls: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []

    @tool()
    async def send_one(self, *, path: SandboxPath) -> str:
        # The dispatcher should have replaced the str with a Path.
        self.calls.append({"path": path})
        assert isinstance(path, Path)
        return "ok"

    @tool()
    async def send_many(self, *, paths: list[SandboxPath]) -> str:
        self.calls.append({"paths": paths})
        for p in paths:
            assert isinstance(p, Path)
        return "ok"

    @tool()
    async def send_optional(self, *, paths: list[SandboxPath] | None = None) -> str:
        self.calls.append({"paths": paths})
        return "ok"

    async def _post_tool_result(  # type: ignore[override]
        self,
        client: Any,
        *,
        connection_id: str,
        session_id: str,
        tool_call_id: str,
        content: Any,
        is_error: bool = False,
    ) -> None:
        del client
        self.tool_results.append(
            {
                "connection_id": connection_id,
                "session_id": session_id,
                "tool_call_id": tool_call_id,
                "content": content,
                "is_error": is_error,
            }
        )


@pytest.fixture
def consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PathConsumer:
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess_1").mkdir()
    (tmp_path / "sess_1" / "photo.jpg").write_bytes(b"x")
    (tmp_path / "sess_1" / "doc.pdf").write_bytes(b"x")
    c = _PathConsumer(tmp_path)
    c._client = AsyncMock()
    return c


class TestDispatchSandboxPathResolution:
    async def test_scalar_path_resolved(self, consumer: _PathConsumer) -> None:
        await consumer.dispatch_call(
            {
                "tool_call_id": "c1",
                "session_id": "sess_1",
                "name": "send_one",
                "arguments": json.dumps({"path": "/workspace/photo.jpg"}),
            }
        )
        assert len(consumer.calls) == 1
        resolved = consumer.calls[0]["path"]
        assert resolved.name == "photo.jpg"
        assert resolved.is_file()

    async def test_list_paths_each_resolved(self, consumer: _PathConsumer) -> None:
        await consumer.dispatch_call(
            {
                "tool_call_id": "c2",
                "session_id": "sess_1",
                "name": "send_many",
                "arguments": json.dumps({"paths": ["/workspace/photo.jpg", "/workspace/doc.pdf"]}),
            }
        )
        paths = consumer.calls[0]["paths"]
        assert {p.name for p in paths} == {"photo.jpg", "doc.pdf"}

    async def test_traversal_rejected_as_error_result(self, consumer: _PathConsumer) -> None:
        await consumer.dispatch_call(
            {
                "tool_call_id": "c3",
                "session_id": "sess_1",
                "name": "send_one",
                "arguments": json.dumps({"path": "/workspace/../../../etc/passwd"}),
            }
        )
        assert consumer.calls == []  # tool body never ran
        assert len(consumer.tool_results) == 1
        result = consumer.tool_results[0]
        assert result["is_error"] is True
        body = json.loads(result["content"])
        assert "outside" in body["error"]

    async def test_optional_list_none_passes_through(self, consumer: _PathConsumer) -> None:
        await consumer.dispatch_call(
            {
                "tool_call_id": "c4",
                "session_id": "sess_1",
                "name": "send_optional",
                "arguments": json.dumps({}),
            }
        )
        # Default kwarg None — not in args dict, so resolver leaves alone.
        assert consumer.calls == [{"paths": None}]
