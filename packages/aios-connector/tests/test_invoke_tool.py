"""Dispatch-side tests for ``Connector._invoke_tool``.

Exercises both the focal-injection branch (signature inspection picks
which kwargs to inject from ``_meta.aios.focal_channel_path``) and the
:data:`SandboxPath` auto-resolution branch (each tool param annotated
with ``SandboxPath`` / ``list[SandboxPath]`` is resolved to a host
:class:`Path` before the tool body runs).

Stub out ``_focal_from_request_meta`` and ``current_session_id`` to
return known values, invoke ``_invoke_tool`` directly, and assert the
right kwargs reach the tool method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from aios_connector import Connector, SandboxPath, focal_required, tool
from aios_connector.base import _TOOL_ATTR, ToolDescriptor


class _DispatchFixture(Connector):
    name = "_dispatch_fixture"

    @tool()
    @focal_required
    async def chat_only(self, *, chat_id: str) -> dict[str, str]:
        """Single-account: chat_id only."""
        return {"chat_id": chat_id}

    @tool()
    @focal_required
    async def both(self, *, account: str, chat_id: str) -> dict[str, str]:
        """Multi-account: both injected."""
        return {"account": account, "chat_id": chat_id}

    @tool()
    @focal_required
    async def account_only(self, *, account: str) -> dict[str, str]:
        """Account only: chat_id ignored."""
        return {"account": account}

    @tool()
    async def non_focal(self, text: str) -> dict[str, str]:
        """No focal injection."""
        return {"text": text}


def _descriptor_for(fn: object) -> ToolDescriptor:
    descriptor = getattr(fn, _TOOL_ATTR, None)
    assert isinstance(descriptor, ToolDescriptor)
    return descriptor


def _decode(content_list: list[Any]) -> dict[str, Any]:
    """Pull the JSON-encoded result back out of the TextContent envelope."""
    assert len(content_list) == 1
    payload = json.loads(content_list[0].text)
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _DispatchFixture:
    connector = _DispatchFixture(spool_dir=tmp_path)
    return connector


def _stub_focal(connector: _DispatchFixture, value: str | None) -> None:
    """Replace ``_focal_from_request_meta`` to return ``value`` verbatim."""
    connector._focal_from_request_meta = lambda: value  # type: ignore[method-assign]


async def test_chat_only_receives_chat_id(fixture: _DispatchFixture) -> None:
    _stub_focal(fixture, "echo-1/chat-42")
    descriptor = _descriptor_for(_DispatchFixture.chat_only)
    result = _decode(await fixture._invoke_tool(descriptor, {}))
    # Account part is silently dropped because the method didn't ask for it.
    assert result == {"chat_id": "chat-42"}


async def test_both_receives_account_and_chat(fixture: _DispatchFixture) -> None:
    _stub_focal(fixture, "echo-1/chat-42")
    descriptor = _descriptor_for(_DispatchFixture.both)
    result = _decode(await fixture._invoke_tool(descriptor, {}))
    assert result == {"account": "echo-1", "chat_id": "chat-42"}


async def test_account_only_drops_chat(fixture: _DispatchFixture) -> None:
    _stub_focal(fixture, "echo-1/chat-42")
    descriptor = _descriptor_for(_DispatchFixture.account_only)
    result = _decode(await fixture._invoke_tool(descriptor, {}))
    assert result == {"account": "echo-1"}


async def test_nested_chat_path_kept_intact(fixture: _DispatchFixture) -> None:
    """Telegram forum-thread style: chat_id contains a slash."""
    _stub_focal(fixture, "bot-12/group-7/thread-3")
    descriptor = _descriptor_for(_DispatchFixture.both)
    result = _decode(await fixture._invoke_tool(descriptor, {}))
    assert result == {"account": "bot-12", "chat_id": "group-7/thread-3"}


async def test_focal_required_without_meta_raises(fixture: _DispatchFixture) -> None:
    _stub_focal(fixture, None)
    descriptor = _descriptor_for(_DispatchFixture.both)
    with pytest.raises(ValueError, match="requires a focal channel"):
        await fixture._invoke_tool(descriptor, {})


async def test_non_focal_tool_ignores_meta(fixture: _DispatchFixture) -> None:
    """A tool without @focal_required gets no focal kwargs even if meta is set."""
    _stub_focal(fixture, "echo-1/chat-42")  # would inject if focal-required
    descriptor = _descriptor_for(_DispatchFixture.non_focal)
    result = _decode(await fixture._invoke_tool(descriptor, {"text": "hi"}))
    assert result == {"text": "hi"}


# ── SandboxPath dispatch: model strings → host Path objects ─────────


class _SandboxDispatchFixture(Connector):
    name = "_sandbox_dispatch_fixture"

    @tool()
    async def take_list(
        self,
        attachments: list[SandboxPath] | None = None,
    ) -> dict[str, Any]:
        """Body sees a list[Path] — the SDK pre-resolved each entry."""
        if attachments is None:
            return {"attachments": None}
        return {
            "attachments": [str(p) for p in attachments],
            "kinds": [type(p).__name__ for p in attachments],
        }

    @tool()
    async def take_scalar(
        self,
        path: SandboxPath,
    ) -> dict[str, Any]:
        """Body sees a Path — the SDK pre-resolved the string."""
        return {"path": str(path), "kind": type(path).__name__}


@pytest.fixture
def sandbox_fixture(tmp_path: Path) -> _SandboxDispatchFixture:
    return _SandboxDispatchFixture(spool_dir=tmp_path)


def _stub_session_id(connector: Connector, value: str | None) -> None:
    connector.current_session_id = lambda: value  # type: ignore[method-assign]


async def test_list_sandbox_path_resolved_to_paths(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatch wrapper resolves each model-supplied string to a host
    :class:`Path` BEFORE the tool body runs."""
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-x").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")
    (ws / "dog.jpg").write_bytes(b"y")

    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    result = _decode(
        await sandbox_fixture._invoke_tool(
            descriptor,
            {"attachments": ["/workspace/cat.jpg", "/workspace/dog.jpg"]},
        )
    )
    assert result["attachments"] == [str(ws / "cat.jpg"), str(ws / "dog.jpg")]
    assert all(k in {"PosixPath", "WindowsPath"} for k in result["kinds"])


async def test_scalar_sandbox_path_resolved_to_path(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-x").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    descriptor = _descriptor_for(_SandboxDispatchFixture.take_scalar)
    result = _decode(
        await sandbox_fixture._invoke_tool(
            descriptor,
            {"path": "/workspace/cat.jpg"},
        )
    )
    assert result["path"] == str(ws / "cat.jpg")
    assert result["kind"] in {"PosixPath", "WindowsPath"}


async def test_omitted_optional_list_sandbox_path_passes_through_as_none(
    sandbox_fixture: _SandboxDispatchFixture,
) -> None:
    """The optional ``attachments`` parameter omitted by the model — no
    session_id, no resolution work, body sees the default ``None``."""
    _stub_session_id(sandbox_fixture, None)
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    result = _decode(await sandbox_fixture._invoke_tool(descriptor, {}))
    assert result == {"attachments": None}


async def test_empty_list_sandbox_path_passes_through_without_session_id(
    sandbox_fixture: _SandboxDispatchFixture,
) -> None:
    """Empty list short-circuits — no path strings means no session_id
    is needed, so text-only call sites work in tests that don't stamp it."""
    _stub_session_id(sandbox_fixture, None)
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    result = _decode(await sandbox_fixture._invoke_tool(descriptor, {"attachments": []}))
    assert result == {"attachments": [], "kinds": []}


async def test_list_sandbox_path_missing_session_id_raises(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_session_id(sandbox_fixture, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await sandbox_fixture._invoke_tool(descriptor, {"attachments": ["/workspace/x.jpg"]})


async def test_list_sandbox_path_containment_violation_uniformly_raises(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uniform error wording across connectors is the whole point — any
    sandbox-path arg that fails containment raises the same ValueError."""
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    with pytest.raises(ValueError, match="could not be resolved"):
        await sandbox_fixture._invoke_tool(descriptor, {"attachments": ["/etc/passwd"]})


async def test_list_sandbox_path_traversal_rejected(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-x").mkdir(parents=True)
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    with pytest.raises(ValueError, match="could not be resolved"):
        await sandbox_fixture._invoke_tool(
            descriptor, {"attachments": ["/workspace/../escape.jpg"]}
        )


async def test_list_sandbox_path_missing_file_raises(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-x").mkdir(parents=True)
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    with pytest.raises(ValueError, match="does not exist"):
        await sandbox_fixture._invoke_tool(descriptor, {"attachments": ["/workspace/typo.jpg"]})


async def test_list_sandbox_path_wrong_arg_type_raises_typeerror(
    sandbox_fixture: _SandboxDispatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model passing a string instead of a list is a contract bug,
    not silent recovery — surface the error so the model retries with
    the right shape."""
    _stub_session_id(sandbox_fixture, "sess-x")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    descriptor = _descriptor_for(_SandboxDispatchFixture.take_list)
    with pytest.raises(TypeError, match="expects a list"):
        await sandbox_fixture._invoke_tool(descriptor, {"attachments": "/workspace/x.jpg"})
