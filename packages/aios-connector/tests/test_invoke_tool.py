"""Dispatch-side tests for ``Connector._invoke_tool`` focal injection.

Exercises the signature-inspection branch end-to-end without a live MCP
server: stub out ``_focal_from_request_meta`` to return a known path,
invoke ``_invoke_tool`` directly, and assert the right kwargs reach the
tool method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from aios_connector import Connector, focal_required, tool
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
