"""Unit coverage for ``signal_send``'s ``attachments`` parameter
and ``_build_send_params``'s ``attachments`` field.

Drives the build helper directly (no signal-cli) plus exercises the
high-level ``signal_send`` method via the SDK's ``_invoke_tool``
dispatch wrapper — that's the layer where ``SandboxPath`` resolution
happens, so connector-side tests must go through it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aios_connector.base import ToolDescriptor

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector, _build_send_params


def test_build_params_no_attachments() -> None:
    params = _build_send_params("+15550001", "alice", "hello", attachments=[])
    assert "attachments" not in params
    assert params["message"] == "hello"


def test_build_params_with_attachments() -> None:
    params = _build_send_params(
        "+15550001",
        "alice",
        "look",
        attachments=[Path("/host/a.jpg"), Path("/host/b.jpg")],
    )
    assert params["attachments"] == ["/host/a.jpg", "/host/b.jpg"]


@pytest.fixture
def connector(tmp_path: Path) -> SignalConnector:
    cfg = Settings(
        phones=["+15550001"],
        config_dir=tmp_path / "cfg",
        cli_bin="/usr/bin/signal-cli",
    )
    c = SignalConnector(cfg)
    c._uuid_to_phone = {"bot-uuid": "+15550001"}
    c._daemon = type(  # type: ignore[assignment]
        "Daemon", (), {"rpc": type("Rpc", (), {"call": AsyncMock(return_value=None)})()}
    )()
    return c


def _descriptor_for(connector: SignalConnector, name: str) -> ToolDescriptor:
    descriptor = next(d for d in connector._tools if d.name == name)
    assert isinstance(descriptor, ToolDescriptor)
    return descriptor


def _signal_send_descriptor(connector: SignalConnector) -> ToolDescriptor:
    return _descriptor_for(connector, "signal_send")


def _stub_focal(connector: SignalConnector, value: str | None) -> None:
    connector._focal_from_request_meta = lambda: value  # type: ignore[method-assign]


def _stub_session_id(connector: SignalConnector, value: str | None) -> None:
    connector.current_session_id = lambda: value  # type: ignore[method-assign]


def _decode(content_list: list[Any]) -> dict[str, Any]:
    assert len(content_list) == 1
    payload = json.loads(content_list[0].text)
    assert isinstance(payload, dict)
    return payload


# Methods bypass focal/sandbox dispatch entirely — verify the descriptor
# carries the right metadata so a regression in ``@tool`` would surface.


def test_signal_send_descriptor_records_sandbox_param(
    connector: SignalConnector,
) -> None:
    descriptor = _signal_send_descriptor(connector)
    assert descriptor.sandbox_params == {"attachments": "list"}


def test_signal_send_schema_publishes_string_array_with_description(
    connector: SignalConnector,
) -> None:
    descriptor = _signal_send_descriptor(connector)
    attachments_schema = descriptor.input_schema["properties"]["attachments"]
    assert attachments_schema["type"] == "array"
    assert attachments_schema["items"]["type"] == "string"
    assert "/workspace/" in attachments_schema["items"]["description"]


# End-to-end via _invoke_tool exercises the dispatch wrapper that
# auto-resolves SandboxPath args before the tool body runs.


async def test_signal_send_text_only_no_session_id_required(
    connector: SignalConnector,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, None)
    descriptor = _signal_send_descriptor(connector)
    result = _decode(await connector._invoke_tool(descriptor, {"text": "hello there"}))
    assert result == {"status": "ok"}
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "hello there"
    assert "attachments" not in sent_params


async def test_signal_send_with_attachments_resolves_to_host_paths(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    descriptor = _signal_send_descriptor(connector)
    await connector._invoke_tool(
        descriptor,
        {"text": "look", "attachments": ["/workspace/cat.jpg"]},
    )

    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "look"
    assert sent_params["attachments"] == [str(ws / "cat.jpg")]


async def test_signal_send_attachments_without_session_id_raises(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    descriptor = _signal_send_descriptor(connector)
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await connector._invoke_tool(
            descriptor,
            {"text": "look", "attachments": ["/workspace/cat.jpg"]},
        )


async def test_signal_send_attachment_traversal_rejected(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    descriptor = _signal_send_descriptor(connector)
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(
            descriptor,
            {"text": "look", "attachments": ["/workspace/../escape.jpg"]},
        )


async def test_signal_send_attachment_disallowed_root_rejected(
    connector: SignalConnector,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    descriptor = _signal_send_descriptor(connector)
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector._invoke_tool(
            descriptor,
            {"text": "boom", "attachments": ["/etc/passwd"]},
        )


async def test_signal_send_attachment_missing_file_raises_clear_error(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_focal(connector, "bot-uuid/alice")
    _stub_session_id(connector, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    descriptor = _signal_send_descriptor(connector)
    with pytest.raises(ValueError, match="does not exist"):
        await connector._invoke_tool(
            descriptor,
            {"text": "look", "attachments": ["/workspace/typo.jpg"]},
        )


async def test_signal_send_unknown_account_raises(
    connector: SignalConnector,
) -> None:
    _stub_focal(connector, "nope/alice")
    descriptor = _signal_send_descriptor(connector)
    with pytest.raises(ValueError, match="unknown account"):
        await connector._invoke_tool(descriptor, {"text": "hi"})
