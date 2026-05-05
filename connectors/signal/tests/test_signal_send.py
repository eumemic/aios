"""Unit coverage for ``signal_send``'s ``attachments`` parameter
and ``_build_send_params``'s ``attachments`` field.

Drives the build helper directly (no signal-cli) plus exercises the
high-level ``signal_send`` method with a stubbed daemon to verify
that attachment paths reach the RPC params resolved to host paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
        attachments=["/host/a.jpg", "/host/b.jpg"],
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


def _patch_session_id(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr(SignalConnector, "current_session_id", lambda self: value)


async def test_signal_send_text_only_no_session_id_required(
    connector: SignalConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_session_id(monkeypatch, None)
    result = await connector.signal_send("hello there", account="bot-uuid", chat_id="alice")
    assert result == {"status": "ok"}
    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "hello there"
    assert "attachments" not in sent_params


async def test_signal_send_with_attachments_resolves_to_host_paths(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    ws = (tmp_path / "sess-1").resolve()
    ws.mkdir(parents=True)
    (ws / "cat.jpg").write_bytes(b"x")

    await connector.signal_send(
        "look",
        attachments=["/workspace/cat.jpg"],
        account="bot-uuid",
        chat_id="alice",
    )

    sent_params = connector._daemon.rpc.call.call_args.args[1]  # type: ignore[union-attr]
    assert sent_params["message"] == "look"
    assert sent_params["attachments"] == [str(ws / "cat.jpg")]


async def test_signal_send_attachments_without_session_id_raises(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, None)
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match=r"aios\.session_id"):
        await connector.signal_send(
            "look",
            attachments=["/workspace/cat.jpg"],
            account="bot-uuid",
            chat_id="alice",
        )


async def test_signal_send_attachment_traversal_rejected(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    with pytest.raises(ValueError, match="could not be resolved"):
        await connector.signal_send(
            "look",
            attachments=["/workspace/../escape.jpg"],
            account="bot-uuid",
            chat_id="alice",
        )


async def test_signal_send_attachment_disallowed_root_rejected(
    connector: SignalConnector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    with pytest.raises(ValueError, match="could not be resolved"):
        await connector.signal_send(
            "boom",
            attachments=["/etc/passwd"],
            account="bot-uuid",
            chat_id="alice",
        )


async def test_signal_send_attachment_missing_file_raises_clear_error(
    connector: SignalConnector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_session_id(monkeypatch, "sess-1")
    monkeypatch.setenv("AIOS_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "sess-1").mkdir(parents=True)

    with pytest.raises(ValueError, match="does not exist"):
        await connector.signal_send(
            "look",
            attachments=["/workspace/typo.jpg"],
            account="bot-uuid",
            chat_id="alice",
        )


async def test_signal_send_unknown_account_raises(
    connector: SignalConnector,
) -> None:
    with pytest.raises(ValueError, match="unknown account"):
        await connector.signal_send("hi", account="nope", chat_id="alice")
