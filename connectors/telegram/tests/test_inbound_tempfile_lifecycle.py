"""Verify that ``_download_attachments`` does not leak per-attachment
temp files into the host's ``/tmp`` on the success path.

``_download_one`` creates a ``tempfile.NamedTemporaryFile(prefix=
"aios-telegram-", delete=False)``, hands the path to
``Application.bot.get_file(...).download_to_drive(...)``, and returns
the path. ``_download_attachments`` then calls ``host_path.read_bytes()``
on each, packs the in-memory blob into the runtime tuple, and forgets
the path. Only the explicit ``except (TelegramError, OSError)`` branch
inside ``_download_one`` unlinks the temp file — every successful
inbound attachment leaks one file into ``/tmp`` for the lifetime of
the connector container.

For a long-running Telegram bot serving any meaningful traffic
(photos, voice messages, documents, stickers, videos) this leaks
unboundedly. Eventual symptom: ``/tmp`` exhaustion → ``OSError("No
space left on device")`` on the NEXT ``NamedTemporaryFile`` →
unhandled exception kills the per-connection serve task → connector
goes mute until container restart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios_telegram.connector import TelegramConnector, _TelegramConnectionState
from aios_telegram.parse import Attachment


@pytest.fixture
def isolated_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``tempfile.gettempdir()`` at a per-test directory so we
    can count leaks deterministically."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _make_fake_state(download_payload: bytes) -> _TelegramConnectionState:
    """Build a minimal ``_TelegramConnectionState`` whose
    ``application.bot.get_file(...).download_to_drive(custom_path=...)``
    writes ``download_payload`` to the requested path. Mirrors PTB's
    real ``File.download_to_drive`` contract."""
    state = MagicMock(spec=_TelegramConnectionState)
    state.application = MagicMock()

    async def fake_download_to_drive(custom_path: Path) -> Path:
        Path(custom_path).write_bytes(download_payload)  # noqa: ASYNC240
        return Path(custom_path)

    fake_file = MagicMock()
    fake_file.download_to_drive = AsyncMock(side_effect=fake_download_to_drive)
    state.application.bot.get_file = AsyncMock(return_value=fake_file)
    return state


async def test_successful_download_does_not_leak_tempfile(
    isolated_tmp: Path,
) -> None:
    """``_download_attachments`` returns the in-memory blob to the
    caller and must clean up the on-disk temp file. Pre-fix every
    successful download leaks one file into ``tempfile.gettempdir()``;
    post-fix the directory is empty after the call."""
    payload = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    state = _make_fake_state(payload)
    connector = TelegramConnector.__new__(TelegramConnector)
    connector.state = {}

    attachment = Attachment(
        file_id="file_test_1",
        content_type="image/png",
        filename="photo.png",
    )

    result = await connector._download_attachments(state, (attachment,))

    # Success path: returned the runtime tuple with the bytes we wrote.
    assert result is not None
    assert len(result) == 1
    filename, blob, content_type = result[0]
    assert filename == "photo.png"
    assert blob == payload
    assert content_type == "image/png"

    # No leak: the temp dir should be empty (no ``aios-telegram-*`` files).
    leaked = list(isolated_tmp.glob("aios-telegram-*"))  # noqa: ASYNC240
    assert leaked == [], (
        f"successful Telegram inbound attachment leaked temp file(s): {leaked!r}. "
        f"Pre-fix every download leaves the tempfile in place because only the "
        f"``download_failed`` error path unlinks. Over a long-running container "
        f"this fills /tmp; eventually ``NamedTemporaryFile`` raises and the "
        f"serve task dies."
    )


async def test_multiple_attachments_all_cleaned_up(isolated_tmp: Path) -> None:
    """N attachments in one inbound → N tempfiles created → N unlinks."""
    state = _make_fake_state(b"data")
    connector = TelegramConnector.__new__(TelegramConnector)
    connector.state = {}

    attachments = tuple(
        Attachment(file_id=f"file_{i}", content_type="image/png", filename=f"img_{i}.png")
        for i in range(5)
    )

    result = await connector._download_attachments(state, attachments)

    assert result is not None
    assert len(result) == 5

    leaked = list(isolated_tmp.glob("aios-telegram-*"))  # noqa: ASYNC240
    assert leaked == [], f"multi-attachment inbound leaked: {leaked!r}"


async def test_read_failure_also_unlinks_tempfile(isolated_tmp: Path) -> None:
    """If ``read_bytes`` raises after download succeeded (FS gone
    read-only, NFS hiccup, etc.), the tempfile must still be unlinked.
    Otherwise this becomes a slow leak whenever filesystem health
    flakes mid-flight."""
    state = _make_fake_state(b"data")
    connector = TelegramConnector.__new__(TelegramConnector)
    connector.state = {}

    # Sabotage read_bytes by removing read permission *after* download
    # but before the attachment loop reads. Easiest: monkeypatch
    # ``Path.read_bytes`` for this scope to raise OSError.
    real_read_bytes = Path.read_bytes

    def boom(self: Path) -> bytes:
        if "aios-telegram-" in str(self):
            raise OSError("simulated read failure")
        return real_read_bytes(self)

    import unittest.mock as _mock

    with _mock.patch.object(Path, "read_bytes", boom):
        result = await connector._download_attachments(
            state,
            (Attachment(file_id="f", content_type="image/png", filename="x.png"),),
        )

    assert result is None  # all attachments were rejected
    leaked = list(isolated_tmp.glob("aios-telegram-*"))  # noqa: ASYNC240
    assert leaked == [], (
        f"read-failure path leaked tempfile(s): {leaked!r}; cleanup must run "
        f"in both the success and the read-failure branches."
    )


# pytest-asyncio mode is set in pyproject; mark explicitly anyway
pytestmark = pytest.mark.asyncio
