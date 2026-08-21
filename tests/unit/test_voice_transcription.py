from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aios.models.model_providers import ProviderAuth
from aios.services.attachment_staging import InboundAttachment
from aios.services.voice_transcription import transcribe_voice_attachments


class MemoryUpload:
    filename: str | None = "voice-3664.ogg"
    content_type: str | None = "audio/ogg"

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.data) - self.position
        chunk = self.data[self.position : self.position + size]
        self.position += len(chunk)
        return chunk

    async def seek(self, offset: int) -> None:
        self.position = offset


def attachment(data: bytes) -> InboundAttachment:
    return InboundAttachment(MemoryUpload(data), "voice-3664.ogg", "audio/ogg")


@pytest.mark.asyncio
async def test_transcription_is_added_and_upload_rewound(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = attachment(b"OggS-valid-audio")
    resolve = AsyncMock(
        return_value=(ProviderAuth("proxy-secret", "http://oai-proxy/v1", "acc_1"), None)
    )
    monkeypatch.setattr(
        "aios.services.voice_transcription.resolve_provider_auth_or_conflict", resolve
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://oai-proxy/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer proxy-secret"
        assert b"OggS-valid-audio" in await request.aread()
        return httpx.Response(200, json={"text": "Please ship the fix."})

    content = await transcribe_voice_attachments(
        pool=MagicMock(),
        crypto_box=MagicMock(),
        account_id="acc_1",
        connector="telegram",
        content="",
        attachments=[upload],
        transport=httpx.MockTransport(handler),
    )

    assert content == "[Voice note transcript: Please ship the fix.]"
    assert await upload.stream.read() == b"OggS-valid-audio"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"corrupt"])
async def test_empty_or_corrupt_audio_failure_is_loud(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    upload = attachment(payload)
    monkeypatch.setattr(
        "aios.services.voice_transcription.resolve_provider_auth_or_conflict",
        AsyncMock(
            return_value=(ProviderAuth("proxy-secret", "http://oai-proxy/v1", "acc_1"), None)
        ),
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": ""})

    content = await transcribe_voice_attachments(
        pool=MagicMock(),
        crypto_box=MagicMock(),
        account_id="acc_1",
        connector="telegram",
        content="",
        attachments=[upload],
        transport=httpx.MockTransport(handler),
    )

    assert content.startswith("[Voice transcription failed:")
    assert await upload.stream.read() == payload


@pytest.mark.asyncio
async def test_missing_proxy_credential_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aios.services.voice_transcription.resolve_provider_auth_or_conflict",
        AsyncMock(return_value=(None, None)),
    )
    content = await transcribe_voice_attachments(
        pool=MagicMock(),
        crypto_box=MagicMock(),
        account_id="acc_1",
        connector="telegram",
        content="caption",
        attachments=[attachment(b"OggS")],
    )
    assert (
        content
        == "caption\n\n[Voice transcription failed: oai-proxy credential is not configured.]"
    )
