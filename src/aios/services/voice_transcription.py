"""Synchronous inbound voice-note transcription through account-scoped inference auth."""

from __future__ import annotations

import json
from typing import Any

import httpx

from aios.crypto.vault import CryptoBox
from aios.services.attachment_staging import InboundAttachment
from aios.services.model_providers import resolve_provider_auth_or_conflict

_TRANSCRIPTION_MODEL = "openai/whisper-1"
_FAILURE_PREFIX = "[Voice transcription failed: "


def _is_telegram_voice(attachment: InboundAttachment) -> bool:
    return attachment.content_type.split(";", 1)[0].lower() in {
        "audio/ogg",
        "audio/opus",
    } and attachment.filename.lower().endswith((".ogg", ".opus"))


def _append_notice(content: str, notice: str) -> str:
    return f"{content}\n\n{notice}" if content else notice


async def transcribe_voice_attachments(
    *,
    pool: Any,
    crypto_box: CryptoBox,
    account_id: str,
    connector: str,
    content: str,
    attachments: list[InboundAttachment],
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Transcribe Telegram voice files before the inbound event wakes its seat.

    Failures become model-visible text rather than exceptions or empty transcripts.
    Upload streams are always rewound so attachment staging sees the original bytes.
    """
    voice_attachments = [
        a for a in attachments if connector == "telegram" and _is_telegram_voice(a)
    ]
    if not voice_attachments:
        return content

    auth, conflict = await resolve_provider_auth_or_conflict(
        pool,
        crypto_box,
        account_id=account_id,
        model=_TRANSCRIPTION_MODEL,
        litellm_extra=None,
    )
    if conflict is not None:
        return _append_notice(content, f"{_FAILURE_PREFIX}{conflict}.]")
    if auth is None or auth.api_base is None:
        return _append_notice(
            content,
            f"{_FAILURE_PREFIX}oai-proxy credential is not configured.]",
        )

    result = content
    endpoint = f"{auth.api_base.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {auth.api_key}"}
    async with httpx.AsyncClient(transport=transport, timeout=60.0) as client:
        for attachment in voice_attachments:
            try:
                data = await attachment.stream.read()
                if not data:
                    raise ValueError("audio file is empty")
                if attachment.content_type.split(";", 1)[
                    0
                ].lower() == "audio/ogg" and not data.startswith(b"OggS"):
                    raise ValueError("audio file is not a valid Ogg stream")
                response = await client.post(
                    endpoint,
                    headers=headers,
                    data={"model": "whisper-1", "response_format": "json"},
                    files={"file": (attachment.filename, data, attachment.content_type)},
                )
                response.raise_for_status()
                payload = response.json()
                transcript = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(transcript, str) or not transcript.strip():
                    raise ValueError("proxy returned an empty transcript")
                result = _append_notice(result, f"[Voice note transcript: {transcript.strip()}]")
            except (httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError) as err:
                result = _append_notice(result, f"{_FAILURE_PREFIX}{err}.]")
            finally:
                await attachment.stream.seek(0)  # type: ignore[attr-defined]
    return result
