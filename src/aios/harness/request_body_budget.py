"""Serialized request-body and aggregate-media limits for model calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ANTHROPIC_BODY_BUDGET_BYTES = 26 * 1024 * 1024
API_MAX_MEDIA_PER_REQUEST = 100
MEDIA_OMITTED_TEXT = "[image omitted: request media budget exceeded]"


@dataclass(frozen=True, slots=True)
class BodyBudgetResult:
    request_bytes: int
    media_removed: int


def serialized_request_bytes(payload: dict[str, Any]) -> int:
    """Measure the compact UTF-8 JSON body sent to the provider.

    Transport-only LiteLLM arguments and credentials are deliberately excluded;
    messages, tools and provider passthrough parameters remain included.
    """
    body = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "api_key",
            "api_base",
            "base_url",
            "timeout",
            "stream_timeout",
        }
    }
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str).encode())


def _media_locations(messages: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    locations: list[tuple[int, int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else ""
            locations.append((message_index, part_index, len(url) if isinstance(url, str) else 0))
    return locations


def _omit(messages: list[dict[str, Any]], location: tuple[int, int, int]) -> None:
    message_index, part_index, _ = location
    content = messages[message_index]["content"]
    content[part_index] = {"type": "text", "text": MEDIA_OMITTED_TEXT}


def enforce_request_body_budget(
    payload: dict[str, Any], *, byte_budget: int | None = None, strip_all_media: bool = False
) -> BodyBudgetResult:
    """Evict oldest media until count and serialized-byte limits are satisfied.

    The payload is the post-cache-mutation LiteLLM kwargs and is mutated in
    place. Text and signed thinking blocks are never removed.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return BodyBudgetResult(serialized_request_bytes(payload), 0)
    locations = _media_locations(messages)
    removed = 0

    excess = (
        len(locations) if strip_all_media else max(0, len(locations) - API_MAX_MEDIA_PER_REQUEST)
    )
    for location in locations[:excess]:
        _omit(messages, location)
        removed += 1

    request_bytes = serialized_request_bytes(payload)
    remaining = locations[excess:]
    while remaining and byte_budget is not None and request_bytes > byte_budget:
        _omit(messages, remaining.pop(0))
        removed += 1
        request_bytes = serialized_request_bytes(payload)
    return BodyBudgetResult(request_bytes, removed)


def body_budget_for_model(model: str) -> int | None:
    lowered = model.lower()
    if "anthropic" in lowered or "claude" in lowered:
        return ANTHROPIC_BODY_BUDGET_BYTES
    return None


def is_request_too_large_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    message = str(exc).lower()
    return (
        status == 413
        or "request_too_large" in message
        or "request exceeds the maximum size" in message
    )
