"""Serialized request-body and aggregate-media limits for model calls.

Anthropic's Messages API rejects any request whose **serialized body** exceeds
32 MB with ``request_too_large`` (HTTP 413). The harness budgets context by
*tokens*, which is uncorrelated with body bytes once base64 media is in the
window: a session can sit comfortably inside its token window and still emit a
40 MB body. Window trimming does not reliably clear it either — trimming drops
whole events by token cost, so a single retained image-heavy event can keep the
body over the ceiling indefinitely. That is the #1999 brick: every wake rebuilds
the same oversize body, the provider 413s before a token is generated, and the
session cannot make progress on its own.

This module is the last gate before the wire. It measures the payload LiteLLM
will actually serialize and evicts **oldest media first**, replacing each
``image_url`` part with a short text marker so the model still sees that
something was there. Text and signed thinking blocks are never removed.

**Limits are provider-scoped.** Both the byte ceiling and the aggregate
media-count cap are Anthropic's, and neither is universal — OpenAI-compatible
endpoints have their own (much larger, or absent) limits. Applying Anthropic's
caps to every provider would silently destroy history on providers that were
never going to reject the request. :func:`body_limits_for_model` resolves both
numbers together from one provider verdict, so there is a single place where
"which provider are we talking to" is decided and no path can pick up one limit
without the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Anthropic's documented request-body ceiling is 32 MB. We budget to 26 MB so
# there is headroom for provider-side serialization overhead (LiteLLM's
# Anthropic adapter re-encodes content blocks, and HTTP framing/JSON escaping
# both inflate the on-wire size relative to our compact local measurement).
ANTHROPIC_BODY_BUDGET_BYTES = 26 * 1024 * 1024

# Anthropic caps a single request at 100 images. This is an Anthropic API
# limit, NOT a universal one — see ``body_limits_for_model``.
ANTHROPIC_MAX_MEDIA_PER_REQUEST = 100

MEDIA_OMITTED_TEXT = "[image omitted: request media budget exceeded]"

# Provider strings that route to Anthropic's Messages API and therefore inherit
# its body/media ceilings. aios reaches Claude through several providers whose
# model strings all still carry ``claude`` or ``anthropic`` (direct Anthropic,
# ``openrouter/anthropic/*``, ``bedrock/anthropic.*``, ``vertex_ai/claude-*``),
# which is the same substring test
# :func:`~aios.harness.completion.model_descriptor` and
# :func:`~aios.harness.vision.supports_vision` use. Kept as a local substring
# check rather than a ``litellm.get_llm_provider`` sniff so this module stays
# import-light and usable from the hot path without pulling in litellm.
_ANTHROPIC_MODEL_MARKERS = ("anthropic", "claude")


@dataclass(frozen=True, slots=True)
class BodyLimits:
    """The provider's request-body ceilings.

    ``None`` on either field means "this provider publishes no such limit we
    need to enforce" — the corresponding trimming pass is skipped entirely
    rather than falling back to some other provider's number.
    """

    byte_budget: int | None = None
    max_media: int | None = None


@dataclass(frozen=True, slots=True)
class BodyBudgetResult:
    request_bytes: int
    media_removed: int


def body_limits_for_model(model: str) -> BodyLimits:
    """Resolve the request-body limits that apply to ``model``.

    Both limits come from one provider verdict so a caller cannot apply the
    byte ceiling while silently inheriting another provider's media cap (or
    vice versa). Non-Anthropic models get :class:`BodyLimits` with both fields
    ``None`` — no proactive trimming at all, because we have no evidence the
    provider would reject the request and dropping history is destructive.
    """
    lowered = model.lower()
    if any(marker in lowered for marker in _ANTHROPIC_MODEL_MARKERS):
        return BodyLimits(
            byte_budget=ANTHROPIC_BODY_BUDGET_BYTES,
            max_media=ANTHROPIC_MAX_MEDIA_PER_REQUEST,
        )
    return BodyLimits()


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
    payload: dict[str, Any],
    *,
    limits: BodyLimits | None = None,
    strip_all_media: bool = False,
) -> BodyBudgetResult:
    """Evict oldest media until ``limits`` are satisfied.

    The payload is the post-cache-mutation LiteLLM kwargs and is mutated in
    place. Text and signed thinking blocks are never removed.

    ``limits`` scopes the enforcement to the bound provider (see
    :func:`body_limits_for_model`); a limit left ``None`` disables that pass, so
    a provider with no published media cap keeps every image and a provider with
    no published byte ceiling is never byte-trimmed. ``limits=None`` means "no
    proactive limits at all" and is the correct default for an unknown provider.

    ``strip_all_media=True`` is the reactive path: the provider already rejected
    the request as too large, so every image is evicted regardless of provider
    scoping. That is not a cross-provider assumption — it is a response to that
    provider's own explicit ``request_too_large``/413 verdict.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return BodyBudgetResult(serialized_request_bytes(payload), 0)
    effective = limits or BodyLimits()
    locations = _media_locations(messages)
    removed = 0

    if strip_all_media:
        excess = len(locations)
    elif effective.max_media is not None:
        excess = max(0, len(locations) - effective.max_media)
    else:
        excess = 0
    for location in locations[:excess]:
        _omit(messages, location)
        removed += 1

    request_bytes = serialized_request_bytes(payload)
    byte_budget = effective.byte_budget
    remaining = locations[excess:]
    while remaining and byte_budget is not None and request_bytes > byte_budget:
        _omit(messages, remaining.pop(0))
        removed += 1
        request_bytes = serialized_request_bytes(payload)
    return BodyBudgetResult(request_bytes, removed)


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
