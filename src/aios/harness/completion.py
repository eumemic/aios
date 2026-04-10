"""Wrapper around :func:`litellm.acompletion`.

Phase 1: a thin wrapper that calls litellm and returns the message dict.
Phase 5 will add cancellation support via an asyncio.Event.

The wrapper deliberately keeps no state — credentials are passed in per call
and never cached on the wrapper instance, so a single misconfigured request
can't leak into a subsequent one.
"""

from __future__ import annotations

from typing import Any

import litellm


async def call_litellm(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call ``litellm.acompletion`` and return the assistant message dict.

    Returns the message exactly as litellm produced it, including any
    provider-specific extensions like ``reasoning_content`` or
    ``thinking_blocks``. The harness stores this dict opaquely.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base
    if extra:
        kwargs.update(extra)

    response = await litellm.acompletion(**kwargs)
    message = response["choices"][0]["message"]
    # litellm returns a Message object that supports model_dump()
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return result
    if isinstance(message, dict):
        return message
    raise TypeError(f"unexpected message type from litellm.acompletion: {type(message).__name__}")
