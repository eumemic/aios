"""Wrapper around :func:`litellm.acompletion`.

Provides two variants:

* :func:`call_litellm` — non-streaming, returns the complete message dict.
* :func:`stream_litellm` — streaming, delivers per-token deltas via
  ``pg_notify`` and returns the assembled message dict.

The wrappers deliberately keep no state — credentials are passed in per call
and never cached on the wrapper instance, so a single misconfigured request
can't leak into a subsequent one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    import asyncpg


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


async def stream_litellm(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra: dict[str, Any] | None = None,
    pool: asyncpg.Pool[Any],
    session_id: str,
) -> dict[str, Any]:
    """Call ``litellm.acompletion`` with streaming, delivering per-token
    deltas via ``pg_notify`` and returning the assembled assistant message.

    Each content delta fires a transient ``pg_notify`` on the session's
    event channel. SSE clients receive these as ``event: delta`` — no DB
    row is created. After the stream exhausts, the complete message is
    assembled via ``litellm.stream_chunk_builder`` and returned for
    storage as a normal event.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
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

    chunks: list[Any] = []
    async for chunk in response:
        chunks.append(chunk)
        content = chunk.choices[0].delta.content
        if content:
            await _notify_delta(pool, session_id, content)

    assembled: Any = litellm.stream_chunk_builder(chunks=chunks)
    message = assembled["choices"][0]["message"]
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return result
    if isinstance(message, dict):
        return message
    raise TypeError(f"unexpected message type from stream_chunk_builder: {type(message).__name__}")


async def _notify_delta(
    pool: asyncpg.Pool[Any],
    session_id: str,
    content: str,
) -> None:
    """Send a transient content delta via pg_notify.

    Uses the same ``events_{session_id}`` channel as persisted events.
    The JSON payload is distinguishable from event-id payloads because
    it starts with ``{``.
    """
    payload = json.dumps({"delta": content})
    async with pool.acquire() as conn:
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            f"events_{session_id}",
            payload,
        )
