"""Wrapper around :func:`litellm.acompletion`.

Provides two variants:

* :func:`call_litellm` — non-streaming, returns ``(message, usage)``.
* :func:`stream_litellm` — streaming, delivers per-token deltas via
  ``pg_notify`` and returns ``(message, usage)``.

Model API keys are resolved by LiteLLM from standard environment variables
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.) based on the model string
prefix.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    import asyncpg


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Strip null-valued fields that break downstream JSONB queries.

    Some LiteLLM providers (e.g. openrouter/moonshotai/kimi-k2.5) return
    ``tool_calls: null`` instead of omitting the key.  Stored verbatim,
    this becomes JSONB ``null`` which ``jsonb_array_length`` rejects.
    """
    if "tool_calls" in msg and msg["tool_calls"] is None:
        del msg["tool_calls"]
    return msg


_CACHE_CONTROL = {"type": "ephemeral"}


def inject_cache_breakpoints(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Annotate messages and tools with ``cache_control`` breakpoints.

    Anthropic's prompt caching requires explicit ``cache_control`` markers
    to create cache entries.  LiteLLM strips them for providers that don't
    support them (e.g. OpenAI), so this is safe to apply unconditionally.

    Places breakpoints on the system message, last tool definition, and
    last conversation message (3 of Anthropic's max 4).
    """
    if not messages:
        return

    if messages[0].get("role") == "system":
        messages[0]["cache_control"] = _CACHE_CONTROL

    if tools:
        tools[-1]["cache_control"] = _CACHE_CONTROL

    last = messages[-1]
    if last.get("role") != "system":
        last["cache_control"] = _CACHE_CONTROL


def _normalize_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Map LiteLLM's usage field names to our canonical names.

    LiteLLM uses OpenAI-style ``prompt_tokens`` / ``completion_tokens``.
    Some providers (Anthropic via LiteLLM) also pass through
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens``
    at the top level. OpenAI-compatible providers put cache reads in
    ``prompt_tokens_details.cached_tokens``.
    """
    prompt_details = raw.get("prompt_tokens_details") or {}
    cache_read = raw.get("cache_read_input_tokens") or prompt_details.get("cached_tokens") or 0
    return {
        "input_tokens": raw.get("prompt_tokens") or 0,
        "output_tokens": raw.get("completion_tokens") or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": raw.get("cache_creation_input_tokens") or 0,
    }


async def call_litellm(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Call ``litellm.acompletion`` and return ``(message, usage)``.

    Returns the message exactly as litellm produced it, including any
    provider-specific extensions like ``reasoning_content`` or
    ``thinking_blocks``. The harness stores the message dict opaquely.
    Usage is normalized to our canonical field names.
    """
    inject_cache_breakpoints(messages, tools)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if api_base is not None:
        kwargs["api_base"] = api_base
    if extra:
        kwargs.update(extra)

    response = await litellm.acompletion(**kwargs)
    usage_obj = response.get("usage")
    usage = _normalize_usage(
        usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj or {}
    )
    message = response["choices"][0]["message"]
    # litellm returns a Message object that supports model_dump()
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return _normalize_message(result), usage
    if isinstance(message, dict):
        return _normalize_message(message), usage
    raise TypeError(f"unexpected message type from litellm.acompletion: {type(message).__name__}")


async def stream_litellm(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    extra: dict[str, Any] | None = None,
    pool: asyncpg.Pool[Any],
    session_id: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Call ``litellm.acompletion`` with streaming, returning ``(message, usage)``.

    Each content delta fires a transient ``pg_notify`` on the session's
    event channel. SSE clients receive these as ``event: delta`` — no DB
    row is created. After the stream exhausts, the complete message is
    assembled via ``litellm.stream_chunk_builder`` and returned for
    storage as a normal event.
    """
    inject_cache_breakpoints(messages, tools)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
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
    usage_obj = assembled.get("usage")
    usage = _normalize_usage(
        usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj or {}
    )
    message = assembled["choices"][0]["message"]
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return _normalize_message(result), usage
    if isinstance(message, dict):
        return _normalize_message(message), usage
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
