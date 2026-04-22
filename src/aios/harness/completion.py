"""Wrapper around :func:`litellm.acompletion`.

Provides two variants:

* :func:`call_litellm` — non-streaming, returns ``(message, usage, cost)``.
* :func:`stream_litellm` — streaming, delivers per-token deltas via
  ``pg_notify`` and returns ``(message, usage, cost)``.

``cost`` is the LiteLLM-computed USD cost for the request, or ``None``
when the provider/model didn't report one.

Model API keys are resolved by LiteLLM from standard environment variables
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.) based on the model string
prefix.
"""

from __future__ import annotations

import json
from functools import cache
from typing import TYPE_CHECKING, Any

import litellm

# Anthropic rejects empty text blocks that some OpenRouter models emit on
# tool-call-only turns; modify_params tells LiteLLM to sanitize them.
litellm.modify_params = True

if TYPE_CHECKING:
    import asyncpg


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider quirks that break downstream consumers.

    Some LiteLLM providers return ``tool_calls: null`` instead of omitting
    the key (breaks ``jsonb_array_length``), and ``content: null`` instead
    of ``content: ""`` (breaks providers like MiniMax and Gemma when the
    message is replayed in a cross-model session).
    """
    if "tool_calls" in msg and msg["tool_calls"] is None:
        del msg["tool_calls"]
    if msg.get("content") is None:
        msg["content"] = ""
    return msg


_CACHE_CONTROL = {"type": "ephemeral"}


def _set_content_block_cache(msg: dict[str, Any]) -> None:
    """Place ``cache_control`` on the last content block of a message.

    Anthropic requires ``cache_control`` on content blocks, not on the
    message dict itself.  If ``content`` is a plain string, it is converted
    to content-block format so the marker has somewhere to live.
    """
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
    elif isinstance(content, list) and content:
        content[-1]["cache_control"] = _CACHE_CONTROL


# LiteLLM providers that proxy Anthropic models and forward ``cache_control``
# markers unchanged (OpenRouter's ``anthropic/*`` routes, Bedrock's
# ``anthropic.*`` SKUs, Vertex AI's ``claude-*`` SKUs). Matching on provider
# alone isn't sufficient — the same providers also host non-Anthropic models
# (``openrouter/openai/*``, ``bedrock/amazon.titan-*``) that break on the
# content-block format. We additionally require the model name to carry
# ``claude`` or ``anthropic``.
_ANTHROPIC_PROXY_PROVIDERS = frozenset({"openrouter", "bedrock", "vertex_ai"})


@cache
def _supports_anthropic_cache_control(model: str) -> bool:
    """True when ``model`` accepts Anthropic ``cache_control`` markers.

    Used to gate ``inject_cache_breakpoints`` — see its docstring for why
    the gate is necessary. Unknown model strings return ``False`` so we
    default to the safe no-op.

    Covers direct Anthropic plus Anthropic-backed routes through
    OpenRouter / Bedrock / Vertex (all of which preserve ``cache_control``
    for Claude models). Non-Claude models on those same proxies stay
    gated out because they don't necessarily handle the content-block
    content shape that applying cache markers forces us into.

    Cached: called once per inference step, result is a pure function of
    the model string, and the distinct-model-string cardinality is low
    (agents typically reuse one or two).
    """
    try:
        model_name, provider, _, _ = litellm.get_llm_provider(model)
    except Exception:
        return False
    if provider == "anthropic":
        return True
    if provider in _ANTHROPIC_PROXY_PROVIDERS:
        lower = (model_name or model).lower()
        return "claude" in lower or "anthropic" in lower
    return False


def inject_cache_breakpoints(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
) -> None:
    """Annotate messages and tools with Anthropic ``cache_control`` breakpoints.

    Anthropic's prompt caching requires explicit ``cache_control`` markers
    on **content blocks** (not on message dicts) to create cache entries.
    Applying them means converting string ``content`` into a list of
    content blocks — because only blocks can carry the marker.

    **Gated on model.** The earlier implementation applied this
    unconditionally under the assumption that "LiteLLM strips
    cache_control for providers that don't support it." In practice
    LiteLLM strips the ``cache_control`` key but leaves the list-of-
    blocks content format, and some OpenAI-compatible servers (notably
    MLX-based local Qwen servers) silently return empty completions
    when a ``tool``-role message arrives as a content-block list. The
    gate keeps the feature for Anthropic-backed routes (direct
    Anthropic, plus ``openrouter/anthropic/*``, ``bedrock/anthropic.*``,
    and ``vertex_ai/claude-*`` — all of which forward cache markers to
    Anthropic) and leaves string content untouched for everyone else.

    Places breakpoints on:

    1. **System message** — cache-stable across steps.
    2. **Last tool definition** — cache-stable while tools don't change.
    3. **Last stable conversation message** — the last event-sourced
       message, skipping the trailing channels tail block (which
       mutates every step: unread counts, previews) and any
       empty-assistant separator inserted before it by
       :func:`separate_adjacent_user_messages`.

    Skipping the tail is load-bearing: with the breakpoint on the tail
    itself, the conversation prefix never gets its own cache entry and
    has to be re-cache-created every step.  Placing it on the last
    stable message lets the prefix cache across steps — next step's
    conversation-through-last-event is byte-identical and hits.
    """
    if not messages:
        return
    if not _supports_anthropic_cache_control(model):
        return

    if messages[0].get("role") == "system":
        _set_content_block_cache(messages[0])

    if tools:
        tools[-1]["cache_control"] = _CACHE_CONTROL

    idx = _last_stable_message_index(messages)
    if idx is not None and messages[idx].get("role") != "system":
        _set_content_block_cache(messages[idx])


def _last_stable_message_index(messages: list[dict[str, Any]]) -> int | None:
    """Return the index of the last cache-stable message, or ``None``.

    Walks backward from the end, skipping:

    * The channels tail block — identified by its content signature
      ``━━━ Channels ━━━`` (always the last user-role message when
      present).
    * Any empty-assistant separator — inserted by
      :func:`~aios.harness.context.separate_adjacent_user_messages` to
      defeat Anthropic's adjacent-user-merge; carries no real content
      and would be a wasted breakpoint.

    If nothing stable remains (messages list is just system + tail +
    separator), returns ``None``.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if _is_tail_block(msg) or _is_empty_assistant(msg):
            continue
        return i
    return None


def _is_tail_block(msg: dict[str, Any]) -> bool:
    """Detect the channels tail block by its content signature.

    The tail block renders with a ``━━━ Channels ━━━`` header as the
    first line of its user-role content.  That string is unlikely to
    appear in genuine peer text, so a substring-match is safe enough
    for cache-breakpoint placement.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.startswith("━━━ Channels ━━━")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.startswith("━━━ Channels ━━━"):
                    return True
    return False


def _is_empty_assistant(msg: dict[str, Any]) -> bool:
    """Detect the role-transition separator: ``assistant`` + empty content."""
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if content == "" or content is None:
        return True
    return isinstance(content, list) and not content


def _extract_cost(response: Any) -> float | None:
    """Pull the per-request USD cost LiteLLM computes post-call.

    LiteLLM populates ``response._hidden_params["response_cost"]`` during
    its logging pipeline. Missing attribute, missing key, or ``None``
    value all mean the provider didn't report cost — the harness passes
    ``None`` through rather than guessing.
    """
    hidden = getattr(response, "_hidden_params", None)
    if not hidden:
        return None
    cost = hidden.get("response_cost")
    if cost is None:
        return None
    return float(cost)


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
) -> tuple[dict[str, Any], dict[str, int], float | None]:
    """Call ``litellm.acompletion`` and return ``(message, usage, cost)``.

    Returns the message exactly as litellm produced it, including any
    provider-specific extensions like ``reasoning_content`` or
    ``thinking_blocks``. The harness stores the message dict opaquely.
    Usage is normalized to our canonical field names. Cost is LiteLLM's
    per-request USD figure, or ``None`` when the provider doesn't report it.
    """
    inject_cache_breakpoints(messages, tools, model)

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
    cost = _extract_cost(response)
    message = response["choices"][0]["message"]
    # litellm returns a Message object that supports model_dump()
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return _normalize_message(result), usage, cost
    if isinstance(message, dict):
        return _normalize_message(message), usage, cost
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
) -> tuple[dict[str, Any], dict[str, int], float | None]:
    """Call ``litellm.acompletion`` with streaming, returning ``(message, usage, cost)``.

    Each content delta fires a transient ``pg_notify`` on the session's
    event channel. SSE clients receive these as ``event: delta`` — no DB
    row is created. After the stream exhausts, the complete message is
    assembled via ``litellm.stream_chunk_builder`` and returned for
    storage as a normal event.
    """
    inject_cache_breakpoints(messages, tools, model)

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
    cost = _extract_cost(assembled)
    message = assembled["choices"][0]["message"]
    if hasattr(message, "model_dump"):
        result: dict[str, Any] = message.model_dump()
        return _normalize_message(result), usage, cost
    if isinstance(message, dict):
        return _normalize_message(message), usage, cost
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
