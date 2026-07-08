"""Token counting helpers.

* :func:`approx_tokens` — cost estimate for a sequence of chat-
  completions messages, delegating to :func:`litellm.token_counter`.
  Single source of truth: the ``cumulative_tokens`` column, context
  windowing, and per-tool budgeting (e.g. the ``switch_channel``
  recap floor) all go through this.

* :func:`tokens_to_drop` — snap boundary math for the DB-level windowed
  reader (:func:`~aios.db.queries.events.read_windowed_events`).

Performance (issue #1744): ``approx_tokens`` / ``approx_tokens_by_class``
are called once per step on the full message slate, but only the tail
changes step to step. Both are built from two memoized primitives —
``_message_body_tokens`` (per-message body cost) and ``_extra_tokens``
(payload-level system/tools framing overhead) — cached in
``_BODY_CACHE`` / ``_EXTRA_CACHE`` keyed by a content digest, so repeat
calls on an (almost) unchanged slate cost ~0 after the first warm pass.
The call site (``loop.py``) additionally runs the pair off the event
loop via ``asyncio.to_thread`` since ``Encoding.encode`` releases the
GIL and is stateless (safe to call concurrently from threads).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

# ─── memoization primitives ────────────────────────────────────────────────

# A stub system message used to measure payload-level "extra" overhead
# (tools schema + the chat-framing tokens litellm adds when a system
# message is present) in isolation from any real message's body cost.
_STUB_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": ""}

_BODY_CACHE_MAX = 65536
_BODY_CACHE: OrderedDict[bytes, int] = OrderedDict()
_EXTRA_CACHE: OrderedDict[bytes, int] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _payload_digest(obj: Any) -> bytes | None:
    """Stable content digest for cache keys.

    Returns ``None`` (cache-bypass sentinel) if ``obj`` can't be
    serialized deterministically — callers must fall back to counting
    directly rather than raising.
    """
    try:
        blob = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        ).encode()
    except Exception:
        return None
    return hashlib.blake2b(blob, digest_size=16).digest()


def _cache_get(cache: OrderedDict[bytes, int], key: bytes) -> int | None:
    with _CACHE_LOCK:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None


def _cache_put(cache: OrderedDict[bytes, int], key: bytes, value: int) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _BODY_CACHE_MAX:
            cache.popitem(last=False)


def _message_body_tokens(msg: Mapping[str, Any]) -> int:
    """The pure per-message body cost of ``msg`` (no payload-level framing).

    Memoized by content digest; digest failures bypass the cache and
    count directly (never raise).
    """
    from litellm import token_counter

    key = _payload_digest(msg)
    if key is not None:
        cached = _cache_get(_BODY_CACHE, key)
        if cached is not None:
            return cached

    value = int(token_counter(messages=[msg], count_response_tokens=True))

    if key is not None:
        _cache_put(_BODY_CACHE, key, value)
    return value


def _extra_tokens(tools: Any, system_present: bool) -> int:
    """Payload-level overhead: tool-schema cost plus (when a system
    message is present) the chat-framing tokens litellm adds for it.

    Computed as ``token_counter(messages=[STUB] if system_present else [],
    tools=tools) - (body(STUB) if system_present else 0)`` — i.e. the
    total minus the stub's own body cost, isolating the framing/tools
    overhead. Memoized by (tools digest, system_present).
    """
    from litellm import token_counter

    tool_list = list(tools) if tools else None
    digest_key = _payload_digest((tool_list, system_present))
    if digest_key is not None:
        cached = _cache_get(_EXTRA_CACHE, digest_key)
        if cached is not None:
            return cached

    if system_present:
        total = int(token_counter(messages=[_STUB_SYSTEM_MESSAGE], tools=tool_list))
        value = total - _message_body_tokens(_STUB_SYSTEM_MESSAGE)
    else:
        value = int(token_counter(messages=[], tools=tool_list))

    if digest_key is not None:
        _cache_put(_EXTRA_CACHE, digest_key, value)
    return value


# ─── estimator (delegates to litellm's local tokenizers) ──────────────────


def approx_tokens(
    messages: Iterable[Mapping[str, Any]],
    *,
    tools: Iterable[Mapping[str, Any]] | None = None,
) -> int:
    """Estimate the chat-completions token cost of ``messages``.

    Delegates to :func:`litellm.token_counter` with no ``model``
    argument, so the default tokenizer applies.  Accurate to within
    ~10 % across providers — fine for windowing boundaries and budget
    decisions, without coupling every call site to a specific model.

    Takes an iterable of chat-completions-shaped dicts, not raw
    strings: callers that want to cost a single message pass ``[msg]``.
    The canonical shape is a list, so passing a bare dict would be a
    bug (it'd iterate the dict's keys as messages).

    ``tools`` is optional and only passed by the model_request_end
    span-stamp call site so the recorded ``local_tokens`` matches the
    full payload the provider actually sees (messages + tools).  The
    per-event ``cumulative_tokens`` call sites in ``append_event`` do
    NOT pass tools: tool-schema overhead isn't per-event, and baking
    it into per-event counts would perturb the running sum whenever
    the agent's tool list changes.  The per-model ratio correction in
    ``read_windowed_events`` absorbs the per-request tools overhead at
    read time.

    ``cumulative_tokens`` storage depends on this formula.  If the
    implementation changes (e.g. a different tokenizer, passing
    ``model=...``), re-run the backfill script to keep stored values
    honest.

    Internally built from memoized per-message + per-payload
    primitives (issue #1744): returns byte-identical results to a
    direct ``litellm.token_counter(messages=..., tools=...)`` call,
    but repeat calls on an (almost) unchanged slate cost ~0 after the
    cache warms.
    """
    msgs = list(messages)
    total = sum(_message_body_tokens(m) for m in msgs)
    system_present = any(m.get("role") == "system" for m in msgs)
    total += _extra_tokens(tools, system_present)
    return total


# ─── per-content-class estimator (issue #1609) ────────────────────────────

# The content classes the per-(model, class) calibration learns coefficients
# for.  ``system`` and ``tools`` are payload-level overhead (one each per
# request); the remaining four partition the per-message body.  The set is
# fixed: the calibration fit and the windower's blend re-derive the same keys.
CONTENT_CLASSES: tuple[str, ...] = (
    "system",
    "tools",
    "text",
    "tool_result",
    "thinking",
    "tool_use",
)


def content_class(role: str | None, data: Mapping[str, Any]) -> str:
    """Classify a single chat-completions message into a content class.

    Returns the dominant body class for the message:

    * ``"system"`` — the system prompt turn.
    * ``"tool_result"`` — a ``role == "tool"`` result message.
    * ``"tool_use"`` — an assistant turn carrying ``tool_calls`` (the
      serialized function-call request the provider re-tokenizes).
    * ``"thinking"`` — an assistant turn carrying reasoning content
      (``reasoning_content`` / ``thinking_blocks``) but no tool calls.
    * ``"text"`` — any other message (user turns, plain assistant text).

    This is the *message-level* dominant class used by the windower to
    re-derive the retained-slate composition from role+data already
    loaded.  ``approx_tokens_by_class`` splits a single message's tokens
    across classes finer-grained than this when a turn mixes (e.g. an
    assistant turn that has both thinking and a tool call).
    """
    if role == "system":
        return "system"
    if role == "tool":
        return "tool_result"
    if role == "assistant":
        if data.get("tool_calls"):
            return "tool_use"
        if any(data.get(f) for f in ("reasoning_content", "thinking_blocks")):
            return "thinking"
    return "text"


def _count(messages: list[Mapping[str, Any]], *, tools: Any = None) -> int:
    """Single-payload count, sourced from the same memoized primitives as
    :func:`approx_tokens` (issue #1744).

    ``_count([m])`` == ``_message_body_tokens(m) + _extra_tokens(None, m.role
    == "system")``; ``_count([], tools)`` == ``_extra_tokens(tools, False)``.
    """
    if not messages and not tools:
        return 0
    if len(messages) == 1:
        msg = messages[0]
        system_present = msg.get("role") == "system"
        return _message_body_tokens(msg) + _extra_tokens(tools, system_present)
    if not messages:
        return _extra_tokens(tools, False)
    # General fallback (not used by the current call sites, which only ever
    # pass 0 or 1 messages here, but keep correct for any future caller).
    total = sum(_message_body_tokens(m) for m in messages)
    system_present = any(m.get("role") == "system" for m in messages)
    total += _extra_tokens(tools, system_present)
    return total


def approx_tokens_by_class(
    messages: Iterable[Mapping[str, Any]],
    *,
    tools: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Split the local token cost of ``messages`` (+ ``tools``) by class.

    Returns a dict over :data:`CONTENT_CLASSES`.  By construction the
    sum of the values reconciles to :func:`approx_tokens` of the same
    payload (modulo small per-message framing overhead, which is folded
    into the message's dominant class), so the call site can compute
    ``local_tokens = sum(by_class.values())`` and stay consistent with
    the model-neutral stored baseline (issue #1609, constraint #1: the
    neutral counter is unchanged — this only *attributes* its output).

    The split is deliberately coarse and model-neutral: it uses the same
    ``litellm.token_counter`` (no ``model=``) as the baseline, costing
    one isolated single-message payload per class slice.  Per-class
    *coefficients* (the provider/local correction) are learned downstream
    by least squares over logged usage spans, not here.
    """
    msgs = list(messages)
    tool_list = list(tools) if tools else None

    by_class: dict[str, int] = {c: 0 for c in CONTENT_CLASSES}

    # Tool-schema overhead: cost the tools payload alone (an empty-message
    # request would add framing tokens, so cost tools as the delta against
    # a bare single user turn is avoided — token_counter with only tools
    # returns the schema cost directly).
    if tool_list:
        by_class["tools"] = _count([], tools=tool_list)

    for msg in msgs:
        role = msg.get("role")
        cls = content_class(role, msg)
        if cls == "system":
            by_class["system"] += _count([msg])
            continue
        if cls == "tool_result":
            by_class["tool_result"] += _count([msg])
            continue
        if cls == "tool_use":
            # An assistant tool-call turn may also carry thinking and/or
            # leading text.  Attribute the thinking/text portions to their
            # classes and the remainder (the serialized tool_calls) to
            # tool_use, so a thinking+tool_use turn trains both coefficients.
            _split_assistant(msg, by_class, primary="tool_use")
            continue
        if cls == "thinking":
            _split_assistant(msg, by_class, primary="thinking")
            continue
        # Plain text (user turns, plain assistant text, orphan tool
        # placeholders rendered as user messages, etc.).
        by_class["text"] += _count([msg])

    return by_class


def _split_assistant(
    msg: Mapping[str, Any],
    by_class: dict[str, int],
    *,
    primary: str,
) -> None:
    """Attribute an assistant turn's tokens across text/thinking/tool_use.

    Costs the turn's text content and its thinking content as isolated
    slices, then assigns the *residual* (full-turn cost minus the slices,
    floored at 0) to ``primary`` — the dominant class of the turn.  This
    keeps the per-class sum reconciled to the full-turn cost while still
    crediting each sub-class that is present.
    """
    full = _count([msg])

    text_content = msg.get("content")
    text_tokens = 0
    if text_content:
        text_tokens = _count([{"role": "assistant", "content": text_content}])

    thinking_tokens = 0
    reasoning = msg.get("reasoning_content")
    if reasoning:
        thinking_tokens = _count([{"role": "assistant", "content": reasoning}])

    by_class["text"] += text_tokens
    by_class["thinking"] += thinking_tokens
    residual = full - text_tokens - thinking_tokens
    if residual < 0:
        residual = 0
    by_class[primary] += residual


# ─── snap boundary math ───────────────────────────────────────────────────


def tokens_to_drop(total: int, *, window_min: int, window_max: int) -> int:
    """Compute how many tokens to drop from the front of a context window.

    Implements the chunked snap policy (described in :mod:`aios.harness.window`):
    drop in ``(max - min)``-sized chunks so the cutoff advances
    monotonically and prefix caching stays stable within a chunk.

    Returns 0 when the total fits within ``window_max``.
    """
    if total <= window_max:
        return 0
    overshoot = total - window_max
    chunk = window_max - window_min
    snaps = (overshoot + chunk - 1) // chunk  # ceil division
    return snaps * chunk
