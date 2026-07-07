"""Direct-to-gateway callers for the $0 cost-routing map (NO aios runtime, NO OpenRouter).

Both gateways are OUR subsidized proxies (zero cash spend):

  * ant-proxy (https://ant-proxy.eumemic.ai) — Anthropic Messages API, SSE streaming.
    Streaming sidesteps the "non-streaming requires max_tokens <= N" provider rule and
    carries exact usage (message_start.usage.input_tokens + message_delta.usage.output_tokens).
  * oai-proxy (https://oai-proxy.eumemic.ai) — OpenAI **Responses** API, SSE streaming.
    The deployed proxy's non-stream timeout is 120s (too short for hard reasoning items)
    vs 600s for streams; the terminal ``response.completed`` event carries exact usage
    including ``reasoning_tokens`` — the chat-completions stream carries none.

ADMINISTRATION RULES (Stage-0 fraud-class defenses, enforced here at the call site):
  * Provider-default decoding EVERYWHERE: temperature/top_p/top_k are NEVER sent
    (opus/fable 400 on them; litellm-style layers silently drop rejected params —
    aios#1674). k-sampling rides native stochasticity; the entropy probe records it.
  * Identity assertion on EVERY call: the response's model field must equal the
    requested model (date-suffix resolution allowed, e.g. gpt-5.5 ->
    gpt-5.5-2026-xx-xx, claude-haiku-4-5 -> claude-haiku-4-5-20251001). A mismatch
    quarantines the call (it is an error, never a scoreable sample).
  * Error != wrong: 429/5xx/timeout/stream-cut are retried with bounded backoff and
    then recorded as a HOLE by the caller — never as an incorrect answer.
  * Truncation (stop_reason max_tokens / status incomplete) is surfaced verbatim so
    the caller can score it as a HOLE, not wrong.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

import requests

ANT_BASE = os.environ.get("ANT_PROXY_BASE", "https://ant-proxy.eumemic.ai")
OAI_BASE = os.environ.get("OAI_PROXY_BASE", "https://oai-proxy.eumemic.ai")

RETRY_BACKOFF_S = (10.0, 30.0, 90.0)  # attempts = len+1


class GatewayError(Exception):
    """A single call attempt failed."""

    def __init__(self, kind: str, detail: str, retryable: bool):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.retryable = retryable


class Hole(Exception):
    """All attempts exhausted (or permanent error): the cell is a HOLE, not a wrong answer."""

    def __init__(self, kind: str, detail: str, attempts: int):
        super().__init__(f"HOLE({kind}) after {attempts} attempts: {detail}")
        self.kind = kind
        self.detail = detail
        self.attempts = attempts


class QuotaExhausted(Exception):
    """The oai-proxy account window is exhausted — halt the lane, do NOT busy-wait."""


@dataclass
class CallResult:
    text: str
    served_model: str
    stop_reason: str  # normalized: end_turn | max_tokens | <other provider reason>
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_s: float
    attempts: int = 1
    identity_ok: bool = True
    raw_events: int = 0
    extra: dict = field(default_factory=dict)


def _ant_key() -> str:
    k = os.environ.get("ANT_PROXY_CLIENT_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise RuntimeError("no ant-proxy client key (ANT_PROXY_CLIENT_KEY / ANTHROPIC_API_KEY)")
    return k


def _oai_key() -> str:
    k = os.environ.get("OAI_CLIENT_KEY")
    if not k:
        raise RuntimeError("no oai-proxy client key in env OAI_CLIENT_KEY (sk-op-...)")
    return k


def identity_matches(requested: str, served: str | None) -> bool:
    """Exact match, or a DATE-suffix resolution of the same model id.

    Strict on purpose: ``gpt-5.4`` -> ``gpt-5.4-2026-03-17`` matches, but
    ``gpt-5.4`` -> ``gpt-5.4-mini-2026-03-17`` must NOT (a cheaper sibling being
    silently served is exactly the fraud class the identity assertion exists for).
    """
    import re

    if not served:
        return False
    return served == requested or bool(re.match(rf"^{re.escape(requested)}-\d{{4}}", served))


def _sse_data_lines(resp: requests.Response):
    """Yield parsed JSON payloads of ``data:`` SSE lines; ignore keepalives/[DONE]."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue  # partial keepalive garbage; the terminal-event check catches real cuts


def _classify_http(status: int, body: str) -> GatewayError:
    retryable = status == 429 or status >= 500 or status == 408
    return GatewayError(f"http_{status}", body[:300], retryable)


# ── ant-proxy: Anthropic Messages API (streaming) ────────────────────────────
def ant_call(
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    timeout_s: float = 900.0,
) -> CallResult:
    t0 = time.time()
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "stream": True,
        # NO temperature / top_p / top_k — provider-default decoding (see module doc).
    }
    headers = {
        "x-api-key": _ant_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        resp = requests.post(
            f"{ANT_BASE}/v1/messages",
            headers=headers,
            json=body,
            stream=True,
            timeout=(20, timeout_s),
        )
    except requests.RequestException as exc:
        raise GatewayError("connect", f"{type(exc).__name__}: {exc}", retryable=True)
    with resp:
        if resp.status_code != 200:
            raise _classify_http(resp.status_code, resp.text)
        text_parts: list[str] = []
        served = stop_reason = None
        in_tok = out_tok = None
        n_events = 0
        got_stop = False
        try:
            for ev in _sse_data_lines(resp):
                n_events += 1
                t = ev.get("type")
                if t == "message_start":
                    msg = ev.get("message", {})
                    served = msg.get("model")
                    in_tok = (msg.get("usage") or {}).get("input_tokens")
                elif t == "content_block_delta":
                    d = ev.get("delta", {})
                    if d.get("type") == "text_delta":
                        text_parts.append(d.get("text", ""))
                elif t == "message_delta":
                    stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
                    u = ev.get("usage") or {}
                    out_tok = u.get("output_tokens", out_tok)
                elif t == "message_stop":
                    got_stop = True
                elif t == "error":
                    err = ev.get("error", {})
                    kind = err.get("type", "stream_error")
                    retryable = kind in ("overloaded_error", "api_error", "internal_server_error")
                    raise GatewayError(kind, str(err)[:300], retryable=retryable)
        except requests.RequestException as exc:  # mid-stream cut
            raise GatewayError("stream_cut", f"{type(exc).__name__}: {exc}", retryable=True)
        if not got_stop:
            raise GatewayError("stream_incomplete", "no message_stop event", retryable=True)
    return CallResult(
        text="".join(text_parts),
        served_model=served or "",
        stop_reason=stop_reason or "unknown",
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=None,
        latency_s=round(time.time() - t0, 2),
        identity_ok=identity_matches(model, served),
        raw_events=n_events,
    )


# ── oai-proxy: OpenAI Responses API (streaming) ──────────────────────────────
def _oai_input_items(messages: list[dict]) -> list[dict]:
    items = []
    for m in messages:
        if m["role"] == "assistant":
            items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": m["content"]}],
                }
            )
        else:
            items.append(
                {
                    "role": m["role"],
                    "content": [{"type": "input_text", "text": m["content"]}],
                }
            )
    return items


def oai_call(
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
    timeout_s: float = 650.0,
) -> CallResult:
    t0 = time.time()
    body = {
        "model": model,
        "instructions": system,
        "input": _oai_input_items(messages),
        "max_output_tokens": max_tokens,
        "stream": True,
        # NO temperature / top_p / top_k — provider-default decoding (see module doc).
    }
    headers = {
        "Authorization": f"Bearer {_oai_key()}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{OAI_BASE}/v1/responses",
            headers=headers,
            json=body,
            stream=True,
            timeout=(20, timeout_s),
        )
    except requests.RequestException as exc:
        raise GatewayError("connect", f"{type(exc).__name__}: {exc}", retryable=True)
    with resp:
        if resp.status_code != 200:
            btxt = resp.text
            if resp.status_code == 429 and ("usage" in btxt.lower() or "limit" in btxt.lower()):
                raise QuotaExhausted(btxt[:300])
            raise _classify_http(resp.status_code, btxt)
        final = None
        n_events = 0
        failed_detail = None
        try:
            for ev in _sse_data_lines(resp):
                n_events += 1
                t = ev.get("type", "")
                if t in ("response.completed", "response.incomplete"):
                    final = ev.get("response", {})
                elif t in ("response.failed", "error"):
                    failed_detail = json.dumps(ev)[:300]
        except requests.RequestException as exc:
            raise GatewayError("stream_cut", f"{type(exc).__name__}: {exc}", retryable=True)
    if failed_detail:
        raise GatewayError("response_failed", failed_detail, retryable=True)
    if final is None:
        raise GatewayError("stream_incomplete", "no response.completed event", retryable=True)

    status = final.get("status")
    text_parts = []
    for item in final.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text_parts.append(c.get("text", ""))
    usage = final.get("usage") or {}
    out_details = usage.get("output_tokens_details") or {}
    if status == "incomplete":
        reason = (final.get("incomplete_details") or {}).get("reason", "incomplete")
        stop = "max_tokens" if reason == "max_output_tokens" else reason
    else:
        stop = "end_turn"
    served = final.get("model")
    return CallResult(
        text="".join(text_parts),
        served_model=served or "",
        stop_reason=stop,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=out_details.get("reasoning_tokens"),
        latency_s=round(time.time() - t0, 2),
        identity_ok=identity_matches(model, served),
        raw_events=n_events,
    )


# ── bounded retry wrapper (error != wrong) ───────────────────────────────────
def call_with_retry(gw: str, model: str, system: str, messages: list[dict],
                    max_tokens: int, timeout_s: float) -> CallResult:
    """Retry retryable failures with bounded backoff; raise Hole when exhausted.
    QuotaExhausted propagates immediately (lane-halt, not a hole)."""
    fn = ant_call if gw == "ant" else oai_call
    last: GatewayError | None = None
    attempts = 0
    for i in range(len(RETRY_BACKOFF_S) + 1):
        attempts = i + 1
        try:
            r = fn(model, system, messages, max_tokens, timeout_s)
            r.attempts = attempts
            return r
        except QuotaExhausted:
            raise
        except GatewayError as exc:
            last = exc
            if not exc.retryable:
                raise Hole(exc.kind, exc.detail, attempts)
            if i < len(RETRY_BACKOFF_S):
                time.sleep(RETRY_BACKOFF_S[i] + random.uniform(0, 5))
    assert last is not None
    raise Hole(last.kind, last.detail, attempts)


def oai_pool_state() -> dict | None:
    """Best-effort window check via the admin API (between batches, never in a hot loop)."""
    token = os.environ.get("OAI_PROXY_ADMIN_API_TOKEN")
    if not token:
        return None
    try:
        r = requests.get(
            f"{OAI_BASE}/admin/api/pool",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None
