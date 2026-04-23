"""Token counting helpers.

* :func:`approx_tokens` — cost estimate for a sequence of chat-
  completions messages, delegating to :func:`litellm.token_counter`.
  Single source of truth: the ``cumulative_tokens`` column, context
  windowing, and per-tool budgeting (e.g. the ``switch_channel``
  recap floor) all go through this.

* :func:`tokens_to_drop` — snap boundary math shared by the DB-level
  windowed reader and the pure-function ``select_window``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from litellm import token_counter

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
    the agent's tool list changes.  The per-model ratio correction
    (issue #160) absorbs the per-request tools overhead at read time.

    ``cumulative_tokens`` storage depends on this formula.  If the
    implementation changes (e.g. a different tokenizer, passing
    ``model=...``), re-run the backfill script to keep stored values
    honest.
    """
    kwargs: dict[str, Any] = {"messages": list(messages)}
    if tools:
        kwargs["tools"] = list(tools)
    return int(token_counter(**kwargs))


# ─── snap boundary math ───────────────────────────────────────────────────


def tokens_to_drop(total: int, *, window_min: int, window_max: int) -> int:
    """Compute how many tokens to drop from the front of a context window.

    Uses the chunked snap policy from :mod:`aios.harness.window`:
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
