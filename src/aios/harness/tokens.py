"""Token counting helpers.

* :func:`approx_tokens` — fast, dependency-free ``len // 4`` estimate
  used by the cumulative-tokens column and context windowing.  Suitable
  for boundary decisions where ±10 % accuracy is fine.

* :func:`tokens_to_drop` — snap boundary math shared by the DB-level
  windowed reader and the pure-function ``select_window``.
"""

from __future__ import annotations

from typing import Any

# ─── cheap estimator (no external deps) ────────────────────────────────────


def approx_tokens(message: dict[str, Any]) -> int:
    """Rough token count for a chat-completions message dict.

    Counts characters in ``content`` plus ``tool_calls[].function.{name,
    arguments}`` and divides by 4.  Returns at least 1.

    This is the single source of truth for the ``cumulative_tokens``
    column on the events table and for the chunked-window boundary
    computation.  If the formula changes, run the backfill script to
    recompute stored values.
    """
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    total_chars = len(content)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        total_chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return max(1, total_chars // 4)


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
