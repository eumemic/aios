"""Per-step current-time tail block.

A long-lived agent needs to know what "now" is to reason about dates and
times — scheduling reminders, answering "what's today", stamping notes —
without guessing (models routinely hallucinate the date) or paying a tool
round-trip just to read a clock.

The harness injects the current UTC time as an *ephemeral tail message*:
appended AFTER :func:`~aios.harness.context.build_messages`, exactly like the
channels tail block, so its per-step value never mutates the cache-stable
prompt prefix. Conversion to the user's local timezone is the agent's job
(the timezone lives in the agent's instructions/memory, not here), so this
block stays a single, locale-free UTC line.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.tokens import approx_tokens

# Stable leading marker, shared between the producer (:func:`build_time_block`)
# and the cache-breakpoint recognizer (``_is_time_block`` in completion.py) so
# the two can never drift — the same producer/recognizer coupling the channels
# tail (``━━━ Channels ━━━``) and the separator placeholder already rely on.
# It also frames the line as harness metadata so the model never reads this
# user-role message as something the human said — load-bearing on no-channels
# sessions, where no other framing prose is present.
TIME_BLOCK_PREFIX = "[current time] "


def build_time_block(now: datetime) -> dict[str, Any]:
    """Ephemeral tail message stating the current UTC time.

    Rendered as a user-role message (matching the channels tail) and appended
    after ``build_messages``. Two safety properties, both shared with the
    channels tail and both load-bearing:

    * Cache-safe — completion.py's breakpoint walker skips it (via
      :data:`TIME_BLOCK_PREFIX`), so its per-step mutation never anchors the
      conversation cache breakpoint.
    * Unambiguous — the ``[current time]`` marker + the explicit disclaimer
      keep the model from attributing this to the human.

    Includes the weekday because relative reasoning ("next Thursday") needs it.
    """
    return {
        "role": "user",
        "content": (
            f"{TIME_BLOCK_PREFIX}{now:%Y-%m-%d %H:%M:%S} UTC ({now:%A}) "
            "— system clock, not a message from anyone."
        ),
    }


# Worst-case local-token cost of :func:`build_time_block`. The date/time
# portion is fixed-width; only the weekday name varies (6 to 9 chars), so the
# fattest of any 7 consecutive days is an exact upper bound. Reserved from
# the window budget alongside ``max_tail_block_local`` so the composed
# payload never exceeds ``window_max``.
TIME_BLOCK_MAX_LOCAL: int = max(
    approx_tokens([build_time_block(datetime(2026, 1, day, 23, 59, 59, tzinfo=UTC))])
    for day in range(1, 8)
)
