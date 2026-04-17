"""Event resource: append-only entries on the session log.

Events come in four kinds, distinguished by `kind`:

* ``message`` — a chat-completions message dict (whatever LiteLLM returns,
  stored opaquely so reasoning_content / thinking_blocks come along for free)
* ``lifecycle`` — session state transitions (turn started/ended, status
  changes, stop_reason)
* ``span`` — observability markers around model calls and tool calls
* ``interrupt`` — user-issued cancel signal

The `data` field is intentionally opaque (`dict[str, Any]`) so we don't
over-validate at the boundary. Per-kind shapes are documented but not
enforced via pydantic discriminated unions, because the message kind in
particular has to round-trip arbitrary LiteLLM extensions without rejecting
them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal["message", "lifecycle", "span", "interrupt"]


class Event(BaseModel):
    """Read view of a single event from the session log."""

    id: str
    session_id: str
    seq: int
    kind: EventKind
    data: dict[str, Any]
    cumulative_tokens: int | None = Field(default=None, exclude=True)
    created_at: datetime
    orig_channel: str | None = Field(default=None, exclude=True)
    focal_channel_at_arrival: str | None = Field(default=None, exclude=True)
