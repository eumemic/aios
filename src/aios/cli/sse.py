"""SSE parser re-export shim.

The canonical home for ``SseMessage`` and ``parse_sse_lines`` is
:mod:`aios_sdk.streaming` — the SDK owns the streaming surface so plugin
authors and external scripts have a single import root. This module
stays as a thin re-export so existing imports inside ``aios.cli`` keep
working without churn during the Phase C refactor (#267).
"""

from __future__ import annotations

from aios_sdk.streaming import SseMessage, parse_sse_lines

__all__ = ["SseMessage", "parse_sse_lines"]
