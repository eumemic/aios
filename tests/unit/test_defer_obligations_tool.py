"""Unit tests for the ``defer_obligations`` tool's boundary (#1533).

No DB: these pin the tool's argument contract and registration flags —

* **session-wide, no ``request_id``** — the arg model is ``extra="forbid"``,
  so a per-obligation ``request_id`` (or a smuggled ``caller``/``account_id``)
  is rejected at the schema boundary before the handler runs;
* **shared bound validation** — ``duration_seconds`` below 1 and above the
  configured ``schedule_wake_max_delay_seconds`` ceiling both raise the typed
  :class:`ScheduleWakeArgumentError` from the REUSED ``schedule_wake``
  resolver (``_resolve_fire_at``), never a re-implemented bound; the ``*1000``
  typo (``86400000``) is rejected at this boundary;
* **registration** — ``transport="agent_tool"`` and ``resumable=True`` (the
  ghost-repair sweep's re-park discriminant, which routes a crashed in-flight
  defer to the retryable ``launch_lost`` branch).

The active-wait semantics themselves (in-flight for the duration, early
resolve on an inbound stimulus, un-nudged while open) are DB-backed and live
in ``tests/integration/test_defer_obligations.py``.
"""

from __future__ import annotations

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.config import get_settings
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import registry
from aios.tools.schedule_wake import ScheduleWakeArgumentError

_SESSION = "ses_defer_unit"


# ─── argument bounds: reuse of the schedule_wake resolver ─────────────────────


async def test_zero_duration_rejected_by_shared_resolver() -> None:
    with pytest.raises(ScheduleWakeArgumentError, match="positive integer"):
        await invoke_builtin(_SESSION, "defer_obligations", {"duration_seconds": 0})


async def test_negative_duration_rejected_by_shared_resolver() -> None:
    with pytest.raises(ScheduleWakeArgumentError, match="positive integer"):
        await invoke_builtin(_SESSION, "defer_obligations", {"duration_seconds": -5})


async def test_duration_above_ceiling_rejected() -> None:
    """The 86400*1000 typo class: over the configured max is rejected at the
    boundary by the SAME ceiling ``schedule_wake`` enforces."""
    over_max = get_settings().schedule_wake_max_delay_seconds + 1
    with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
        await invoke_builtin(_SESSION, "defer_obligations", {"duration_seconds": over_max})
    with pytest.raises(ScheduleWakeArgumentError, match="exceeds the max allowed"):
        await invoke_builtin(_SESSION, "defer_obligations", {"duration_seconds": 86_400_000})


# ─── session-wide contract: no request_id, no smuggled identity ───────────────


async def test_request_id_argument_is_rejected() -> None:
    """Per-obligation targeting is explicitly forbidden by the settled design:
    the defer is session-wide, and a ``request_id`` key fails ``extra=forbid``."""
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _SESSION, "defer_obligations", {"duration_seconds": 60, "request_id": "req_1"}
        )


async def test_smuggled_identity_is_rejected() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _SESSION, "defer_obligations", {"duration_seconds": 60, "account_id": "acc_evil"}
        )
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _SESSION,
            "defer_obligations",
            {"duration_seconds": 60, "caller": {"kind": "session", "id": "evil"}},
        )


async def test_missing_duration_is_rejected() -> None:
    with pytest.raises(ToolBail):
        await invoke_builtin(_SESSION, "defer_obligations", {})


# ─── registration flags ───────────────────────────────────────────────────────


def test_registered_agent_tool_and_resumable() -> None:
    """``resumable=True`` is the crash-recovery contract: the ghost sweep
    re-park path finds no servicer edge for a defer and lands the retryable
    ``launch_lost`` result instead of the pessimistic may-have-completed one."""
    tool = registry.get("defer_obligations")
    assert tool.transport == "agent_tool"
    assert tool.resumable is True
    assert "defer_obligations" in registry.resumable_tool_names()
    # Session-wide: the schema has exactly one property and forbids extras.
    assert set(tool.parameters_schema.get("properties", {})) == {"duration_seconds"}
    assert tool.parameters_schema.get("additionalProperties") is False
