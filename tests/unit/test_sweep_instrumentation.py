"""Unit tests for sweep-SQL instrumentation (issue #139, prereq for #132).

Covers:

- ``sweep_start``/``sweep_end`` span pair at the **tail site**
  (``_trigger_sweep`` in ``tool_dispatch``), with real ``repaired_ghosts``
  and ``woken_sessions`` counts from :class:`SweepResult`.
- Same pair at the **entry site** (``_run_session_step_body`` guard),
  where ``repaired_ghosts`` is always 0 and ``woken_sessions`` is 0 or 1
  (load-bearing: 0 marks a wasted wake).
- ``sweep_end`` fires on exception via ``try/finally``.
- The periodic "all"-scope sweep in ``worker._periodic_sweep`` does
  **not** emit a sweep span — spans are per-session and the periodic
  sweep scans across sessions, so its instrumentation is deferred.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _span_events(append_event: AsyncMock) -> list[dict[str, object]]:
    """Extract every span payload (the 4th positional arg) in order."""
    return [call.args[3] for call in append_event.await_args_list if call.args[2] == "span"]


def _sweep_events(append_event: AsyncMock) -> list[dict[str, object]]:
    return [
        payload
        for payload in _span_events(append_event)
        if payload.get("event", "").startswith("sweep_")
    ]


# ─── tail site (tool_dispatch._trigger_sweep) ────────────────────────────────


class TestTailSweepSpan:
    async def test_emits_pair_with_counts_from_sweep_result(self) -> None:
        from aios.harness.sweep import SweepResult
        from aios.harness.tool_dispatch import _trigger_sweep

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        wake_mock = AsyncMock(return_value=SweepResult(repaired_ghosts=2, woken_sessions=5))

        with (
            patch("aios.harness.tool_dispatch.sessions_service.append_event", append_event),
            patch(
                "aios.harness.tool_dispatch.runtime.require_task_registry", return_value=MagicMock()
            ),
            patch("aios.harness.sweep.wake_sessions_needing_inference", wake_mock),
        ):
            await _trigger_sweep(MagicMock(), "sess_x", MagicMock())

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[0] == {"event": "sweep_start", "site": "tail"}
        assert sweep_events[1] == {
            "event": "sweep_end",
            "sweep_start_id": "ev_sweep",
            "repaired_ghosts": 2,
            "woken_sessions": 5,
        }

    async def test_sweep_end_fires_when_sweep_raises(self) -> None:
        """``_trigger_sweep`` catches and logs sweep failures, so the pair
        still closes cleanly — but the counts fall back to zero because
        no ``SweepResult`` was produced."""
        from aios.harness.tool_dispatch import _trigger_sweep

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        wake_mock = AsyncMock(side_effect=RuntimeError("db down"))
        bound_log = MagicMock()

        with (
            patch("aios.harness.tool_dispatch.sessions_service.append_event", append_event),
            patch(
                "aios.harness.tool_dispatch.runtime.require_task_registry", return_value=MagicMock()
            ),
            patch("aios.harness.sweep.wake_sessions_needing_inference", wake_mock),
        ):
            await _trigger_sweep(MagicMock(), "sess_x", bound_log)

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[1]["repaired_ghosts"] == 0
        assert sweep_events[1]["woken_sessions"] == 0
        bound_log.warning.assert_called_once_with("tool.sweep_failed")


# ─── entry site (loop._run_session_step_body guard) ──────────────────────────


@pytest.fixture
def mock_runtime() -> Iterator[None]:
    with (
        patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
        patch(
            "aios.harness.loop.runtime.require_task_registry",
            return_value=MagicMock(),
        ),
    ):
        yield


class TestEntrySweepSpan:
    async def test_wasted_wake_marks_woken_sessions_zero(self, mock_runtime: None) -> None:
        """Guard early-out: find_sessions_needing_inference returns empty set.
        ``sweep_end`` stamps ``woken_sessions=0`` — the profiler's wasted-wake signal.
        """
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        with (
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value=set()),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
        ):
            await run_session_step("sess_x", cause="message")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[0] == {"event": "sweep_start", "site": "entry"}
        assert sweep_events[1] == {
            "event": "sweep_end",
            "sweep_start_id": "ev_sweep",
            "repaired_ghosts": 0,
            "woken_sessions": 0,
        }

    async def test_happy_path_marks_woken_sessions_one(self) -> None:
        """Guard passes: session is in the needs set. ``woken_sessions=1``
        marks "keep going" for the profiler."""
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x", agent_id="agt_x", agent_version=None, focal_channel=None
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
        )
        start_event = SimpleNamespace(id="ev_sweep")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=MagicMock()),
            patch(
                "aios.harness.loop.runtime.require_task_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.get_agent",
                AsyncMock(return_value=agent),
            ),
            patch(
                "aios.harness.channels.list_bindings_and_connections",
                AsyncMock(return_value=([], [])),
            ),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop._dispatch_confirmed_tools",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.compose_step_context",
                AsyncMock(
                    return_value=SimpleNamespace(
                        model="openrouter/x",
                        messages=[],
                        tools=[],
                        reacting_to=0,
                        skill_versions=[],
                    )
                ),
            ),
            patch("aios.harness.loop.sessions_service.set_session_status", AsyncMock()),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=start_event),
            ) as append_event,
            patch(
                "aios.harness.loop.call_litellm",
                AsyncMock(return_value=({"role": "assistant", "content": "ok"}, {}, 0.0)),
            ),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=({"role": "assistant", "content": "ok"}, {}, 0.0)),
            ),
            patch(
                "aios.harness.loop.sessions_service.increment_usage",
                AsyncMock(),
            ),
            patch(
                "aios.db.sse_lock.has_subscriber",
                AsyncMock(return_value=False),
            ),
        ):
            await run_session_step("sess_x")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[0] == {"event": "sweep_start", "site": "entry"}
        assert sweep_events[1] == {
            "event": "sweep_end",
            "sweep_start_id": "ev_sweep",
            "repaired_ghosts": 0,
            "woken_sessions": 1,
        }

    async def test_sweep_end_fires_when_find_raises(self, mock_runtime: None) -> None:
        """If ``find_sessions_needing_inference`` raises, the outer try/finally
        in ``run_session_step`` still emits ``step_end``, and the body's own
        try/finally still emits ``sweep_end`` (with ``woken_sessions=0``)."""
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        with (
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            pytest.raises(RuntimeError, match="db down"),
        ):
            await run_session_step("sess_x", cause="message")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[1]["woken_sessions"] == 0
        assert sweep_events[1]["repaired_ghosts"] == 0


# ─── periodic sweep (regression fence on the "deferred entirely" decision) ───


class TestPeriodicSweepEmitsNoSpan:
    async def test_periodic_sweep_does_not_emit_sweep_span(self) -> None:
        """The 30s ``_periodic_sweep`` in worker.py runs at worker scope,
        scanning across all sessions. Spans are per-session; there is no
        natural ``session_id`` to stamp against, so instrumentation is
        deferred. Regression fence: catch any accidental emission."""
        from aios.harness.sweep import SweepResult
        from aios.harness.worker import _periodic_sweep

        append_event = AsyncMock()
        wake_mock = AsyncMock(return_value=SweepResult(repaired_ghosts=0, woken_sessions=0))

        with (
            patch(
                "aios.harness.worker.wake_sessions_needing_inference",
                wake_mock,
            ),
            # Make ``asyncio.sleep`` raise on the first sleep to break the
            # infinite loop after exactly one iteration.
            patch("aios.harness.worker.asyncio.sleep", AsyncMock(side_effect=StopAsyncIteration)),
            patch("aios.services.sessions.append_event", append_event),
            pytest.raises(StopAsyncIteration),
        ):
            await _periodic_sweep(MagicMock(), MagicMock(), interval=30)

        # The periodic sweep must not emit any sweep_start/sweep_end spans.
        for call in append_event.await_args_list:
            assert call.args[2] != "span" or not call.args[3].get("event", "").startswith("sweep_")
