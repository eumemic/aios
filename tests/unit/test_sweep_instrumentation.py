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

from aios.harness.completion import LlmResponse
from aios.harness.window import WindowedEvents
from aios.services.sessions import AssistantAppendResult


def _llm_response(
    message: dict[str, object],
    *,
    usage: dict[str, int] | None = None,
    cost: float | None = 0.0,
    finish_reason: str | None = None,
) -> LlmResponse:
    """Build an ``LlmResponse`` the way ``call_litellm``/``stream_litellm`` do."""
    return LlmResponse.from_message(
        dict(message),
        usage=usage or {},
        cost=cost,
        finish_reason=finish_reason,
    )


def _span_events(append_event: AsyncMock) -> list[dict[str, object]]:
    """Extract every span payload (the 4th positional arg) in order."""
    return [call.args[3] for call in append_event.await_args_list if call.args[2] == "span"]


def _sweep_events(append_event: AsyncMock) -> list[dict[str, object]]:
    result = []
    for payload in _span_events(append_event):
        event = payload.get("event")
        if isinstance(event, str) and event.startswith("sweep_"):
            result.append(payload)
    return result


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
                "aios.harness.tool_dispatch.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch("aios.harness.sweep.wake_sessions_needing_inference", wake_mock),
        ):
            await _trigger_sweep(MagicMock(), "sess_x", account_id="acc_test_stub")

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
        """Sweep failures propagate out of ``_trigger_sweep``, but the
        ``sweep_start``/``sweep_end`` pair still closes cleanly via the
        outer try/finally — counts fall back to zero because no
        ``SweepResult`` was produced. Mirrors the entry-site behavior in
        ``test_sweep_end_fires_when_find_raises``."""
        from aios.harness.tool_dispatch import _trigger_sweep

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        wake_mock = AsyncMock(side_effect=RuntimeError("db down"))

        with (
            patch("aios.harness.tool_dispatch.sessions_service.append_event", append_event),
            patch(
                "aios.harness.tool_dispatch.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch("aios.harness.sweep.wake_sessions_needing_inference", wake_mock),
            pytest.raises(RuntimeError, match="db down"),
        ):
            await _trigger_sweep(MagicMock(), "sess_x", account_id="acc_test_stub")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[1]["repaired_ghosts"] == 0
        assert sweep_events[1]["woken_sessions"] == 0


# ─── entry site (loop._run_session_step_body guard) ──────────────────────────


def _acm_pool() -> MagicMock:
    """A pool mock whose ``acquire()`` is an async context manager (MagicMock
    auto-provides ``__aenter__``/``__aexit__`` as AsyncMocks), so the fast-path
    guard's ``async with pool.acquire() as conn`` works under mock — the guard
    never touches the yielded conn (``session_has_pending_work`` is patched)."""
    return MagicMock()


@pytest.fixture
def mock_runtime() -> Iterator[None]:
    with (
        patch("aios.harness.loop.runtime.require_pool", return_value=_acm_pool()),
        patch(
            "aios.harness.loop.runtime.require_inflight_tool_registry",
            return_value=MagicMock(),
        ),
    ):
        yield


class TestEntrySweepSpan:
    async def test_wasted_wake_marks_woken_sessions_zero(
        self, mock_runtime: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard early-out: the fast-path ``session_has_pending_work`` returns
        ``False`` (provably no work), so the full ``find_sessions_needing_inference``
        is never called and ``sweep_end`` stamps ``woken_sessions=0`` — the
        profiler's wasted-wake signal.

        Flag-on assertion (``AIOS_SWEEP_SPAN_DEBUG=1``): the entry-site sweep
        spans are gated off by default (#1749); ``test_sweep_spans_suppressed_when_flag_off``
        covers the (now-default) flag-off path on this exact wasted-wake route.
        """
        monkeypatch.setenv("AIOS_SWEEP_SPAN_DEBUG", "1")
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        full_sweep = AsyncMock(return_value=set())
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(return_value=False),
            ),
            patch("aios.harness.loop.find_sessions_needing_inference", full_sweep),
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
            "fast_path": True,
        }
        # (A): a provable "no work" from the fast path never touches the full sweep.
        full_sweep.assert_not_awaited()

    async def test_fast_path_pool_and_query_spans_emitted(
        self, mock_runtime: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(C) observability: the guard emits the ``sweep.pool_acquire`` and
        ``sweep.query_exec`` child span pairs (nested inside the outer
        ``sweep_start``/``sweep_end``) so the pool-wait vs event-loop-time-share
        split is *measured*, not inferred.

        Flag-on assertion (``AIOS_SWEEP_SPAN_DEBUG=1``) — these spans are
        gated off by default (#1749)."""
        monkeypatch.setenv("AIOS_SWEEP_SPAN_DEBUG", "1")
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(return_value=False),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
        ):
            await run_session_step("sess_x", cause="message")

        # Only the guard's own spans (``sweep_*`` outer pair + ``sweep.*`` child
        # pairs); other step spans (``step_start`` etc.) are filtered out.
        events = [
            e["event"] for e in _span_events(append_event) if str(e["event"]).startswith("sweep")
        ]
        # The child spans appear, bracketed by the outer sweep pair, in order.
        assert events == [
            "sweep_start",
            "sweep.pool_acquire_start",
            "sweep.pool_acquire_end",
            "sweep.query_exec_start",
            "sweep.query_exec_end",
            "sweep_end",
        ]

    async def test_happy_path_marks_woken_sessions_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard passes: session is in the needs set. ``woken_sessions=1``
        marks "keep going" for the profiler.

        Flag-on assertion (``AIOS_SWEEP_SPAN_DEBUG=1``) — these spans are
        gated off by default (#1749)."""
        monkeypatch.setenv("AIOS_SWEEP_SPAN_DEBUG", "1")
        from aios.harness.loop import run_session_step

        session = SimpleNamespace(
            id="sess_x",
            agent_id="agt_x",
            agent_version=None,
            focal_channel=None,
            origin="foreground",
            parent_run_id=None,
            archive_when_idle=False,
        )
        agent = SimpleNamespace(
            model="openrouter/x",
            tools=[],
            mcp_servers=[],
            http_servers=[],
            skills=[],
            system="sys",
            litellm_extra={},
            window_min=1000,
            window_max=10000,
            preempt_policy="wait",
        )
        start_event = SimpleNamespace(id="ev_sweep")

        with (
            patch("aios.harness.loop.runtime.require_pool", return_value=_acm_pool()),
            patch(
                "aios.harness.loop.runtime.require_inflight_tool_registry",
                return_value=MagicMock(),
            ),
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(return_value=True),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(return_value={"sess_x"}),
            ),
            patch(
                "aios.harness.loop.sessions_service.get_session_basic",
                AsyncMock(return_value=session),
            ),
            patch(
                "aios.harness.loop.agents_service.load_for_session",
                AsyncMock(return_value=agent),
            ),
            patch(
                "aios.services.channels.list_session_channels",
                AsyncMock(return_value=[]),
            ),
            patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
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
            patch("aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()),
            patch(
                "aios.harness.loop.sessions_service.append_event",
                AsyncMock(return_value=start_event),
            ) as append_event,
            patch(
                "aios.harness.loop.sessions_service.append_assistant_and_guard_quiescence",
                AsyncMock(return_value=AssistantAppendResult(False, None, None)),
            ),
            patch(
                "aios.harness.loop.call_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
            ),
            patch(
                "aios.harness.loop.stream_litellm",
                AsyncMock(return_value=_llm_response({"role": "assistant", "content": "ok"})),
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
            "fast_path": True,
        }

    async def test_sweep_end_fires_when_fast_path_raises(
        self, mock_runtime: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the fast-path ``session_has_pending_work`` raises, the guard's own
        try/finally still emits ``sweep_end`` (with ``woken_sessions=0``), and the
        exception propagates out (budget exhausted → re-raise).

        Flag-on assertion (``AIOS_SWEEP_SPAN_DEBUG=1``) — these spans are
        gated off by default (#1749); ``test_bare_guard_propagates_fast_path_error``
        covers the flag-off raise path.
        """
        monkeypatch.setenv("AIOS_SWEEP_SPAN_DEBUG", "1")
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            # Budget exhausted → retry_delay=None → re-raise (original test intent preserved).
            patch("aios.harness.loop._apply_retry_or_failure", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="db down"),
        ):
            await run_session_step("sess_x", cause="message")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        assert sweep_events[1]["woken_sessions"] == 0
        assert sweep_events[1]["repaired_ghosts"] == 0

    async def test_sweep_end_fires_when_full_sweep_raises(
        self, mock_runtime: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the fast path says "maybe work" (``True``) the guard falls through
        to the full ``find_sessions_needing_inference``. If THAT raises, the guard's
        try/finally still emits ``sweep_end`` and the exception propagates.

        Flag-on assertion (``AIOS_SWEEP_SPAN_DEBUG=1``) — these spans are
        gated off by default (#1749).
        """
        monkeypatch.setenv("AIOS_SWEEP_SPAN_DEBUG", "1")
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_sweep"))
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(return_value=True),
            ),
            patch(
                "aios.harness.loop.find_sessions_needing_inference",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch("aios.harness.loop._apply_retry_or_failure", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="db down"),
        ):
            await run_session_step("sess_x", cause="message")

        sweep_events = _sweep_events(append_event)
        assert [e["event"] for e in sweep_events] == ["sweep_start", "sweep_end"]
        # The guard early-marked woken_sessions from the fast path (True → 1) —
        # the sweep_end still closes even though the downstream full sweep raised.
        assert sweep_events[1]["woken_sessions"] == 1
        assert sweep_events[1]["repaired_ghosts"] == 0

    async def test_sweep_spans_suppressed_when_flag_off(self, mock_runtime: None) -> None:
        """#1749: with ``AIOS_SWEEP_SPAN_DEBUG`` unset (the default), none of the
        6 entry-site sweep spans are emitted on a wasted wake — only
        ``step_start``/``step_end`` (2 span appends total), and the fast-path
        early-out / full-sweep-skip behavior is unchanged."""
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_step"))
        full_sweep = AsyncMock(return_value=set())
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(return_value=False),
            ),
            patch("aios.harness.loop.find_sessions_needing_inference", full_sweep),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
        ):
            await run_session_step("sess_x", cause="message")

        span_names = [e["event"] for e in _span_events(append_event)]
        assert span_names == ["step_start", "step_end"]
        assert not _sweep_events(append_event)
        # Two span txns total: step_start, step_end.
        assert append_event.await_count == 2
        full_sweep.assert_not_awaited()

    async def test_bare_guard_propagates_fast_path_error(self, mock_runtime: None) -> None:
        """#1749: with the flag OFF, a ``pool.acquire()``/``session_has_pending_work``
        exception must still propagate to ``run_session_step``'s outer handler
        so ``_apply_retry_or_failure`` runs — the single-path ``_span`` helper
        only ever wraps span emission in ``finally``, never ``except``, so it
        cannot swallow the error even though it's a no-op flag-off."""
        from aios.harness.loop import run_session_step

        append_event = AsyncMock(return_value=SimpleNamespace(id="ev_step"))
        with (
            patch(
                "aios.harness.loop.session_has_pending_work",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch("aios.harness.loop.sessions_service.append_event", append_event),
            patch("aios.harness.loop._apply_retry_or_failure", AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="db down"),
        ):
            await run_session_step("sess_x", cause="message")

        span_names = [e["event"] for e in _span_events(append_event)]
        # step_start fired before the guard; the exception is caught by
        # run_session_step's outer ``except Exception`` (harness_error span +
        # _apply_retry_or_failure), then re-raised since the budget is
        # exhausted; step_end still fires from the outermost finally. No
        # sweep spans at all (flag off) — the guard's plain try/finally
        # never masks the RuntimeError with a stray except of its own.
        assert span_names == ["step_start", "harness_error", "step_end"]
        assert not _sweep_events(append_event)


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
