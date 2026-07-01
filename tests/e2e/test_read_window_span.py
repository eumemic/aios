"""E2E tests for the pre-inference read-window span pairs (issue #1658).

The ~4s pre-inference read cost used to run in an *unspanned* window between
``step_start`` and ``context_build_start`` — invisible to perf profiling. This
adds two span pairs around ``compute_step_prelude`` and ``read_windowed_events``
so the cost is a query, not a manual full-window decomposition.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import Harness, assistant


class TestReadWindowSpan:
    async def test_prelude_span_pair_per_step(self, harness: Harness) -> None:
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "compute_prelude_start"]
        ends = [s for s in spans if s.data["event"] == "compute_prelude_end"]

        assert len(starts) == 1, starts
        assert len(ends) == 1, ends
        assert ends[0].data["compute_prelude_start_id"] == starts[0].id
        assert ends[0].data["is_error"] is False

    async def test_read_window_span_pair_per_step(self, harness: Harness) -> None:
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "read_window_start"]
        ends = [s for s in spans if s.data["event"] == "read_window_end"]

        assert len(starts) == 1, starts
        assert len(ends) == 1, ends
        assert ends[0].data["read_window_start_id"] == starts[0].id
        assert ends[0].data["is_error"] is False
        assert ends[0].data["event_count_read"] >= 1

    async def test_read_window_spans_ordered_within_step(self, harness: Harness) -> None:
        """The prelude and read-window spans land inside the step_start /
        context_build_start window, in that order — they measure the previously
        unspanned pre-inference read cost."""
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]

        def seq(name: str) -> int:
            return next(s.seq for s in spans if s.data["event"] == name)

        step_start = seq("step_start")
        prelude_start = seq("compute_prelude_start")
        prelude_end = seq("compute_prelude_end")
        read_start = seq("read_window_start")
        read_end = seq("read_window_end")
        cb_start = seq("context_build_start")

        assert step_start < prelude_start < prelude_end < read_start < read_end < cb_start

    async def test_prelude_span_end_emitted_with_is_error_on_failure(
        self, harness: Harness
    ) -> None:
        """If ``compute_step_prelude`` raises, the end span still lands with
        ``is_error: True`` (no orphan starts)."""
        from unittest import mock

        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")

        async def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom-prelude")

        async def _noop_defer_wake(*args: object, **kwargs: object) -> None:
            return None

        with (
            mock.patch("aios.harness.loop.compute_step_prelude", side_effect=_boom),
            mock.patch("aios.harness.loop.defer_wake", _noop_defer_wake),
            pytest.raises(RuntimeError, match="boom-prelude"),
        ):
            await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "compute_prelude_start"]
        ends = [s for s in spans if s.data["event"] == "compute_prelude_end"]

        assert len(starts) == len(ends), (
            f"mismatched spans: {len(starts)} starts vs {len(ends)} ends"
        )
        assert any(e.data.get("is_error") is True for e in ends)

    async def test_read_window_span_end_emitted_with_is_error_on_failure(
        self, harness: Harness
    ) -> None:
        """If ``read_windowed_events`` raises, the end span still lands with
        ``is_error: True`` (no orphan starts)."""
        from unittest import mock

        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")

        async def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("boom-read")

        async def _noop_defer_wake(*args: object, **kwargs: object) -> None:
            return None

        with (
            mock.patch(
                "aios.harness.loop.sessions_service.read_windowed_events",
                side_effect=_boom,
            ),
            mock.patch("aios.harness.loop.defer_wake", _noop_defer_wake),
            pytest.raises(RuntimeError, match="boom-read"),
        ):
            await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "read_window_start"]
        ends = [s for s in spans if s.data["event"] == "read_window_end"]

        assert len(starts) == len(ends), (
            f"mismatched spans: {len(starts)} starts vs {len(ends)} ends"
        )
        assert any(e.data.get("is_error") is True for e in ends)
