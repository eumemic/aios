"""E2E tests for the ``context_build_*`` span pair (issue #78, first stage)."""

from __future__ import annotations

from unittest import mock

import pytest

from tests.e2e.harness import Harness, assistant


class TestContextBuildSpan:
    async def test_context_build_span_pair_per_step(self, harness: Harness) -> None:
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "context_build_start"]
        ends = [s for s in spans if s.data["event"] == "context_build_end"]

        assert len(starts) == 1, starts
        assert len(ends) == 1, ends

        end = ends[0]
        assert end.data["context_build_start_id"] == starts[0].id
        assert end.data["is_error"] is False
        assert end.data["event_count_read"] >= 1
        assert end.data["message_count"] >= 1
        assert end.data["tools_count"] >= 0

    async def test_context_build_precedes_model_request(self, harness: Harness) -> None:
        """The context_build_end span must land before the model_request_start
        for the same step — the builder feeds the model call.
        """
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]

        cb_end_seq = next(s.seq for s in spans if s.data["event"] == "context_build_end")
        model_start_seq = next(s.seq for s in spans if s.data["event"] == "model_request_start")
        assert cb_end_seq < model_start_seq

    async def test_context_build_end_emitted_with_is_error_on_failure(
        self, harness: Harness
    ) -> None:
        """If the prologue raises, the end span still lands with ``is_error: True``
        (matches the ``model_request_*`` symmetry — operators shouldn't see
        orphan starts)."""
        harness.script_model([assistant("Hi!")])
        session = await harness.start("hello")

        # Poison ``build_messages`` on the specific step we drive next.
        with (
            mock.patch(
                "aios.harness.step_context.build_messages",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await harness.run_until_idle(session.id)

        events = await harness.all_events(session.id)
        spans = [e for e in events if e.kind == "span"]
        starts = [s for s in spans if s.data["event"] == "context_build_start"]
        ends = [s for s in spans if s.data["event"] == "context_build_end"]

        assert len(starts) == len(ends), (
            f"mismatched spans: {len(starts)} starts vs {len(ends)} ends"
        )
        error_ends = [e for e in ends if e.data.get("is_error") is True]
        assert error_ends, "expected at least one context_build_end with is_error"
