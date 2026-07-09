"""Unit tests for the bash memory-reconcile telemetry surface (#1748).

Corroboration-only: verifies the counters/timers record what they claim to,
never that they gate anything (they must not).
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.tools import bash_memory_telemetry as telemetry


@pytest.fixture(autouse=True)
def _reset() -> Any:
    telemetry.reset()
    yield
    telemetry.reset()


class TestCandidateReadCounter:
    def test_zero_reads_reported_for_all_unchanged(self) -> None:
        telemetry.record_candidate_reads(0)
        assert telemetry.last_candidate_read_count() == 0
        assert telemetry.total_candidate_reads() == 0

    def test_records_nonzero_count(self) -> None:
        telemetry.record_candidate_reads(3)
        assert telemetry.last_candidate_read_count() == 3
        assert telemetry.total_candidate_reads() == 3

    def test_accumulates_across_calls(self) -> None:
        telemetry.record_candidate_reads(2)
        telemetry.record_candidate_reads(5)
        # last reflects only the most recent call...
        assert telemetry.last_candidate_read_count() == 5
        # ...but the cumulative total across the process lifetime sums both.
        assert telemetry.total_candidate_reads() == 7

    def test_never_raises_even_if_prometheus_sink_errors(self) -> None:
        """A metrics-sink failure must never propagate — corroboration only."""
        import aios.tools.bash_memory_telemetry as mod

        class _BoomCounter:
            def inc(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("sink exploded")

        prev = mod._CANDIDATE_READS
        mod._CANDIDATE_READS = _BoomCounter()
        try:
            telemetry.record_candidate_reads(1)  # must not raise
        finally:
            mod._CANDIDATE_READS = prev
        assert telemetry.last_candidate_read_count() == 1


class TestPhaseDurations:
    def test_timed_phase_records_a_sample(self) -> None:
        with telemetry.timed_phase("after_scan"):
            pass
        assert telemetry.phase_counts().get("after_scan") == 1
        assert telemetry.phase_totals().get("after_scan", 0.0) >= 0.0

    def test_timed_phase_reraises_body_exception(self) -> None:
        with pytest.raises(ValueError), telemetry.timed_phase("db_phase"):
            raise ValueError("boom")
        # Still recorded a sample despite the exception.
        assert telemetry.phase_counts().get("db_phase") == 1

    def test_nested_phases_both_recorded(self) -> None:
        with telemetry.timed_phase("memory_reconcile"), telemetry.timed_phase("after_scan"):
            pass
        assert telemetry.phase_counts().get("memory_reconcile") == 1
        assert telemetry.phase_counts().get("after_scan") == 1


class TestReset:
    def test_reset_clears_everything(self) -> None:
        telemetry.record_candidate_reads(4)
        with telemetry.timed_phase("after_scan"):
            pass
        telemetry.reset()
        assert telemetry.total_candidate_reads() == 0
        assert telemetry.last_candidate_read_count() == 0
        assert telemetry.phase_counts() == {}
        assert telemetry.phase_totals() == {}
