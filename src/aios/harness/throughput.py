"""In-process wake-completion counter feeding the throughput dead-man.

The dead-man (:mod:`aios.harness.production_watchdogs`) used to count
``procrastinate_events`` rows of type ``succeeded``. The worker runs with
``delete_jobs="successful"`` (#888, 2026-06-10), and procrastinate's
``procrastinate_finish_job_v1`` *deletes* a successful job instead of updating
its status — so no ``succeeded`` event is ever written, and the job's earlier
events cascade away with it. The detector read a structural zero and could not
tell a stalled worker from a busy one (#2541).

Completion is therefore measured from something the worker itself does: the
``wake_session`` / ``wake_workflow`` task bodies run inside
:meth:`CompletionCounter.track`, and the watchdog — running in the same
process — samples the counter's delta each tick. Only one worker process runs
per database (the startup advisory lock), so a process-local counter sees every
wake completion that ``procrastinate_jobs``' ``doing`` rows can claim.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class CompletionCounter:
    """Monotonic counts of finished wake jobs in this process.

    ``total`` counts only wakes that *returned normally* — that is the progress
    signal the throughput dead-man consumes. Wakes that raised are counted
    separately in ``failed`` and are deliberately NOT progress: a stream of
    failing wakes must not reset the stall timer while another claimed wake is
    stuck (the replaced ``procrastinate_events`` signal counted only
    ``succeeded``, too).
    """

    def __init__(self) -> None:
        self._total = 0
        self._failed = 0

    @property
    def total(self) -> int:
        """Wakes that returned normally (the dead-man's progress signal)."""
        return self._total

    @property
    def failed(self) -> int:
        """Wakes that raised an ``Exception`` — visible, but never progress."""
        return self._failed

    def record(self) -> None:
        self._total += 1

    def record_failure(self) -> None:
        self._failed += 1

    @contextmanager
    def track(self) -> Iterator[None]:
        """Count the wrapped job by outcome.

        A normal return increments ``total``. An ``Exception`` increments
        ``failed`` only — an error is not throughput. Cancellation
        (``BaseException``) counts as neither: that is shutdown or abort.
        """
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        self.record()


WAKE_COMPLETIONS = CompletionCounter()
