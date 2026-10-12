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
    """Monotonic count of finished wake jobs in this process."""

    def __init__(self) -> None:
        self._total = 0

    @property
    def total(self) -> int:
        return self._total

    def record(self) -> None:
        self._total += 1

    @contextmanager
    def track(self) -> Iterator[None]:
        """Count the wrapped job when it returns or raises.

        A step that raised still finished: the worker made progress and moved
        on. Cancellation (``BaseException``) is not counted — that is shutdown
        or abort, not throughput.
        """
        try:
            yield
        except Exception:
            self.record()
            raise
        self.record()


WAKE_COMPLETIONS = CompletionCounter()
