"""``await_completion`` — block until a monotonic done-predicate holds, or a timeout.

The one mechanism behind "await-a-completion": block on a target's notify queue, re-reading
state on each notify until ``is_done`` holds (or the deadline passes). The predicate must be
*monotonic* — a terminal record never un-completes — so the loop safely stops the first time it
holds. The unified awaiter :func:`aios.services.invocations.await_invocation` drives it for both
a run (``status in {completed, errored, cancelled}``) and a session (``request_response``
correlation); :func:`aios.services.sessions.await_session` drives it for watermarked quiescence.
None is the non-monotonic "wait for idle", which would be the reverted Stop hook.

The caller owns the subscription lifecycle; this helper only consumes the ``queue``, staying a
pure function of ``(queue, read_state, is_done)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def await_completion[S](
    queue: asyncio.Queue[Any],
    *,
    read_state: Callable[[], Awaitable[S]],
    is_done: Callable[[S], bool],
    timeout_seconds: float,
) -> S:
    """Block until ``is_done(state)`` or ``timeout_seconds`` elapse; return the last state read.

    ``read_state`` is called **after** the caller has subscribed (the LISTEN-before-read
    invariant: a state change between subscribe and the first read is already queued, so it
    can't be missed), then again on every notify. Any queue item triggers a re-read — for runs
    each notify is a committed event; a backing whose channel also carries non-advancing pokes
    just re-reads harmlessly. Returns the terminal state when done, or the current (non-terminal)
    state on timeout — the caller's signal to re-poll.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    state = await read_state()
    while not is_done(state):
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(queue.get(), timeout=remaining)
        except TimeoutError:
            break
        state = await read_state()
    return state
