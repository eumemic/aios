"""Unit coverage for SSE response cleanup on un-invoked dispatch.

The four SSE route handlers in ``aios.api.routers`` open a dedicated
asyncpg connection via ``open_listen_for_*`` BEFORE constructing
``EventSourceResponse`` (issue #376 — preflight failures must surface
as 503 before 200 OK lands on the wire). The wrapped generator owns
``subscription.terminate()`` in ``finally`` for the normal path.

That ``finally`` does NOT run if the wrapped async generator never
starts. Python's async-generator semantics: ``aclose()`` and GC on an
unstarted generator skip the body entirely — no ``finally``, no
cleanup. The route handler creates the subscription before
constructing the response, so if the request task is cancelled in
FastAPI's dispatch gap (between ``response = await handler(request)``
and the subsequent ``await response(scope, receive, send)``), the
asyncpg connection, its ``LISTEN``, and the SSE subscriber advisory
lock leak until TCP keepalive reaps the backend (~2h).

``make_sse_response`` is the cleanup-aware response builder these
routes use; this test pins the contract that it terminates the
subscription on the un-invoked path.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import asyncpg
from sse_starlette import ServerSentEvent

from aios.api.sse import make_sse_response
from aios.db.listen import ListenSubscription


async def _unstarted_generator() -> AsyncIterator[ServerSentEvent]:
    """An async generator that the test never iterates.

    Mirrors the shape ``EventSourceResponse`` receives in production
    (the various ``sse_*_stream`` generators from ``aios.api.sse``).
    The ``finally`` clause is the production cleanup hook whose
    absence-of-execution this test is about: an unstarted generator
    skips it entirely.
    """
    try:
        yield ServerSentEvent(data="unreachable", event="event")
    finally:  # pragma: no cover — unreachable in the dispatch-gap path
        pass


def test_make_sse_response_terminates_subscription_when_response_uninvoked() -> None:
    """If the EventSourceResponse is dropped without ``__call__`` ever
    running (the FastAPI dispatch-gap cancellation case), the wrapped
    generator never starts and its ``finally:
    subscription.terminate()`` never runs. ``make_sse_response`` must
    register a cleanup hook that terminates the subscription's asyncpg
    connection when the response is GC'd on this path.
    """
    conn = MagicMock(spec=asyncpg.Connection)
    queue: asyncio.Queue[str] = asyncio.Queue()
    subscription = ListenSubscription(queue=queue, _conn=conn)

    # Build the response exactly as a route handler would, then drop it
    # without invoking ``__call__``. Refcount → 0 → response is GC'd →
    # the cleanup hook (if registered) fires.
    response = make_sse_response(subscription, _unstarted_generator())
    del response
    gc.collect()

    assert conn.terminate.called, (
        "asyncpg connection leaked: EventSourceResponse was constructed "
        "but never invoked, so the wrapped generator's finally never ran. "
        "make_sse_response must register a finalizer that terminates "
        "the subscription on the no-call path (otherwise the Postgres "
        "backend lingers ~2h until TCP keepalive reaps it)."
    )
