"""Health and readiness endpoints. Unauthenticated."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from aios import __version__

router = APIRouter()


@router.get("/health", operation_id="get_health")
async def health() -> dict[str, str | None]:
    """Liveness probe. Unauthenticated; returns the running aios version + build SHA.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{"status": "ok", "version": <version>, "build_sha": <sha-or-None>}``
    if the process is up. Deliberately does NOT touch the DB pool — a post-startup
    Postgres outage must not flip liveness (that's ``/ready``'s job), or an
    orchestrator would kill an otherwise-healthy process during a DB blip.

    ``build_sha`` (aios#1327, Unit A2) is the git commit baked into THIS image at
    build time (``ENV AIOS_BUILD_SHA`` in the Dockerfile ``base`` stage). It is the
    in-process running-truth read: unlike ``version`` (a static package string that
    never changes between builds), it tells you which commit is actually running.
    ``None`` when ``AIOS_BUILD_SHA`` is unset — an un-instrumented build — which the
    running==merged reconciler reads as ``cannot-determine``, never a false match.
    ``version`` stays the static package string, untouched.

    The return type is ``dict[str, str | None]`` (not ``dict[str, str]``): ``build_sha``
    is genuinely nullable because ``os.environ.get`` returns ``str | None``.
    """
    return {
        "status": "ok",
        "version": __version__,
        "build_sha": os.environ.get("AIOS_BUILD_SHA"),
    }


@router.get("/ready", operation_id="get_ready")
async def ready(request: Request) -> Response:
    """Readiness probe. Unauthenticated; ``SELECT 1`` under a short timeout.

    Returns 200 ``{"status": "ready"}`` when Postgres answers, 503
    ``{"status": "unavailable"}`` when the pool can't be acquired, the query
    raises, or it exceeds the 2 s budget. This is the signal the Docker/compose
    healthcheck watches, so a silent post-startup DB outage becomes a visibly
    unhealthy container instead of a 200-returning black hole.
    """
    pool = request.app.state.pool
    try:
        # The timeout intentionally covers both ``pool.acquire()`` and the query: a
        # pool that can't hand out a connection (exhausted/stuck) is genuinely "not
        # ready" to serve DB-backed traffic, same as a DB that won't answer — both are
        # a correct 503. We don't distinguish the two; the probe's job is liveness of
        # the DB path, not root-cause attribution.
        async with asyncio.timeout(2.0):
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(status_code=200, content={"status": "ready"})
