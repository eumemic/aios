"""Health and readiness endpoints. Unauthenticated."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from aios import __version__
from aios.db.queries.events import calibration_telemetry

router = APIRouter()


@router.get("/health", operation_id="get_health")
async def health() -> dict[str, str]:
    """Liveness probe. Unauthenticated; returns the running aios version.

    Suitable for load balancer health checks and monitoring probes. Always
    returns 200 with ``{"status": "ok", "version": <version>,
    "source_commit": <sha>}`` if the process is up. Deliberately does NOT
    touch the DB pool — a post-startup Postgres outage must not flip liveness
    (that's ``/ready``'s job), or an orchestrator would kill an otherwise-
    healthy process during a DB blip.

    ``source_commit`` echoes the container's build-time ``SOURCE_COMMIT`` env
    (baked into the image at ``docker build`` from Coolify's commit build-arg).
    It makes the *truthful running SHA* verifiable over pure HTTPS: ``fleet
    --drift`` only sees Coolify's deploy-queue commit (build INTENT), which can
    silently diverge from what a zombie/failed-deploy container is actually
    running — and reading the real SHA otherwise needs SSH, which is often
    blocked. Surfacing it here lets seat + ops audits confirm a promote landed
    and that api/worker are in lockstep over plain HTTP (issue #1669). Falls
    back to ``"unknown"`` outside a built container (local dev), so the key is
    always present for automated audits to key off of.
    """
    return {
        "status": "ok",
        "version": __version__,
        "source_commit": os.environ.get("SOURCE_COMMIT") or "unknown",
    }


@router.get("/telemetry/calibration", operation_id="get_calibration_telemetry")
async def calibration(request: Request) -> dict[str, object]:
    """On-demand calibration observability for models active in the last 24h."""
    async with request.app.state.pool.acquire() as conn:
        models = await calibration_telemetry(conn)
    return {"models": models, "window_hours": 24}


@router.get("/ready", operation_id="get_ready")
async def ready(request: Request) -> Response:
    """Readiness probe. Unauthenticated; ``SELECT 1`` under a short timeout.

    Returns 200 ``{"status": "ready"}`` when Postgres answers AND the
    fail-closed boot-admission gate (#1575) has admitted this process; 503
    ``{"status": "unavailable"}`` when the pool can't be acquired, the query
    raises, it exceeds the 2 s budget, OR the boot-gate has not yet admitted.
    This is the signal the Docker/compose healthcheck watches, so a silent
    post-startup DB outage becomes a visibly unhealthy container instead of a
    200-returning black hole.

    The boot-gate flag (``app.state.retirements_ok``) stays ``False`` from
    process start until the gate proves the live DB is safe for this code
    (alembic at/past every ``contract_rev`` AND zero live residue). Under a
    rolling deploy this keeps the NEW container unready — so the OLD healthy
    container keeps serving — until the post-deploy migrate lands. Checked
    BEFORE the DB round-trip: an un-admitted process is unready regardless of
    DB liveness.
    """
    if not getattr(request.app.state, "retirements_ok", False):
        return JSONResponse(status_code=503, content={"status": "unavailable"})
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
