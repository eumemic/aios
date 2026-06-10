"""Unit tests for the ``/ready`` readiness probe and ``/health`` liveness purity.

``/ready`` runs a ``SELECT 1`` against the pool under a short timeout: 200 when
the DB answers, 503 when ``acquire``/``fetchval`` raises or the query times out.
``/health`` stays pure liveness — it must NEVER touch the pool — so a Postgres
outage is loud on ``/ready`` while ``/health`` still reports the process is up.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.routers import health


class _FakeConn:
    def __init__(self, fetchval_impl: Any) -> None:
        self._fetchval_impl = fetchval_impl

    async def fetchval(self, query: str) -> Any:
        return await self._fetchval_impl(query)


class _FakeAcquireCM:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    """Minimal stand-in for ``asyncpg.Pool``.

    ``acquire()`` yields a conn whose ``fetchval`` runs *fetchval_impl*; if
    *acquire_raises* is set, ``acquire()`` itself raises (models a dead pool).
    """

    def __init__(
        self, fetchval_impl: Any = None, *, acquire_raises: BaseException | None = None
    ) -> None:
        self._fetchval_impl = fetchval_impl
        self._acquire_raises = acquire_raises

    def acquire(self) -> _FakeAcquireCM:
        if self._acquire_raises is not None:
            raise self._acquire_raises
        return _FakeAcquireCM(_FakeConn(self._fetchval_impl))


def _app_with_pool(pool: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    app.state.pool = pool
    return app


def test_ready_200_when_select_ok() -> None:
    async def _ok(_query: str) -> int:
        return 1

    client = TestClient(_app_with_pool(_FakePool(_ok)))
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_ready_503_when_fetchval_raises() -> None:
    async def _boom(_query: str) -> int:
        raise RuntimeError("db gone")

    client = TestClient(_app_with_pool(_FakePool(_boom)))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable"}


def test_ready_503_when_acquire_raises() -> None:
    client = TestClient(_app_with_pool(_FakePool(acquire_raises=RuntimeError("pool closed"))))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable"}


def test_ready_503_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow ``SELECT 1`` trips the timeout → 503. We shrink the timeout to a
    tiny value so the test stays fast and deterministic."""

    async def _slow(_query: str) -> int:
        await asyncio.sleep(1.0)
        return 1

    real_timeout = asyncio.timeout

    def _tiny_timeout(_delay: float) -> Any:
        return real_timeout(0.01)

    monkeypatch.setattr("aios.api.routers.health.asyncio.timeout", _tiny_timeout)
    client = TestClient(_app_with_pool(_FakePool(_slow)))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable"}


def test_health_still_200_and_pool_untouched() -> None:
    """``/health`` must not touch the pool: a pool that raises on ``acquire``
    still yields a 200 from ``/health``."""

    class _ExplodingPool:
        def acquire(self) -> Any:
            raise AssertionError("/health must not touch the pool")

    client = TestClient(_app_with_pool(_ExplodingPool()))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
