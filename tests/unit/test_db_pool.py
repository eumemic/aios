"""Unit tests for aios.db.pool."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from aios.db.pool import create_pool, normalize_dsn


def _make_fake_pool(pg_max_connections: str) -> MagicMock:
    """Build a fake asyncpg pool whose acquire() yields a conn with fetchval."""
    fake_conn = AsyncMock()
    fake_conn.fetchval = AsyncMock(return_value=pg_max_connections)

    fake_pool = MagicMock()

    @asynccontextmanager
    async def _acquire() -> Any:
        yield fake_conn

    fake_pool.acquire = _acquire
    return fake_pool


@pytest.fixture
def capture_logs() -> Iterator[structlog.testing.LogCapture]:
    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        yield cap
    finally:
        structlog.reset_defaults()


@pytest.mark.asyncio
async def test_default_max_size_is_8() -> None:
    fake_pool = _make_fake_pool("200")
    with patch(
        "aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
    ) as mock_create:
        await create_pool("postgresql://stub/db")
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["max_size"] == 8


@pytest.mark.asyncio
async def test_pool_sets_timeouts_and_keepalive() -> None:
    fake_pool = _make_fake_pool("200")
    with patch(
        "aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
    ) as mock_create:
        await create_pool("postgresql://stub/db")
        _, kwargs = mock_create.call_args
        ss = kwargs["server_settings"]
        assert ss["statement_timeout"] == "30000"
        assert ss["idle_in_transaction_session_timeout"] == "60000"
        assert ss["tcp_keepalives_idle"] == "60"
        assert ss["tcp_keepalives_interval"] == "10"
        assert ss["tcp_keepalives_count"] == "5"


@pytest.mark.asyncio
async def test_warning_emitted_when_capacity_reaches_pg_max(
    capture_logs: structlog.testing.LogCapture,
) -> None:
    # 8 (max_size) * 2 (known pool count) == 16 >= 16 (pg_max_connections) → warning
    fake_pool = _make_fake_pool("16")
    with patch("aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)):
        await create_pool("postgresql://stub/db")

    matches = [e for e in capture_logs.entries if e.get("event") == "db.pool.unsafe_max_size"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["log_level"] == "warning"
    assert entry["total_pool_capacity"] == 16
    assert entry["pg_max_connections"] == 16


@pytest.mark.asyncio
async def test_no_warning_when_capacity_is_safe(
    capture_logs: structlog.testing.LogCapture,
) -> None:
    fake_pool = _make_fake_pool("100")
    with patch("aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)):
        await create_pool("postgresql://stub/db")

    matches = [e for e in capture_logs.entries if e.get("event") == "db.pool.unsafe_max_size"]
    assert len(matches) == 0


def test_normalize_dsn_strips_asyncpg_prefix() -> None:
    assert normalize_dsn("postgresql+asyncpg://u:p@host/db") == "postgresql://u:p@host/db"


def test_normalize_dsn_strips_psycopg_prefix() -> None:
    assert normalize_dsn("postgresql+psycopg://u:p@host/db") == "postgresql://u:p@host/db"


@pytest.mark.asyncio
async def test_raises_when_asyncpg_returns_none() -> None:
    with (
        patch("aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=None)),
        pytest.raises(RuntimeError),
    ):
        await create_pool("postgresql://stub/db")
