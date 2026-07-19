"""Unit tests for aios.db.pool."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from aios.db.pool import (
    _jsonb_encoder,
    create_pool,
    normalize_dsn,
    register_jsonb_codec,
)


def _make_fake_pool(pg_max_connections: str, listener_count: int = 0) -> MagicMock:
    """Build a fake asyncpg pool whose acquire() yields a conn with fetchval."""
    fake_conn = AsyncMock()
    fake_conn.fetchval = AsyncMock(side_effect=[pg_max_connections, listener_count])

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


@pytest.mark.asyncio
async def test_pool_registers_jsonb_codec_init() -> None:
    """create_pool wires register_jsonb_codec as the pool init callback."""
    fake_pool = _make_fake_pool("200")
    with patch(
        "aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
    ) as mock_create:
        await create_pool("postgresql://stub/db")
        _, kwargs = mock_create.call_args
        assert kwargs["init"] is register_jsonb_codec


@pytest.mark.asyncio
async def test_register_jsonb_codec_wires_pg_catalog_codec() -> None:
    """The init callback registers the pg_catalog jsonb codec with our en/decoders."""
    conn = AsyncMock()
    await register_jsonb_codec(conn)
    conn.set_type_codec.assert_awaited_once_with(
        "jsonb",
        encoder=_jsonb_encoder,
        decoder=json.loads,
        schema="pg_catalog",
    )


def test_jsonb_encoder_serializes_bare_python() -> None:
    """Bare dicts/lists (the Stage 2 write shape) are json.dumps'd."""
    assert json.loads(_jsonb_encoder({"a": 1})) == {"a": 1}
    assert json.loads(_jsonb_encoder([1, 2, 3])) == [1, 2, 3]


def test_jsonb_encoder_passes_through_preserialized_string() -> None:
    """A pre-serialized JSON string passes through untouched — NOT re-dumped.

    This is the additive-Stage-1 guarantee: existing writes that already
    json.dumps(...) their value must not be double-encoded by the codec.
    """
    pre = json.dumps({"a": 1})
    assert _jsonb_encoder(pre) == pre  # no extra layer of quoting
    # A plain json.dumps encoder would have produced a double-encoded string.
    assert _jsonb_encoder(pre) != json.dumps(pre)


@pytest.mark.asyncio
async def test_warning_includes_live_listener_backends(
    capture_logs: structlog.testing.LogCapture,
) -> None:
    fake_pool = _make_fake_pool("20", listener_count=5)
    with patch("aios.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)):
        await create_pool("postgresql://stub/db")
    entry = next(e for e in capture_logs.entries if e.get("event") == "db.pool.unsafe_max_size")
    assert entry["live_listener_count"] == 5
    assert entry["total_connection_budget"] == 21
