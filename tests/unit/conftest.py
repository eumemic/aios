"""Unit-test fixtures.

Provides dummy environment variables so ``get_settings()`` succeeds without
a ``.env`` file or real credentials.  Session-scoped and autouse so every
unit test gets them automatically.
"""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector


@pytest.fixture(autouse=True, scope="session")
def _unit_env() -> Iterator[None]:
    """Inject minimal env vars required by ``Settings()``."""
    env = {
        "AIOS_API_KEY": secrets.token_urlsafe(16),
        "AIOS_VAULT_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
        "AIOS_DB_URL": "postgresql://test:test@localhost:5432/test",
    }
    with mock.patch.dict(os.environ, env):
        from aios.config import get_settings

        get_settings.cache_clear()
        yield
        get_settings.cache_clear()


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    """Patch the aios procrastinate app to use an in-memory connector.

    Tests can read ``app.connector.jobs`` directly to inspect deferred
    job rows (lock, queueing_lock, schedule, args).
    """
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


def fake_pool_yielding_conn(conn: Any) -> Any:
    """Stand-in for ``asyncpg.Pool`` whose ``async with pool.acquire()`` yields *conn*."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool


# ── image magic-byte fixtures (shared across vision + context tests) ──


def png_b64(payload: bytes = b"\x00" * 32) -> str:
    """Base64-encoded PNG with valid magic header followed by *payload*."""
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + payload).decode()


def jpeg_b64(payload: bytes = b"\x00" * 24) -> str:
    """Base64-encoded JPEG with a valid SOI + JFIF prefix followed by *payload*."""
    return base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF" + payload).decode()


def gif_b64(payload: bytes = b"\x00" * 32, *, version: bytes = b"89a") -> str:
    """Base64-encoded GIF with a valid magic header followed by *payload*."""
    assert version in (b"87a", b"89a")
    return base64.b64encode(b"GIF" + version + payload).decode()
