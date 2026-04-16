"""Unit-test fixtures.

Provides dummy environment variables so ``get_settings()`` succeeds without
a ``.env`` file or real credentials.  Session-scoped and autouse so every
unit test gets them automatically.
"""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest


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


def fake_pool_yielding_conn(conn: Any) -> Any:
    """Stand-in for ``asyncpg.Pool`` whose ``async with pool.acquire()`` yields *conn*."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool
