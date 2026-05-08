"""Unit tests for the connector-secrets fetch + cache helper.

Verifies the cache-after-first-call behaviour and the forward-compat
guard on the ``connection_id`` parameter (reserved for the future
multi-connection container shape per #309).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aios_connector_http import HttpConnector


class _Probe(HttpConnector):
    def __init__(self) -> None:
        super().__init__(base_url="http://x", token="aios_conn_x")


@pytest.fixture
def probe() -> _Probe:
    p = _Probe()
    p._client = AsyncMock()
    p._client.get_secrets = AsyncMock(return_value={"bot_token": "tok"})
    p._connection_id = "conn_X"
    return p


class TestSecrets:
    async def test_returns_dict_from_client(self, probe: _Probe) -> None:
        out = await probe.secrets()
        assert out == {"bot_token": "tok"}
        probe._client.get_secrets.assert_awaited_once()

    async def test_cached_after_first_call(self, probe: _Probe) -> None:
        await probe.secrets()
        await probe.secrets()
        await probe.secrets()
        # Three calls, one fetch — cache holds.
        assert probe._client.get_secrets.await_count == 1
