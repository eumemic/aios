from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from aios_matrix.appservice import create_appservice
from aios_matrix.config import MatrixConfig


@pytest.fixture
def config() -> MatrixConfig:
    return MatrixConfig(
        hs_url="http://synapse:8008",
        server_name="your.server",
        as_token="as-secret",
        hs_token="hs-secret",
        sender_localpart="_aios",
        user_namespace_regex=r"^@_aios_agent_[a-z0-9]+:your\.server$",
        listen_addr="127.0.0.1:29328",
        database_url="postgresql://aios:aios@postgres/matrix",
    )


@pytest.mark.asyncio
async def test_ping_and_transaction_require_hs_token(config: MatrixConfig) -> None:
    appservice = create_appservice(config, state_store=None)
    async with TestClient(TestServer(appservice.app)) as client:
        ping = await client.post(
            "/_matrix/app/v1/ping",
            json={"transaction_id": "ping-1"},
            params={"access_token": "hs-secret"},
        )
        assert ping.status == 200

        transaction = await client.put(
            "/_matrix/app/v1/transactions/txn-1",
            json={"events": []},
            headers={"Authorization": "Bearer hs-secret"},
        )
        assert transaction.status == 200

        rejected = await client.put(
            "/_matrix/app/v1/transactions/txn-2",
            json={"events": []},
            headers={"Authorization": "Bearer wrong"},
        )
        assert rejected.status == 401


@pytest.mark.asyncio
async def test_state_store_is_asyncpg_backed(
    config: MatrixConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = asyncio.Event()

    class FakeDatabase:
        async def start(self) -> None:
            opened.set()

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(
        "aios_matrix.appservice.Database.create", lambda *args, **kwargs: FakeDatabase()
    )
    appservice = create_appservice(config)
    await appservice.state_store.open()
    assert opened.is_set()
