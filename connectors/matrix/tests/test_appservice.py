from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator

import pytest
from aiohttp.test_utils import TestClient, TestServer
from mautrix.types import Membership, RoomID, UserID

from aios_matrix.appservice import PostgresStateStore, create_appservice
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
    create_kwargs: dict[str, object] = {}

    class FakeDatabase:
        async def start(self) -> None:
            opened.set()

        async def stop(self) -> None:
            pass

    def create_database(*args: object, **kwargs: object) -> FakeDatabase:
        create_kwargs.update(kwargs)
        return FakeDatabase()

    monkeypatch.setattr("aios_matrix.appservice.Database.create", create_database)
    appservice = create_appservice(config)
    await appservice.state_store.open()
    assert opened.is_set()
    assert create_kwargs["upgrade_table"] is PostgresStateStore.upgrade_table


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available")
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, check=False, timeout=10
    )
    if probe.returncode != 0:
        pytest.skip("Docker daemon is not available")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    name = f"aios-matrix-test-{port}"
    run = subprocess.run(
        [
            "docker", "run", "--detach", "--rm", "--name", name,
            "--publish", f"127.0.0.1:{port}:5432",
            "--env", "POSTGRES_PASSWORD=aios",
            "--env", "POSTGRES_USER=aios",
            "--env", "POSTGRES_DB=matrix",
            "postgres:16-alpine",
        ],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start Postgres container: {run.stderr}")
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "aios"],
                capture_output=True, check=False, timeout=5,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("Postgres container did not become ready")
        yield f"postgresql://aios:aios@127.0.0.1:{port}/matrix"
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, check=False, timeout=10
        )


@pytest.mark.asyncio
async def test_fresh_postgres_store_persists_membership(
    config: MatrixConfig, postgres_url: str
) -> None:
    config = config.model_copy(update={"database_url": postgres_url})
    room_id = RoomID("!room:your.server")
    user_id = UserID("@_aios_agent_one:your.server")

    first = create_appservice(config).state_store
    await first.open()
    try:
        await first.set_membership(room_id, user_id, Membership.JOIN)
    finally:
        await first.close()

    reopened = create_appservice(config).state_store
    await reopened.open()
    try:
        assert await reopened.get_members(room_id) == [user_id]
    finally:
        await reopened.close()
