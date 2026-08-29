"""E2E round-trip for the ssh_servers surface arm against a real Postgres.

Validates the hand-renumbered ``insert_agent`` / ``update_agent`` SQL (the
error-prone part of the surface change): an agent's ``ssh_servers`` persists
through create, the version snapshot, and an update that bumps the version.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.models.agents import SshServerSpec

_ACCOUNT = "acc_test_stub"
_HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyDataForTestsOnly"


def _spec(name: str = "prod", host: str = "web.example.com") -> SshServerSpec:
    return SshServerSpec(
        name=name,
        host=host,
        username="deploy",
        host_keys=[_HOST_KEY],
        credential="PROD_KEY",
    )


@pytest.mark.asyncio
async def test_ssh_servers_round_trip_create_update(pool: Any) -> None:
    from aios.services import agents as svc

    created = await svc.create_agent(
        pool,
        account_id=_ACCOUNT,
        name="ssh-persist-test",
        model="openrouter/x",
        system="sys",
        tools=[],
        ssh_servers=[_spec()],
        description=None,
        metadata={},
        window_min=1000,
        window_max=10000,
    )
    assert [s.name for s in created.ssh_servers] == ["prod"]
    assert created.ssh_servers[0].host == "web.example.com"

    # Read back through get (fresh hydrate, not the create return value).
    fetched = await svc.get_agent(pool, created.id, account_id=_ACCOUNT)
    assert fetched.ssh_servers == created.ssh_servers

    # An update that changes ssh_servers bumps the version and persists the new set.
    updated = await svc.update_agent(
        pool,
        created.id,
        account_id=_ACCOUNT,
        expected_version=created.version,
        ssh_servers=[_spec(name="db", host="db.example.com")],
    )
    assert updated.version == created.version + 1
    assert [s.name for s in updated.ssh_servers] == ["db"]

    refetched = await svc.get_agent(pool, created.id, account_id=_ACCOUNT)
    assert [s.name for s in refetched.ssh_servers] == ["db"]
