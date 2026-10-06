"""``/v1/triggers`` over HTTP (#2473): one round trip through the operator routes."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aios.services import environments as environments_service
from tests.helpers.connections import authed_client, wired_app


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver", aios_env["AIOS_API_KEY"], transport=transport
    ) as client:
        yield client


async def test_create_read_update_list_runs_and_delete(
    pool: Any, http_client: httpx.AsyncClient
) -> None:
    env = await environments_service.create_environment(
        pool, name=f"op-trig-{secrets.token_hex(4)}", account_id="acc_test_stub"
    )
    wf = await http_client.post(
        "/v1/workflows",
        json={
            "name": f"op-trig-{secrets.token_hex(4)}",
            "script": "async def main(input):\n    return input\n",
        },
    )
    assert wf.status_code == 201, wf.text
    name = f"weekly-{secrets.token_hex(3)}"
    action: dict[str, Any] = {
        "kind": "workflow",
        "workflow_id": wf.json()["id"],
        "budget_usd": 3,
    }
    body: dict[str, Any] = {
        "name": name,
        "source": {"kind": "cron", "schedule": "0 9 * * 1"},
        "action": action,
        "environment_id": env.id,
    }

    created = await http_client.post("/v1/triggers", json=body)
    assert created.status_code == 201, created.text
    assert created.json()["environment_id"] == env.id
    assert created.json()["next_fire"] is not None

    duplicate = await http_client.post("/v1/triggers", json=body)
    assert duplicate.status_code == 409, duplicate.text
    event_source = await http_client.post(
        "/v1/triggers", json={**body, "name": "hook", "source": {"kind": "external_event"}}
    )
    assert event_source.status_code == 422, event_source.text
    no_budget = await http_client.post(
        "/v1/triggers",
        json={**body, "name": "nobudget", "action": {**action, "budget_usd": None}},
    )
    assert no_budget.status_code == 422, no_budget.text

    listed = await http_client.get("/v1/triggers")
    assert name in [t["name"] for t in listed.json()["data"]]
    updated = await http_client.put(f"/v1/triggers/{name}", json={"enabled": False})
    assert updated.status_code == 200, updated.text
    assert updated.json()["enabled"] is False and updated.json()["next_fire"] is None
    runs = await http_client.get(f"/v1/triggers/{name}/runs")
    assert runs.status_code == 200 and runs.json()["data"] == []

    deleted = await http_client.delete(f"/v1/triggers/{name}")
    assert deleted.status_code == 204, deleted.text
    gone = await http_client.get(f"/v1/triggers/{name}")
    assert gone.status_code == 404
