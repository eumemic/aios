"""``/v1/triggers`` over HTTP (#2473): one round trip through the operator routes."""

from __future__ import annotations

import json
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


@pytest.mark.parametrize("literal", ["1e400", "Infinity", "NaN"])
async def test_a_non_finite_budget_is_a_422_and_writes_nothing(
    pool: Any, http_client: httpx.AsyncClient, literal: str
) -> None:
    """``1e400`` parses to ``inf`` (and Python's JSON reader accepts ``Infinity`` /
    ``NaN``). ``gt=0`` admits ``inf``, and the JSONB insert then failed with a 500
    (#2525). Both the create and the replace body refuse it with a 422, and no row
    is written or changed."""
    env = await environments_service.create_environment(
        pool, name=f"op-inf-{secrets.token_hex(4)}", account_id="acc_test_stub"
    )
    wf = await http_client.post(
        "/v1/workflows",
        json={
            "name": f"op-inf-{secrets.token_hex(4)}",
            "script": "async def main(input):\n    return input\n",
        },
    )
    assert wf.status_code == 201, wf.text
    wf_id = wf.json()["id"]
    name = f"inf-{secrets.token_hex(3)}"
    create_body = {
        "name": name,
        "source": {"kind": "cron", "schedule": "0 9 * * 1"},
        "action": {"kind": "workflow", "workflow_id": wf_id, "budget_usd": "BUDGET"},
        "environment_id": env.id,
    }
    raw_create = json.dumps(create_body).replace('"BUDGET"', literal)
    headers = {"content-type": "application/json"}

    created = await http_client.post("/v1/triggers", content=raw_create, headers=headers)
    assert created.status_code == 422, created.text
    assert (await http_client.get(f"/v1/triggers/{name}")).status_code == 404
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM triggers WHERE name = $1", name) == 0

    ok = await http_client.post(
        "/v1/triggers",
        json={
            "name": name,
            "source": {"kind": "cron", "schedule": "0 9 * * 1"},
            "action": {"kind": "workflow", "workflow_id": wf_id, "budget_usd": 3},
            "environment_id": env.id,
        },
    )
    assert ok.status_code == 201, ok.text
    replace_body = {
        "action": {
            "kind": "workflow",
            "workflow_id": wf_id,
            "workflow_version": None,
            "version": None,
            "input_template": None,
            "vault_ids": [],
            "max_outstanding_runs": None,
            "budget_usd": "BUDGET",
        }
    }
    raw_replace = json.dumps(replace_body).replace('"BUDGET"', literal)
    replaced = await http_client.put(f"/v1/triggers/{name}", content=raw_replace, headers=headers)
    assert replaced.status_code == 422, replaced.text
    after = await http_client.get(f"/v1/triggers/{name}")
    assert after.json()["action"]["budget_usd"] == 3
