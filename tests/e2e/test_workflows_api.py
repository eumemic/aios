"""E2E tests for the ``/v1/workflows`` + ``/v1/runs`` HTTP endpoints (Block 3).

Covers the router contract over real routes: auth, status codes, create→get→list,
the run journal, and the 404 paths. The full gate-resume round-trip (which needs the
worker to drive a step) is covered by the service test + the live smoke; here we
assert the HTTP wiring and the resume endpoint's 404s.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app

_SCRIPT = "async def main(input):\n    return input\n"


def _uniq() -> str:
    return secrets.token_hex(4)


@pytest.fixture
async def pool(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    # create_run + resume defer a wake; the procrastinate app isn't open in this ASGI
    # test (the worker drives steps separately), so mock BOTH bindings out — the
    # convention the e2e conftest uses for session defer_wake. (create_run uses the
    # binding in aios.workflows.service; resume_gate_by_nonce uses its own copy in
    # aios.services.workflows.)
    with (
        mock.patch("aios.workflows.service.defer_run_wake", new=mock.AsyncMock()),
        mock.patch("aios.services.workflows.defer_run_wake", new=mock.AsyncMock()),
    ):
        async with authed_client(
            "http://testserver", aios_env["AIOS_API_KEY"], transport=transport
        ) as client:
            yield client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _mint_tenant(http_client: httpx.AsyncClient, name: str) -> str:
    """Mint a child tenant under the root account; return its plaintext API key."""
    r = await http_client.post(
        "/v1/accounts/children", json={"display_name": name, "can_mint_children": False}
    )
    assert r.status_code in (200, 201), r.text
    return str(r.json()["plaintext_key"])


async def _create_env(http_client: httpx.AsyncClient) -> str:
    r = await http_client.post("/v1/environments", json={"name": f"wf-env-{_uniq()}"})
    assert r.status_code in (200, 201), r.text
    return str(r.json()["id"])


async def test_create_workflow_run_and_observe(http_client: httpx.AsyncClient) -> None:
    env_id = await _create_env(http_client)

    # Create a workflow (201).
    name = f"wf-{_uniq()}"
    r = await http_client.post("/v1/workflows", json={"name": name, "script": _SCRIPT})
    assert r.status_code == 201, r.text
    workflow = r.json()
    assert workflow["name"] == name and workflow["version"] == 1

    # It shows up in the list (filtered by name).
    r = await http_client.get("/v1/workflows", params={"name": name})
    assert r.status_code == 200, r.text
    assert [w["id"] for w in r.json()["data"]] == [workflow["id"]]

    # Launch a run (201, pending).
    r = await http_client.post(
        "/v1/runs",
        json={"workflow_id": workflow["id"], "environment_id": env_id, "input": {"x": 1}},
    )
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["workflow_id"] == workflow["id"] and run["status"] == "pending"
    assert run["input"] == {"x": 1}

    # Fetch + list it.
    r = await http_client.get(f"/v1/runs/{run['id']}")
    assert r.status_code == 200 and r.json()["id"] == run["id"]
    r = await http_client.get("/v1/runs", params={"workflow_id": workflow["id"]})
    assert r.status_code == 200 and [x["id"] for x in r.json()["data"]] == [run["id"]]

    # Its journal is readable (a fresh pending run has no events yet — the worker
    # isn't running in this ASGI test).
    r = await http_client.get(f"/v1/runs/{run['id']}/events")
    assert r.status_code == 200 and r.json()["data"] == []


async def test_runs_cross_tenant_isolated(http_client: httpx.AsyncClient) -> None:
    """A run owned by tenant A is invisible to tenant B over the HTTP surface — the
    end-to-end proof that AccountIdDep threads into the scoped reads/ops (a router
    regression that dropped account_id would leak here, where the query/service tests
    can't see it)."""
    key_b = await _mint_tenant(http_client, f"tenant-b-{_uniq()}")

    # Tenant A (the default bearer) creates a workflow + run.
    env_a = await _create_env(http_client)
    wf = (
        await http_client.post("/v1/workflows", json={"name": f"a-{_uniq()}", "script": _SCRIPT})
    ).json()
    run = (
        await http_client.post("/v1/runs", json={"workflow_id": wf["id"], "environment_id": env_a})
    ).json()

    # Tenant B sees none of it: 404 on the point reads + resume, empty list.
    assert (
        await http_client.get(f"/v1/runs/{run['id']}", headers=_bearer(key_b))
    ).status_code == 404
    assert (
        await http_client.get(f"/v1/runs/{run['id']}/events", headers=_bearer(key_b))
    ).status_code == 404
    resume_b = await http_client.post(
        f"/v1/runs/{run['id']}/resume", headers=_bearer(key_b), json={"gate_nonce": "x"}
    )
    assert resume_b.status_code == 404, resume_b.text
    list_b = await http_client.get("/v1/runs", headers=_bearer(key_b))
    assert list_b.status_code == 200 and run["id"] not in [x["id"] for x in list_b.json()["data"]]

    # Tenant A still owns it.
    assert (await http_client.get(f"/v1/runs/{run['id']}")).status_code == 200


async def test_create_run_with_unknown_workflow_404s(http_client: httpx.AsyncClient) -> None:
    env_id = await _create_env(http_client)
    r = await http_client.post(
        "/v1/runs", json={"workflow_id": "wf_does_not_exist", "environment_id": env_id}
    )
    assert r.status_code == 404, r.text


async def test_create_run_cross_tenant_env_404(http_client: httpx.AsyncClient) -> None:
    """POST /v1/runs binding tenant B's environment_id as tenant A must 404.

    Regression guard for the runs path (already fixed): ``create_run`` validates
    the env as account-owned at ``services.py`` before inserting the run, so a
    cross-tenant ``environment_id`` is rejected as NotFound. The sibling sessions
    path is the subject of issue #755; this locks the already-correct runs side.
    """
    key_b = await _mint_tenant(http_client, f"tenant-b-{_uniq()}")

    # Tenant B owns env_b.
    env_b = await http_client.post(
        "/v1/environments", json={"name": f"wf-env-{_uniq()}"}, headers=_bearer(key_b)
    )
    assert env_b.status_code in (200, 201), env_b.text
    env_b_id = env_b.json()["id"]

    # Tenant A (default bearer) owns the workflow.
    wf_resp = await http_client.post(
        "/v1/workflows", json={"name": f"a-{_uniq()}", "script": _SCRIPT}
    )
    assert wf_resp.status_code == 201, wf_resp.text
    wf = wf_resp.json()

    # Tenant A targets tenant B's env_id; expected 404, not a bound run.
    cross = await http_client.post(
        "/v1/runs", json={"workflow_id": wf["id"], "environment_id": env_b_id}
    )
    assert cross.status_code == 404, cross.text


async def test_run_reads_and_resume_404_on_unknown(http_client: httpx.AsyncClient) -> None:
    assert (await http_client.get("/v1/runs/wfr_nope")).status_code == 404
    assert (await http_client.get("/v1/runs/wfr_nope/events")).status_code == 404
    r = await http_client.post("/v1/runs/wfr_nope/resume", json={"gate_nonce": "x"})
    assert r.status_code == 404, r.text


async def test_workflow_description_round_trips(http_client: httpx.AsyncClient) -> None:
    """The optional ``description`` (agent-parity blurb) survives create → get → list,
    and stays ``None`` when omitted."""
    name = f"wf-{_uniq()}"
    r = await http_client.post(
        "/v1/workflows", json={"name": name, "script": _SCRIPT, "description": "does a thing"}
    )
    assert r.status_code == 201, r.text
    wf_id = r.json()["id"]
    assert r.json()["description"] == "does a thing"
    assert (await http_client.get(f"/v1/workflows/{wf_id}")).json()["description"] == "does a thing"
    listed = (await http_client.get("/v1/workflows", params={"name": name})).json()["data"]
    assert listed[0]["description"] == "does a thing"
    # Omitting it is allowed (nullable).
    r2 = await http_client.post("/v1/workflows", json={"name": f"wf-{_uniq()}", "script": _SCRIPT})
    assert r2.status_code == 201 and r2.json()["description"] is None


async def test_cancel_run_endpoint(http_client: httpx.AsyncClient) -> None:
    """The cancel endpoint is wired + account-scoped + idempotent. (The terminal flip
    lands on a worker wake — mocked out here, as for create/resume — so the returned
    run still shows its pre-cancel status; the finalize-to-``cancelled`` path is the
    step integration test's job.)"""
    env_id = await _create_env(http_client)
    wf = (
        await http_client.post("/v1/workflows", json={"name": f"c-{_uniq()}", "script": _SCRIPT})
    ).json()
    run = (
        await http_client.post("/v1/runs", json={"workflow_id": wf["id"], "environment_id": env_id})
    ).json()

    r = await http_client.post(f"/v1/runs/{run['id']}/cancel")
    assert r.status_code == 200 and r.json()["id"] == run["id"], r.text
    # Idempotent: a second cancel is still 200.
    assert (await http_client.post(f"/v1/runs/{run['id']}/cancel")).status_code == 200
    # Unknown + cross-tenant both 404.
    assert (await http_client.post("/v1/runs/wfr_nope/cancel")).status_code == 404
    key_b = await _mint_tenant(http_client, f"tenant-b-{_uniq()}")
    cross = await http_client.post(f"/v1/runs/{run['id']}/cancel", headers=_bearer(key_b))
    assert cross.status_code == 404, cross.text


async def test_runs_parent_run_id_filter(http_client: httpx.AsyncClient, pool: Any) -> None:
    """``GET /v1/runs?parent_run_id=`` scopes to a run's children. Child runs are
    spawned internally (a nested ``workflow()``), so seed one via the query layer and
    assert only it comes back for the parent."""
    from aios.db.queries import workflows as wf_queries
    from aios.workflows.determinism import HOST_SEMANTICS_EPOCH

    env_id = await _create_env(http_client)
    wf = (
        await http_client.post("/v1/workflows", json={"name": f"p-{_uniq()}", "script": _SCRIPT})
    ).json()
    parent = (
        await http_client.post("/v1/runs", json={"workflow_id": wf["id"], "environment_id": env_id})
    ).json()
    async with pool.acquire() as conn:
        child = await wf_queries.insert_wf_run(
            conn,
            account_id=parent["account_id"],
            workflow_id=wf["id"],
            environment_id=env_id,
            script=_SCRIPT,
            script_sha="sha",
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            depth=10,  # #1124: root-budget seed for a directly-inserted run
            parent_run_id=parent["id"],
        )
    # An unrelated (parentless) run that must be excluded.
    await http_client.post("/v1/runs", json={"workflow_id": wf["id"], "environment_id": env_id})

    r = await http_client.get("/v1/runs", params={"parent_run_id": parent["id"]})
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["data"]] == [child.id]  # only the child


async def test_requires_auth(http_client: httpx.AsyncClient) -> None:
    r = await http_client.get("/v1/workflows", headers={"Authorization": ""})
    assert r.status_code == 401, r.text


async def test_update_workflow_roundtrip_and_stale_409(http_client: httpx.AsyncClient) -> None:
    name = f"upd-{_uniq()}"
    r = await http_client.post("/v1/workflows", json={"name": name, "script": _SCRIPT})
    assert r.status_code == 201
    wf = r.json()
    assert wf["version"] == 1

    # PUT with the current version token → 200, version bumps, omitted fields preserved.
    r = await http_client.put(
        f"/v1/workflows/{wf['id']}",
        json={"version": 1, "script": "async def main(input):\n    return 'v2'\n"},
    )
    assert r.status_code == 200
    updated = r.json()
    assert updated["version"] == 2
    assert "v2" in updated["script"]
    assert updated["name"] == name

    # Stale token → 409; unknown id → 404.
    r = await http_client.put(f"/v1/workflows/{wf['id']}", json={"version": 1, "script": "x"})
    assert r.status_code == 409
    r = await http_client.put("/v1/workflows/wf_nope", json={"version": 1, "script": "x"})
    assert r.status_code == 404
