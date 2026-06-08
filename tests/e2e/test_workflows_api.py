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

from tests.helpers.connections import authed_client

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
    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    settings = get_settings()
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()

    transport = httpx.ASGITransport(app=app)
    # create_run defers a wake; the procrastinate app isn't open in this ASGI test
    # (the worker drives steps separately), so mock it out — the convention the e2e
    # conftest uses for session defer_wake.
    with mock.patch("aios.workflows.service.defer_run_wake", new=mock.AsyncMock()):
        async with authed_client(
            "http://testserver", aios_env["AIOS_API_KEY"], transport=transport
        ) as client:
            yield client


async def _create_env(http_client: httpx.AsyncClient) -> str:
    r = await http_client.post("/v1/environments", json={"name": f"wf-env-{_uniq()}"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


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


async def test_create_run_with_unknown_workflow_404s(http_client: httpx.AsyncClient) -> None:
    env_id = await _create_env(http_client)
    r = await http_client.post(
        "/v1/runs", json={"workflow_id": "wf_does_not_exist", "environment_id": env_id}
    )
    assert r.status_code == 404, r.text


async def test_run_reads_and_resume_404_on_unknown(http_client: httpx.AsyncClient) -> None:
    assert (await http_client.get("/v1/runs/wfr_nope")).status_code == 404
    assert (await http_client.get("/v1/runs/wfr_nope/events")).status_code == 404
    r = await http_client.post("/v1/runs/wfr_nope/resume", json={"gate_nonce": "x"})
    assert r.status_code == 404, r.text


async def test_requires_auth(http_client: httpx.AsyncClient) -> None:
    r = await http_client.get("/v1/workflows", headers={"Authorization": ""})
    assert r.status_code == 401, r.text
