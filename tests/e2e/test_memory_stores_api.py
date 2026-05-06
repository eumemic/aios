"""E2E tests for ``/v1/memory-stores`` HTTP endpoints.

Real Postgres (testcontainer) + real FastAPI app via ASGI transport. Covers
CRUD on stores, memories, and versions, plus the failure modes the design
rests on: precondition mismatch, path conflict, redact-head rejection,
archive-then-write rejection.

No Docker container is provisioned in this file — these tests stay at the
HTTP/DB layer. Sandbox-mount sync is exercised by the harness E2E tests.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest


def _uniq() -> str:
    return secrets.token_hex(4)


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"},
    ) as client:
        yield client


async def _create_store(http_client: httpx.AsyncClient, **kwargs: Any) -> dict[str, Any]:
    r = await http_client.post(
        "/v1/memory-stores",
        json={"name": kwargs.get("name", f"store-{_uniq()}"), **kwargs},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _create_memory(
    http_client: httpx.AsyncClient, store_id: str, path: str, content: str
) -> dict[str, Any]:
    r = await http_client.post(
        f"/v1/memory-stores/{store_id}/memories",
        json={"path": path, "content": content},
    )
    assert r.status_code == 201, r.text
    return r.json()


class TestStoreCrud:
    async def test_create_get_update_archive_delete(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client, description="initial", metadata={"team": "x"})
        store_id = store["id"]
        assert store_id.startswith("memstore_")
        assert store["type"] == "memory_store"
        assert store["archived_at"] is None

        r = await http_client.get(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 200, r.text
        assert r.json()["description"] == "initial"

        r = await http_client.post(
            f"/v1/memory-stores/{store_id}",
            json={"description": "updated"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["description"] == "updated"

        r = await http_client.post(f"/v1/memory-stores/{store_id}/archive")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        r = await http_client.delete(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 404, r.text


class TestMemoryCrud:
    async def test_create_retrieve_round_trip(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/notes.md", "hello")
        assert mem["id"].startswith("mem_")
        assert mem["memory_version_id"].startswith("memver_")
        assert mem.get("content") is None  # create response omits content
        assert mem["content_sha256"] == _sha("hello")
        assert mem["content_size_bytes"] == 5

        r = await http_client.get(f"/v1/memory-stores/{store['id']}/memories/{mem['id']}")
        assert r.status_code == 200, r.text
        assert r.json()["content"] == "hello"

    async def test_path_conflict_returns_409(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "first")
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories",
            json={"path": "/x.md", "content": "second"},
        )
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["error"]["type"] == "memory_path_conflict_error"
        assert body["error"]["detail"]["conflicting_memory_id"] == mem["id"]
        assert body["error"]["detail"]["conflicting_path"] == "/x.md"

    async def test_update_with_stale_precondition_409(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "v1")

        # Bump content via update so the head sha changes.
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories/{mem['id']}",
            json={"content": "v2"},
        )
        assert r.status_code == 200, r.text

        # Now try to update again with the original sha — should 409.
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories/{mem['id']}",
            json={
                "content": "v3",
                "precondition": {
                    "type": "content_sha256",
                    "content_sha256": _sha("v1"),
                },
            },
        )
        assert r.status_code == 409, r.text
        assert r.json()["error"]["type"] == "memory_precondition_failed_error"

    async def test_rename_via_update(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "x")
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories/{mem['id']}",
            json={"path": "/y.md"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["path"] == "/y.md"
        assert r.json()["id"] == mem["id"]  # same id, new version

    async def test_delete_creates_tombstone_version(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "x")
        r = await http_client.delete(f"/v1/memory-stores/{store['id']}/memories/{mem['id']}")
        assert r.status_code == 204, r.text

        r = await http_client.get(
            f"/v1/memory-stores/{store['id']}/memory-versions",
            params={"memory_id": mem["id"]},
        )
        assert r.status_code == 200, r.text
        ops = [v["operation"] for v in r.json()["data"]]
        assert "deleted" in ops

    async def test_list_with_depth_returns_prefixes(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        await _create_memory(http_client, store["id"], "/facts/team.md", "x")
        await _create_memory(http_client, store["id"], "/preferences/style.md", "y")

        r = await http_client.get(
            f"/v1/memory-stores/{store['id']}/memories",
            params={"path_prefix": "/", "depth": 1, "order_by": "path"},
        )
        assert r.status_code == 200, r.text
        types = {item["type"] for item in r.json()["data"]}
        assert "memory_prefix" in types

    async def test_depth_without_order_by_409(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        r = await http_client.get(
            f"/v1/memory-stores/{store['id']}/memories",
            params={"depth": 1},
        )
        assert r.status_code == 409, r.text

    async def test_list_path_prefix_treats_underscore_literally(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """``path_prefix`` is a literal prefix, not a SQL ``LIKE`` pattern.

        The schema allows ``_`` and ``%`` in path segments, so they must
        not act as wildcards in filter queries.
        """
        store = await _create_store(http_client)
        await _create_memory(http_client, store["id"], "/notes/draft_alice.md", "a")
        await _create_memory(http_client, store["id"], "/notes/draftXalice.md", "b")

        r = await http_client.get(
            f"/v1/memory-stores/{store['id']}/memories",
            params={"path_prefix": "/notes/draft_a"},
        )
        assert r.status_code == 200, r.text
        paths = sorted(item["path"] for item in r.json()["data"])
        assert paths == ["/notes/draft_alice.md"]

        # ``%`` must not match every path in the store.
        r = await http_client.get(
            f"/v1/memory-stores/{store['id']}/memories",
            params={"path_prefix": "%"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["data"] == []


class TestVersionsAndRedact:
    async def test_redact_head_rejected(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "v1")
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memory-versions/{mem['memory_version_id']}/redact"
        )
        assert r.status_code == 409, r.text

    async def test_redact_old_succeeds(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        mem = await _create_memory(http_client, store["id"], "/x.md", "v1")
        head_v1 = mem["memory_version_id"]

        # Modify so v1 is no longer the head.
        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories/{mem['id']}",
            json={"content": "v2"},
        )
        assert r.status_code == 200, r.text

        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memory-versions/{head_v1}/redact"
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["redacted_at"] is not None
        assert body["redacted_by"]["type"] == "api_actor"
        # Content fields stripped on redact:
        assert body.get("path") is None
        assert body.get("content") is None
        assert body.get("content_sha256") is None
        assert body.get("content_size_bytes") is None


class TestArchivedStoreRejectsWrites:
    async def test_archived_store_blocks_create(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        await http_client.post(f"/v1/memory-stores/{store['id']}/archive")

        r = await http_client.post(
            f"/v1/memory-stores/{store['id']}/memories",
            json={"path": "/x.md", "content": "x"},
        )
        assert r.status_code == 400, r.text
        assert r.json()["error"]["type"] == "memory_store_archived_error"


async def _create_session_for_resources_test(
    pool: Any,
    *,
    initial_resources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Spin up an agent + environment + session via service layer.

    Uses the service layer directly (not HTTP) for setup boilerplate to
    keep the resource-update tests below focused on the new code path.
    Returns the session as a serializable dict so call-sites can read .id
    without importing the model.
    """
    from aios.models.memory_stores import MemoryStoreResource
    from aios.services import agents as agents_svc
    from aios.services import environments as env_svc
    from aios.services import sessions as sess_svc

    env = await env_svc.create_environment(pool, name=f"resources-update-{_uniq()}")
    agent = await agents_svc.create_agent(
        pool,
        name=f"resources-update-agent-{_uniq()}",
        model="fake/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    resources = (
        None
        if initial_resources is None
        else [MemoryStoreResource.model_validate(r) for r in initial_resources]
    )
    session = await sess_svc.create_session(
        pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="resources-update",
        metadata={},
        resources=resources,
    )
    return {"id": session.id, "agent_id": agent.id, "env_id": env.id}


class TestSessionResourcesUpdate:
    """``PUT /v1/sessions/{id}`` with ``resources`` attaches/detaches memory
    stores on a running session (issue #198)."""

    async def test_attach_via_update(self, http_client: httpx.AsyncClient, pool: Any) -> None:
        store = await _create_store(http_client, name=f"attach-{_uniq()}")
        session = await _create_session_for_resources_test(pool)

        r = await http_client.put(
            f"/v1/sessions/{session['id']}",
            json={
                "resources": [
                    {"type": "memory_store", "memory_store_id": store["id"]},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["resources"]) == 1
        echo = body["resources"][0]
        assert echo["memory_store_id"] == store["id"]
        assert echo["name"] == store["name"]
        assert echo["mount_path"] == f"/mnt/memory/{store['name']}"
        assert echo["access"] == "read_write"

        # GET round-trips the same set.
        r = await http_client.get(f"/v1/sessions/{session['id']}")
        assert r.status_code == 200, r.text
        assert len(r.json()["resources"]) == 1

    async def test_detach_via_empty_list(self, http_client: httpx.AsyncClient, pool: Any) -> None:
        store = await _create_store(http_client, name=f"detach-{_uniq()}")
        session = await _create_session_for_resources_test(
            pool,
            initial_resources=[{"type": "memory_store", "memory_store_id": store["id"]}],
        )

        r = await http_client.put(f"/v1/sessions/{session['id']}", json={"resources": []})
        assert r.status_code == 200, r.text
        assert r.json()["resources"] == []

    async def test_omitted_resources_preserves_existing(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        store = await _create_store(http_client, name=f"preserve-{_uniq()}")
        session = await _create_session_for_resources_test(
            pool,
            initial_resources=[{"type": "memory_store", "memory_store_id": store["id"]}],
        )

        # Update unrelated field; resources field omitted entirely.
        r = await http_client.put(f"/v1/sessions/{session['id']}", json={"title": "new"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "new"
        assert len(body["resources"]) == 1
        assert body["resources"][0]["memory_store_id"] == store["id"]

    async def test_replace_swaps_attached_set(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        s1 = await _create_store(http_client, name=f"swap-a-{_uniq()}")
        s2 = await _create_store(http_client, name=f"swap-b-{_uniq()}")
        session = await _create_session_for_resources_test(
            pool,
            initial_resources=[{"type": "memory_store", "memory_store_id": s1["id"]}],
        )

        r = await http_client.put(
            f"/v1/sessions/{session['id']}",
            json={
                "resources": [
                    {"type": "memory_store", "memory_store_id": s2["id"]},
                ],
            },
        )
        assert r.status_code == 200, r.text
        ids = [e["memory_store_id"] for e in r.json()["resources"]]
        assert ids == [s2["id"]]

    async def test_dup_id_rejected(self, http_client: httpx.AsyncClient, pool: Any) -> None:
        store = await _create_store(http_client, name=f"dup-{_uniq()}")
        session = await _create_session_for_resources_test(pool)

        r = await http_client.put(
            f"/v1/sessions/{session['id']}",
            json={
                "resources": [
                    {"type": "memory_store", "memory_store_id": store["id"]},
                    {"type": "memory_store", "memory_store_id": store["id"]},
                ],
            },
        )
        assert r.status_code == 422, r.text

    async def test_name_conflict_rolls_back(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        """Two stores resolving to the same mount path must reject and
        leave the prior set intact."""
        prior = await _create_store(http_client, name=f"prior-{_uniq()}")
        clash_name = f"clash-{_uniq()}"
        s1 = await _create_store(http_client, name=clash_name)
        s2 = await _create_store(http_client, name=clash_name + "-x")
        # Rename s2 to clash with s1 so two distinct ids share the same name.
        r = await http_client.post(f"/v1/memory-stores/{s2['id']}", json={"name": clash_name})
        assert r.status_code == 200, r.text

        session = await _create_session_for_resources_test(
            pool,
            initial_resources=[{"type": "memory_store", "memory_store_id": prior["id"]}],
        )

        r = await http_client.put(
            f"/v1/sessions/{session['id']}",
            json={
                "resources": [
                    {"type": "memory_store", "memory_store_id": s1["id"]},
                    {"type": "memory_store", "memory_store_id": s2["id"]},
                ],
            },
        )
        assert r.status_code == 409, r.text

        # Prior attachment survives the rollback.
        r = await http_client.get(f"/v1/sessions/{session['id']}")
        assert r.status_code == 200, r.text
        ids = [e["memory_store_id"] for e in r.json()["resources"]]
        assert ids == [prior["id"]]

    async def test_unknown_store_rejected_and_rolls_back(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        prior = await _create_store(http_client, name=f"prior-{_uniq()}")
        session = await _create_session_for_resources_test(
            pool,
            initial_resources=[{"type": "memory_store", "memory_store_id": prior["id"]}],
        )

        r = await http_client.put(
            f"/v1/sessions/{session['id']}",
            json={
                "resources": [
                    {"type": "memory_store", "memory_store_id": "memstore_doesnotexist"},
                ],
            },
        )
        assert r.status_code == 404, r.text

        # Prior attachment survives the rollback.
        r = await http_client.get(f"/v1/sessions/{session['id']}")
        assert r.status_code == 200, r.text
        ids = [e["memory_store_id"] for e in r.json()["resources"]]
        assert ids == [prior["id"]]
