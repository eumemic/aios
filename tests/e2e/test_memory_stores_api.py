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
