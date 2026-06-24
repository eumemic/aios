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

import httpx
import pytest

from tests.helpers.connections import authed_client, wired_app


def _uniq() -> str:
    return secrets.token_hex(4)


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@pytest.fixture
async def http_client(pool: Any, aios_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=wired_app(pool))
    async with authed_client(
        "http://testserver",
        aios_env["AIOS_API_KEY"],
        transport=transport,
    ) as client:
        yield client


async def _create_store(http_client: httpx.AsyncClient, **kwargs: Any) -> dict[str, Any]:
    r = await http_client.post(
        "/v1/memory-stores",
        json={"name": kwargs.get("name", f"store-{_uniq()}"), **kwargs},
    )
    assert r.status_code == 201, r.text
    result: dict[str, Any] = r.json()
    return result


async def _create_memory(
    http_client: httpx.AsyncClient, store_id: str, path: str, content: str
) -> dict[str, Any]:
    r = await http_client.post(
        f"/v1/memory-stores/{store_id}/memories",
        json={"path": path, "content": content},
    )
    assert r.status_code == 201, r.text
    result: dict[str, Any] = r.json()
    return result


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

        r = await http_client.put(
            f"/v1/memory-stores/{store_id}",
            json={"description": "updated"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["description"] == "updated"

        r = await http_client.post(f"/v1/memory-stores/{store_id}/archive")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        # Bare DELETE soft-archives (T2): the row persists, still fetchable.
        r = await http_client.delete(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 200, r.text
        assert r.json()["archived_at"] is not None

        r = await http_client.get(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 200, r.text

        # Explicit /purge hard-deletes the row.
        r = await http_client.post(f"/v1/memory-stores/{store_id}/purge")
        assert r.status_code == 204, r.text

        r = await http_client.get(f"/v1/memory-stores/{store_id}")
        assert r.status_code == 404, r.text


class TestStoreListPagination:
    """The opaque ``next_cursor`` returned by ``GET /v1/memory-stores`` must be
    acceptable as ``?cursor=`` on the next call, advancing past the prior page.
    """

    async def test_cursor_advances_pagination(self, http_client: httpx.AsyncClient) -> None:
        # Create 5 stores so a limit=2 walk takes 3 pages.
        for _ in range(5):
            await _create_store(http_client)

        r1 = await http_client.get("/v1/memory-stores", params={"limit": 2})
        assert r1.status_code == 200, r1.text
        page1 = r1.json()
        assert len(page1["data"]) == 2
        assert page1["has_more"] is True
        assert page1["next_cursor"]

        # Second page via ?cursor= → ids MUST be disjoint from page1.
        r2 = await http_client.get("/v1/memory-stores", params={"cursor": page1["next_cursor"]})
        assert r2.status_code == 200, r2.text
        page2 = r2.json()
        assert len(page2["data"]) == 2
        page1_ids = {row["id"] for row in page1["data"]}
        page2_ids = {row["id"] for row in page2["data"]}
        assert page1_ids.isdisjoint(page2_ids), (
            f"page 2 must advance past page 1's cursor; got overlap {page1_ids & page2_ids}."
        )


class TestMemoryListLimit:
    """``GET /v1/memory-stores/{id}/memories`` must cap response size.

    Pre-fix the router had no ``limit`` parameter and the underlying SQL
    had no ``LIMIT`` clause, so a store with thousands of memories
    returned them all in one response — unbounded memory pressure on
    the API process and unbounded payload size on the wire."""

    async def test_default_limit_caps_response(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        store_id = store["id"]
        # Insert 150 memories — above the default cap (100) so the
        # response must be truncated even without an explicit ?limit=.
        for i in range(150):
            await _create_memory(http_client, store_id, f"/note-{i:03d}.md", f"body {i}")

        r = await http_client.get(f"/v1/memory-stores/{store_id}/memories")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 100, (
            f"default limit should cap at 100; got {len(body['data'])}. Pre-fix "
            f"symptom: no LIMIT clause returned all 150 rows in one response."
        )
        assert body["has_more"] is True

    async def test_explicit_limit_honored(self, http_client: httpx.AsyncClient) -> None:
        store = await _create_store(http_client)
        store_id = store["id"]
        for i in range(15):
            await _create_memory(http_client, store_id, f"/x-{i:03d}.md", f"b{i}")
        r = await http_client.get(f"/v1/memory-stores/{store_id}/memories", params={"limit": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 5
        assert body["has_more"] is True


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
        r = await http_client.put(
            f"/v1/memory-stores/{store['id']}/memories/{mem['id']}",
            json={"content": "v2"},
        )
        assert r.status_code == 200, r.text

        # Now try to update again with the original sha — should 409.
        r = await http_client.put(
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
        r = await http_client.put(
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
        r = await http_client.put(
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
    account_id = "acc_test_stub"  # PR 3 scaffolding
    from aios.models.github_repositories import GithubRepositoryResource
    from aios.models.memory_stores import MemoryStoreResource
    from aios.services import agents as agents_svc
    from aios.services import environments as env_svc
    from aios.services import sessions as sess_svc

    env = await env_svc.create_environment(
        pool, name=f"resources-update-{_uniq()}", account_id=account_id
    )
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
        account_id=account_id,
    )
    resources: list[MemoryStoreResource | GithubRepositoryResource] | None = (
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
        account_id=account_id,
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
        r = await http_client.put(f"/v1/memory-stores/{s2['id']}", json={"name": clash_name})
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


class TestSessionListResourcesBatched:
    """``GET /v1/sessions`` enriches each row with its ``resources`` field via a
    single batched query per echo family — regression guard for issue #617."""

    async def test_list_returns_resources_for_each_session(
        self, http_client: httpx.AsyncClient, pool: Any
    ) -> None:
        from aios.models.memory_stores import MemoryStoreResource
        from aios.services import agents as agents_svc
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        account_id = "acc_test_stub"
        store_a = await _create_store(http_client, name=f"list-a-{_uniq()}")
        store_b = await _create_store(http_client, name=f"list-b-{_uniq()}")
        store_c = await _create_store(http_client, name=f"list-c-{_uniq()}")

        env = await env_svc.create_environment(
            pool, name=f"list-resources-{_uniq()}", account_id=account_id
        )
        agent = await agents_svc.create_agent(
            pool,
            name=f"list-resources-agent-{_uniq()}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )

        async def _mk(initial: list[str]) -> str:
            s = await sess_svc.create_session(
                pool,
                agent_id=agent.id,
                environment_id=env.id,
                title="list-resources",
                metadata={},
                resources=[
                    MemoryStoreResource(type="memory_store", memory_store_id=sid) for sid in initial
                ],
                account_id=account_id,
            )
            return s.id

        two_id = await _mk([store_a["id"], store_b["id"]])
        zero_id = await _mk([])
        one_id = await _mk([store_c["id"]])

        r = await http_client.get(f"/v1/sessions?agent_id={agent.id}")
        assert r.status_code == 200, r.text
        by_id = {s["id"]: s for s in r.json()["data"]}
        assert by_id.keys() == {two_id, zero_id, one_id}

        ids_two = [e["memory_store_id"] for e in by_id[two_id]["resources"]]
        assert ids_two == [store_a["id"], store_b["id"]]

        assert by_id[zero_id]["resources"] == []

        ids_one = [e["memory_store_id"] for e in by_id[one_id]["resources"]]
        assert ids_one == [store_c["id"]]
