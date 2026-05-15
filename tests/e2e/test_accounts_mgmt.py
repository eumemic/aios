"""E2E tests for the management plane on ``/v1/accounts``.

Covers self-read, child mint / list / get / archive, key mint / list / revoke,
and the authorization invariants (scope = self-or-direct-child, mint requires
``can_mint_children``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.helpers.connections import asgi_client


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
    async with asgi_client(pool) as client:
        yield client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestSelfRead:
    async def test_get_me_returns_caller_account(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.get("/v1/accounts/me", headers=_bearer(aios_env["AIOS_API_KEY"]))
        assert r.status_code == 200, r.text
        body = r.json()
        # The aios_env fixture inserts acc_test_stub as the root with can_mint_children=True.
        assert body["id"] == "acc_test_stub"
        assert body["parent_account_id"] is None
        assert body["can_mint_children"] is True


class TestMintChild:
    async def test_mint_child_returns_plaintext_once(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "tenant-a", "can_mint_children": False},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["account_id"].startswith("acc_")
        assert body["key_id"].startswith("acckey_")
        assert body["plaintext_key"].startswith("aios_")

    async def test_minted_child_can_authenticate(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "tenant-b", "can_mint_children": False},
        )
        assert r.status_code == 201, r.text
        child_key = r.json()["plaintext_key"]
        me = await http_client.get("/v1/accounts/me", headers=_bearer(child_key))
        assert me.status_code == 200
        assert me.json()["display_name"] == "tenant-b"
        assert me.json()["parent_account_id"] == "acc_test_stub"
        assert me.json()["can_mint_children"] is False

    async def test_child_without_mint_flag_403s_on_grandchild(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "tenant-c", "can_mint_children": False},
        )
        assert r.status_code == 201
        child_key = r.json()["plaintext_key"]
        r2 = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(child_key),
            json={"display_name": "grandchild", "can_mint_children": False},
        )
        assert r2.status_code == 403, r2.text

    async def test_sibling_unique_display_name(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        a = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "dup-name"},
        )
        assert a.status_code == 201
        b = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "dup-name"},
        )
        assert b.status_code == 409, b.text


class TestChildScope:
    async def test_list_children_shows_minted(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "kid-1"},
        )
        await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "kid-2"},
        )
        r = await http_client.get(
            "/v1/accounts/children", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 200
        names = {a["display_name"] for a in r.json()}
        assert {"kid-1", "kid-2"} <= names

    async def test_get_child_by_id(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "kid-3"},
        )
        child_id = m.json()["account_id"]
        r = await http_client.get(
            f"/v1/accounts/{child_id}", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 200
        assert r.json()["id"] == child_id

    async def test_cross_tenant_get_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """Account A can't see account B even if both exist under the same root."""
        # Two siblings under root.
        a = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "siblingA"},
        )
        b = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "siblingB"},
        )
        a_key = a.json()["plaintext_key"]
        b_id = b.json()["account_id"]
        # A tries to read B → 404 (not 403 — the API doesn't reveal existence).
        r = await http_client.get(f"/v1/accounts/{b_id}", headers=_bearer(a_key))
        assert r.status_code == 404, r.text


class TestByPath:
    async def test_root_path_returns_caller(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        for path in ("", "/"):
            r = await http_client.get(
                "/v1/accounts/by-path",
                params={"path": path},
                headers=_bearer(aios_env["AIOS_API_KEY"]),
            )
            assert r.status_code == 200, r.text
            assert r.json()["id"] == "acc_test_stub"

    async def test_resolves_child_by_name(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "by-path-target"},
        )
        r = await http_client.get(
            "/v1/accounts/by-path",
            params={"path": "by-path-target"},
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 200, r.text
        assert r.json()["display_name"] == "by-path-target"

    async def test_resolves_grandchild_by_two_segments(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        # Mint child with mint-children, then mint grandchild under it.
        c = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "tenant-x", "can_mint_children": True},
        )
        ck = c.json()["plaintext_key"]
        await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(ck),
            json={"display_name": "team-a"},
        )
        r = await http_client.get(
            "/v1/accounts/by-path",
            params={"path": "tenant-x/team-a"},
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 200, r.text
        assert r.json()["display_name"] == "team-a"

    async def test_missing_segment_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.get(
            "/v1/accounts/by-path",
            params={"path": "no-such-child"},
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 404


class TestKeys:
    async def test_mint_key_on_self(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/acc_test_stub/keys",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"label": "ci-rotation-1"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["key_id"].startswith("acckey_")
        assert body["plaintext_key"].startswith("aios_")
        # New key authenticates immediately.
        me = await http_client.get("/v1/accounts/me", headers=_bearer(body["plaintext_key"]))
        assert me.status_code == 200
        assert me.json()["id"] == "acc_test_stub"

    async def test_list_keys_excludes_hash(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        await http_client.post(
            "/v1/accounts/acc_test_stub/keys",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"label": "rotation-2"},
        )
        r = await http_client.get(
            "/v1/accounts/acc_test_stub/keys", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) >= 2  # bootstrap key + the one we just minted
        for row in rows:
            assert "hash" not in row
            assert set(row.keys()) == {"key_id", "label", "created_at", "revoked_at"}

    async def test_revoke_key_then_fails_auth(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        # Mint a key, use it once, revoke it, use again → 401.
        m = await http_client.post(
            "/v1/accounts/acc_test_stub/keys",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"label": "shortlived"},
        )
        new_key = m.json()["plaintext_key"]
        new_key_id = m.json()["key_id"]
        ok = await http_client.get("/v1/accounts/me", headers=_bearer(new_key))
        assert ok.status_code == 200
        rev = await http_client.delete(
            f"/v1/accounts/acc_test_stub/keys/{new_key_id}",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert rev.status_code == 204
        denied = await http_client.get("/v1/accounts/me", headers=_bearer(new_key))
        assert denied.status_code == 401

    async def test_revoke_unknown_key_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.delete(
            "/v1/accounts/acc_test_stub/keys/acckey_nonexistent",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 404, r.text


class TestUpdate:
    async def test_update_display_name(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "before-rename"},
        )
        child_id = m.json()["account_id"]
        r = await http_client.patch(
            f"/v1/accounts/{child_id}",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "after-rename"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["display_name"] == "after-rename"

    async def test_update_can_mint_children(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "mint-flag-toggle", "can_mint_children": False},
        )
        child_id = m.json()["account_id"]
        r = await http_client.patch(
            f"/v1/accounts/{child_id}",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"can_mint_children": True},
        )
        assert r.status_code == 200
        assert r.json()["can_mint_children"] is True

    async def test_update_cross_tenant_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        a = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "patch-a"},
        )
        b = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "patch-b"},
        )
        a_key = a.json()["plaintext_key"]
        b_id = b.json()["account_id"]
        r = await http_client.patch(
            f"/v1/accounts/{b_id}",
            headers=_bearer(a_key),
            json={"display_name": "stolen"},
        )
        assert r.status_code == 404, r.text

    async def test_update_no_fields_is_no_op(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.patch(
            "/v1/accounts/acc_test_stub",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={},
        )
        assert r.status_code == 200
        assert r.json()["id"] == "acc_test_stub"


class TestPurge:
    async def test_purge_unarchived_409(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        """Cannot purge a non-archived account — must archive first."""
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "purge-not-yet-archived"},
        )
        child_id = m.json()["account_id"]
        r = await http_client.post(
            f"/v1/accounts/{child_id}/purge",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 409, r.text

    async def test_archive_then_purge_succeeds(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "purge-target"},
        )
        child_id = m.json()["account_id"]
        # Step 1: archive.
        r = await http_client.delete(
            f"/v1/accounts/{child_id}", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 200
        # Step 2: purge.
        r = await http_client.post(
            f"/v1/accounts/{child_id}/purge",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 204, r.text
        # Step 3: confirm the row is gone.
        r = await http_client.get(
            f"/v1/accounts/{child_id}",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 404

    async def test_purge_self_409(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/acc_test_stub/purge",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
        )
        assert r.status_code == 409

    async def test_purge_cross_tenant_404(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        a = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "purge-isolated-a"},
        )
        b = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "purge-isolated-b"},
        )
        a_key = a.json()["plaintext_key"]
        b_id = b.json()["account_id"]
        r = await http_client.post(f"/v1/accounts/{b_id}/purge", headers=_bearer(a_key))
        assert r.status_code == 404, r.text


class TestArchive:
    async def test_self_archive_409(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        r = await http_client.delete(
            "/v1/accounts/acc_test_stub", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 409, r.text

    async def test_archive_child(
        self, http_client: httpx.AsyncClient, aios_env: dict[str, str]
    ) -> None:
        m = await http_client.post(
            "/v1/accounts/children",
            headers=_bearer(aios_env["AIOS_API_KEY"]),
            json={"display_name": "to-archive"},
        )
        child_id = m.json()["account_id"]
        r = await http_client.delete(
            f"/v1/accounts/{child_id}", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        assert r.status_code == 200
        assert r.json()["archived_at"] is not None
        # Listing children no longer surfaces it.
        listed = await http_client.get(
            "/v1/accounts/children", headers=_bearer(aios_env["AIOS_API_KEY"])
        )
        ids = {a["id"] for a in listed.json()}
        assert child_id not in ids
