"""E2E tests for ``POST /v1/accounts/bootstrap``."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest import mock

import httpx
import pytest

from tests.helpers.connections import asgi_client


@pytest.fixture
async def pool(aios_env_minimal: dict[str, str]) -> AsyncIterator[Any]:
    """Pool against a freshly-migrated DB with NO seeded root account —
    the bootstrap-endpoint tests need to exercise the pre-bootstrap state.
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    p = await create_pool(settings.db_url, min_size=1, max_size=4)
    yield p
    await p.close()


@pytest.fixture
def bootstrap_env(aios_env_minimal: dict[str, str]) -> Iterator[dict[str, str]]:
    token = secrets.token_urlsafe(32)
    extra = {"AIOS_BOOTSTRAP_TOKEN": token}
    with mock.patch.dict(os.environ, extra):
        from aios.config import get_settings

        get_settings.cache_clear()
        yield {**aios_env_minimal, **extra}
        get_settings.cache_clear()


@pytest.fixture
async def http_client(pool: Any, bootstrap_env: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    from aios.config import get_settings

    get_settings.cache_clear()
    async with asgi_client(pool) as client:
        yield client


@pytest.fixture
async def http_client_no_bootstrap_env(
    pool: Any, aios_env_minimal: dict[str, str]
) -> AsyncIterator[httpx.AsyncClient]:
    """Client built without ``AIOS_BOOTSTRAP_TOKEN`` in env — used by the
    env-unset test, which would otherwise see the token set by
    ``bootstrap_env`` that the regular ``http_client`` depends on.
    """
    from aios.config import get_settings

    get_settings.cache_clear()
    async with asgi_client(pool) as client:
        yield client


def _bootstrap_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestBootstrap:
    async def test_success_returns_account_and_plaintext_key(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers(token),
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["account_id"].startswith("acc_")
        assert body["key_id"].startswith("acckey_")
        assert body["plaintext_key"].startswith("aios_")

    async def test_persists_hashed_key_not_plaintext(
        self,
        http_client: httpx.AsyncClient,
        bootstrap_env: dict[str, str],
        pool: Any,
    ) -> None:
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers(token),
        )
        from aios.services.accounts import hash_key

        assert r.status_code == 201
        plaintext = r.json()["plaintext_key"]
        expected_hash = hash_key(plaintext)
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT hash, label FROM account_keys")
        assert row is not None
        assert bytes(row["hash"]) == expected_hash
        assert row["label"] == "bootstrap"

    async def test_404_when_root_already_exists(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        first = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers(token),
        )
        assert first.status_code == 201
        second = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers(token),
        )
        assert second.status_code == 404, second.text

    async def test_404_takes_precedence_over_401(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        """Wrong token + existing root = 404, not 401.

        Bootstrapped systems must look identical to outsiders regardless
        of whether they hold the bootstrap token. Otherwise the response
        code leaks whether the system has ever been bootstrapped.
        """
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        first = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers(token),
        )
        assert first.status_code == 201
        wrong = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers("totally-wrong-token"),
        )
        assert wrong.status_code == 404, wrong.text

    async def test_401_on_wrong_token(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers("not-the-right-token"),
        )
        assert r.status_code == 401, r.text

    async def test_401_on_missing_authorization(self, http_client: httpx.AsyncClient) -> None:
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
        )
        assert r.status_code == 401, r.text

    async def test_401_when_bootstrap_token_env_unset(
        self,
        http_client_no_bootstrap_env: httpx.AsyncClient,
    ) -> None:
        """When ``AIOS_BOOTSTRAP_TOKEN`` is unset, every call to the
        bootstrap endpoint is 401 — even one carrying a non-empty
        header. The endpoint requires an explicit opt-in from the
        operator on every deploy that wants bootstrap available.
        """
        r = await http_client_no_bootstrap_env.post(
            "/v1/accounts/bootstrap",
            json={"display_name": "root"},
            headers=_bootstrap_headers("anything-at-all"),
        )
        assert r.status_code == 401, r.text

    async def test_400_on_missing_display_name(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={},
            headers=_bootstrap_headers(token),
        )
        # FastAPI's RequestValidationError handler returns 422
        assert r.status_code == 422, r.text

    async def test_400_on_empty_display_name(
        self, http_client: httpx.AsyncClient, bootstrap_env: dict[str, str]
    ) -> None:
        token = bootstrap_env["AIOS_BOOTSTRAP_TOKEN"]
        r = await http_client.post(
            "/v1/accounts/bootstrap",
            json={"display_name": ""},
            headers=_bootstrap_headers(token),
        )
        assert r.status_code == 422, r.text

    async def test_concurrent_bootstrap_loser_gets_404(self, pool: Any) -> None:
        """The race-losing call gets 404, not 409.

        Simulates the TOCTOU window: an existence-check passes for both
        callers because no root exists yet, and they race to INSERT. The
        ``accounts_one_active_root`` index lets exactly one win; the
        loser's ``UniqueViolationError`` must surface as ``NotFoundError``
        (404), matching the post-bootstrap behavior — otherwise the
        endpoint's stated invariant breaks under concurrency.
        """
        # Pre-seed a root to put the DB in the "already bootstrapped"
        # state, then call ``bootstrap_root_account`` directly — this is
        # the same code path the race-loser hits at the INSERT step.
        from aios.db import queries
        from aios.errors import NotFoundError

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_existing_root', NULL, TRUE, 'preexisting')
                """
            )
            with pytest.raises(NotFoundError):
                await queries.bootstrap_root_account(
                    conn,
                    display_name="loser",
                    key_hash=b"\x00" * 32,
                    key_label="bootstrap",
                )


class TestAccountInvariants:
    """Direct DB tests for the ``accounts`` partial-unique indexes.

    These exercise the DDL itself (not the bootstrap endpoint) so a
    future refactor of the endpoint doesn't accidentally remove the
    safety net at the schema layer.
    """

    async def test_two_active_roots_violate_uniqueness(self, pool: Any) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_test_root_a', NULL, TRUE, 'first-root')
                """
            )
            import asyncpg.exceptions

            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                    VALUES ('acc_test_root_b', NULL, TRUE, 'second-root')
                    """
                )

    async def test_archived_root_allows_new_root(self, pool: Any) -> None:
        """Soft-archived roots free up the active-root slot.

        The recovery doc relies on this: archive the stale root, then
        bootstrap a fresh one.
        """
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name, archived_at)
                VALUES ('acc_archived_root', NULL, TRUE, 'old-root', now())
                """
            )
            # Should now succeed because the only existing root is archived.
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_fresh_root', NULL, TRUE, 'new-root')
                """
            )

    async def test_sibling_name_uniqueness(self, pool: Any) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_parent', NULL, TRUE, 'parent')
                """
            )
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_child_a', 'acc_parent', FALSE, 'alice')
                """
            )
            import asyncpg.exceptions

            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                    VALUES ('acc_child_b', 'acc_parent', FALSE, 'alice')
                    """
                )

    async def test_distinct_parents_allow_same_child_name(self, pool: Any) -> None:
        """``jarvis-prod/alice`` and ``product-x/alice`` are both legal —
        siblings only need to be unique within their own parent.

        Realistic shape: one root, two products under it, each product has
        an ``alice`` child. The active-root index allows only one root, so
        the products live underneath that single root.
        """
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES
                    ('acc_root',   NULL,        TRUE,  'root'),
                    ('acc_jarvis', 'acc_root',  TRUE,  'jarvis-prod'),
                    ('acc_prodx',  'acc_root',  TRUE,  'product-x'),
                    ('acc_alice_jarvis', 'acc_jarvis', FALSE, 'alice'),
                    ('acc_alice_prodx',  'acc_prodx',  FALSE, 'alice')
                """
            )
