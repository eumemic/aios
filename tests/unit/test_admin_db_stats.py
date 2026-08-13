from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_pool
from aios.api.routers import admin
from aios.api.strict_query_params import reject_unknown_query_params
from aios.errors import install_exception_handlers
from aios.models.accounts import Account
from aios.models.db_stats import DatabaseStats
from aios.services import accounts as accounts_service
from aios.services import db_stats

ROOT_KEY = "root-secret"
TENANT_KEY = "tenant-secret"


def _account(account_id: str, parent: str | None) -> Account:
    return Account(
        id=account_id,
        parent_account_id=parent,
        can_mint_children=parent is None,
        display_name=account_id,
        metadata={},
        created_at=datetime.now(UTC),
    )


class _Connection:
    async def lookup(self, key_hash: bytes) -> tuple[Account, str] | None:
        if key_hash == accounts_service.hash_key(ROOT_KEY):
            return _account("acc_root", None), "key_root"
        if key_hash == accounts_service.hash_key(TENANT_KEY):
            return _account("acc_tenant", "acc_root"), "key_tenant"
        return None


class _Pool:
    def __init__(self) -> None:
        self.conn = _Connection()

    def acquire(self) -> Any:
        @contextlib.asynccontextmanager
        async def _acquire() -> Any:
            yield self.conn

        return _acquire()


def _app(monkeypatch: Any) -> FastAPI:
    async def lookup(conn: _Connection, *, key_hash: bytes) -> tuple[Account, str] | None:
        return await conn.lookup(key_hash)

    async def get_account(conn: _Connection, account_id: str) -> Account | None:
        if account_id == "acc_root":
            return _account(account_id, None)
        if account_id == "acc_tenant":
            return _account(account_id, "acc_root")
        return None

    monkeypatch.setattr("aios.db.queries.lookup_account_by_key_hash", lookup)
    monkeypatch.setattr("aios.db.queries.get_account", get_account)
    app = FastAPI(dependencies=[Depends(reject_unknown_query_params)])
    install_exception_handlers(app)
    app.include_router(admin.router)
    app.dependency_overrides[get_pool] = lambda: _Pool()
    return app


def test_real_auth_chain_hides_admin_route_from_tenant(
    monkeypatch: Any,
) -> None:
    called = False

    async def collect(pool: Any) -> DatabaseStats:
        nonlocal called
        called = True
        return DatabaseStats(generated_at=datetime.now(UTC), database_bytes=1)

    monkeypatch.setattr(db_stats, "collect_database_stats", collect)
    response = TestClient(_app(monkeypatch)).get(
        "/v1/admin/db-stats", headers={"Authorization": f"Bearer {TENANT_KEY}"}
    )
    assert response.status_code == 404
    assert not called


def test_real_auth_chain_allows_root_key(monkeypatch: Any) -> None:
    async def collect(pool: Any) -> DatabaseStats:
        return DatabaseStats(generated_at=datetime(2026, 8, 13, tzinfo=UTC), database_bytes=42)

    monkeypatch.setattr(db_stats, "collect_database_stats", collect)
    response = TestClient(_app(monkeypatch)).get(
        "/v1/admin/db-stats", headers={"Authorization": f"Bearer {ROOT_KEY}"}
    )
    assert response.status_code == 200
    assert response.json()["database_bytes"] == 42


def test_real_auth_chain_rejects_missing_malformed_and_unknown_keys(
    monkeypatch: Any,
) -> None:
    client = TestClient(_app(monkeypatch))
    for headers in (
        {},
        {"Authorization": "not-bearer"},
        {"Authorization": "Bearer unknown"},
    ):
        assert client.get("/v1/admin/db-stats", headers=headers).status_code == 401


def test_endpoint_accepts_no_sql_influencing_parameters(monkeypatch: Any) -> None:
    response = TestClient(_app(monkeypatch)).get(
        "/v1/admin/db-stats?table=sessions&months=60&sql=SELECT+1",
        headers={"Authorization": f"Bearer {ROOT_KEY}"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "http_error"


def test_admin_route_is_not_published(monkeypatch: Any) -> None:
    assert "/v1/admin/db-stats" not in _app(monkeypatch).openapi()["paths"]
