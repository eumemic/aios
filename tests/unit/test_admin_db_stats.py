from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_pool
from aios.api.routers import admin
from aios.db import queries
from aios.errors import install_exception_handlers
from aios.models.accounts import Account
from aios.models.db_stats import DatabaseStats
from aios.services import db_stats


class _Pool:
    def acquire(self) -> Any:
        @contextlib.asynccontextmanager
        async def _acquire() -> Any:
            yield object()

        return _acquire()


def _app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(admin.router)
    app.dependency_overrides[get_pool] = lambda: _Pool()
    return app


def _account(parent: str | None) -> Account:
    return Account(
        id="acc_root" if parent is None else "acc_tenant",
        parent_account_id=parent,
        can_mint_children=True,
        display_name="account",
        metadata={},
        created_at=datetime.now(UTC),
    )


def _install_key(monkeypatch: Any, account: Account) -> None:
    async def lookup(conn: Any, *, key_hash: str) -> tuple[Account, str]:
        return account, "key_1"

    async def get_account(conn: Any, account_id: str) -> Account:
        return account

    monkeypatch.setattr(queries, "lookup_account_by_key_hash", lookup)
    monkeypatch.setattr(queries, "get_account", get_account)


def test_non_root_is_hidden_and_collection_is_not_called(monkeypatch: Any) -> None:
    _install_key(monkeypatch, _account("acc_parent"))

    async def collect(pool: Any) -> DatabaseStats:
        raise AssertionError("tenant must not collect global stats")

    monkeypatch.setattr(db_stats, "collect_database_stats", collect)
    response = TestClient(_app()).get(
        "/v1/admin/db-stats", headers={"Authorization": "Bearer tenant-key"}
    )
    assert response.status_code == 404


def test_root_account_key_gets_stats(monkeypatch: Any) -> None:
    _install_key(monkeypatch, _account(None))

    async def collect(pool: Any) -> DatabaseStats:
        return DatabaseStats(generated_at=datetime(2026, 8, 13, tzinfo=UTC), database_bytes=42)

    monkeypatch.setattr(db_stats, "collect_database_stats", collect)
    response = TestClient(_app()).get(
        "/v1/admin/db-stats", headers={"Authorization": "Bearer root-key"}
    )
    assert response.json()["database_bytes"] == 42


def test_missing_and_malformed_keys_are_rejected() -> None:
    client = TestClient(_app())
    assert client.get("/v1/admin/db-stats").status_code == 401
    assert (
        client.get(
            "/v1/admin/db-stats", headers={"Authorization": "not-a-bearer-token"}
        ).status_code
        == 401
    )


def test_admin_route_is_absent_from_openapi() -> None:
    assert "/v1/admin/db-stats" not in TestClient(_app()).get("/openapi.json").json()["paths"]
