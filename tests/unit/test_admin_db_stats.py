from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_account_id, get_pool
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


def _app(account_id: str = "acc_root") -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(admin.router)
    app.dependency_overrides[get_pool] = lambda: _Pool()
    app.dependency_overrides[get_account_id] = lambda: account_id
    return app


def _account(parent: str | None) -> Account:
    return Account(
        id="acc_root",
        parent_account_id=parent,
        can_mint_children=True,
        display_name="root",
        metadata={},
        created_at=datetime.now(UTC),
    )


def test_non_root_is_hidden(monkeypatch: Any) -> None:
    async def get_account(conn: Any, account_id: str) -> Account:
        return _account("acc_parent")

    monkeypatch.setattr(queries, "get_account", get_account)
    response = TestClient(_app()).get("/v1/admin/db-stats")
    assert response.status_code == 404


def test_root_gets_stats(monkeypatch: Any) -> None:

    async def get_account(conn: Any, account_id: str) -> Account:
        return _account(None)

    async def collect(pool: Any) -> DatabaseStats:
        return DatabaseStats(generated_at=datetime(2026, 8, 13, tzinfo=UTC), database_bytes=42)

    monkeypatch.setattr(queries, "get_account", get_account)
    monkeypatch.setattr(db_stats, "collect_database_stats", collect)
    app = _app()
    client = TestClient(app)
    assert client.get("/v1/admin/db-stats").json()["database_bytes"] == 42
