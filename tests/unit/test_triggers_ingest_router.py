"""Unit tests for the external-event ingress router (#1281).

Check-ordering and contract surface, NO real DB: the byte-cap (413) and
non-object-JSON (422) rejections fire BEFORE any pool/lookup touch, so a fake
pool that explodes if acquired proves the cheapest-first ordering. The
happy-path / 404 cases use a fake pool+conn that records the calls and routes
to monkeypatched query functions.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_pool
from aios.api.routers import triggers_ingest
from aios.db.queries.triggers import ResolvedExternalEventTrigger
from aios.errors import install_exception_handlers
from aios.models.triggers import MAX_INGEST_EVENT_BYTES


class _ExplodingPool:
    """Any acquire is a test failure — proves a check ran before the DB."""

    def acquire(self) -> Any:  # pragma: no cover - must never be called
        raise AssertionError("pool acquired before pre-DB validation")


class _FakeConn:
    def transaction(self) -> Any:
        @contextlib.asynccontextmanager
        async def _txn() -> Any:
            yield

        return _txn()


class _FakePool:
    def __init__(self) -> None:
        self.conn = _FakeConn()

    def acquire(self) -> Any:
        conn = self.conn

        @contextlib.asynccontextmanager
        async def _acq() -> Any:
            yield conn

        return _acq()


def _app(pool: Any) -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(triggers_ingest.router)
    app.dependency_overrides[get_pool] = lambda: pool
    return app


def test_oversized_body_413_before_db() -> None:
    client = TestClient(_app(_ExplodingPool()))
    big = json.dumps({"x": "a" * (MAX_INGEST_EVENT_BYTES + 1)})
    assert len(big.encode()) > MAX_INGEST_EVENT_BYTES
    resp = client.post("/v1/triggers/ingest/aios_evt_whatever", content=big)
    assert resp.status_code == 413


def test_non_object_body_422_before_db() -> None:
    client = TestClient(_app(_ExplodingPool()))
    # A bare JSON array is valid JSON but not an object — rejected before lookup.
    resp = client.post("/v1/triggers/ingest/aios_evt_x", content=json.dumps([1, 2, 3]))
    assert resp.status_code == 422


def test_unparseable_body_422_before_db() -> None:
    client = TestClient(_app(_ExplodingPool()))
    resp = client.post("/v1/triggers/ingest/aios_evt_x", content="{not json")
    assert resp.status_code == 422


def test_unknown_token_uniform_404(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _resolve(conn: Any, *, ingest_token_hash: str) -> None:
        return None

    monkeypatch.setattr(triggers_ingest.queries, "resolve_external_event_trigger", _resolve)
    client = TestClient(_app(_FakePool()))
    resp = client.post("/v1/triggers/ingest/aios_evt_missing", content=json.dumps({"a": 1}))
    assert resp.status_code == 404


def test_happy_path_records_fire_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    async def _resolve(conn: Any, *, ingest_token_hash: str) -> ResolvedExternalEventTrigger:
        recorded["token_hash"] = ingest_token_hash
        return ResolvedExternalEventTrigger(
            trigger_id="trig_1",
            account_id="acct_1",
            owner_session_id="sess_1",
            trigger_name="gh-hook",
        )

    async def _insert(conn: Any, **kwargs: Any) -> str:
        recorded["insert"] = kwargs
        return "trun_99"

    async def _defer(trigger_id: str, trigger_run_id: str) -> None:
        recorded["defer"] = (trigger_id, trigger_run_id)

    monkeypatch.setattr(triggers_ingest.queries, "resolve_external_event_trigger", _resolve)
    monkeypatch.setattr(triggers_ingest.queries, "insert_external_event_fire", _insert)
    monkeypatch.setattr(triggers_ingest, "defer_trigger_fire", _defer)

    client = TestClient(_app(_FakePool()))
    body = {"action": "labeled", "issue": {"number": 7}}
    resp = client.post("/v1/triggers/ingest/aios_evt_good", content=json.dumps(body))

    assert resp.status_code == 202
    assert resp.json() == {"trigger_run_id": "trun_99"}
    # The token is hashed, never used raw; the resolved row's account scopes
    # the carrier; the event rides verbatim; dispatch fires post-commit.
    assert recorded["token_hash"] != "aios_evt_good"
    assert recorded["insert"]["account_id"] == "acct_1"
    assert recorded["insert"]["event"] == body
    assert recorded["defer"] == ("trig_1", "trun_99")
