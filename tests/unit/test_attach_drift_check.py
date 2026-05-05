"""Unit tests for ``api.routers.connections._assert_account_in_snapshot``.

Validates that the attach-time drift guard (audit fix #1, design §5.6)
correctly maps the worker's snapshot envelope onto the right HTTP
status:

* Account present in snapshot → no raise.
* Account absent → :class:`AccountDriftError` (409 ``account_drift``).
* Connector ``status != "running"`` → 503 (snapshot unavailable, not drift).
* Worker error envelope → 503.
* Timeout on the LISTEN queue → 408.

Mocks :func:`listen_for_connector_result` and
:func:`defer_connector_status` so no DB / procrastinate work runs.

All ``aios.api`` imports are deferred to the test bodies because
``aios.harness.procrastinate_app`` runs ``get_settings()`` at module
import time and the conftest env-var fixture only fires after pytest's
collection phase finishes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest


@asynccontextmanager
async def _fake_listen(payload: str | None) -> AsyncIterator[asyncio.Queue[str]]:
    """Stand-in for :func:`listen_for_connector_result`.

    ``payload`` is pushed onto the queue immediately so the awaiter
    sees a NOTIFY without an actual Postgres round-trip.  ``None``
    leaves the queue empty (forces the await to time out).
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    if payload is not None:
        queue.put_nowait(payload)
    yield queue


def _patch_rpc(
    monkeypatch: pytest.MonkeyPatch, *, payload: str | None
) -> list[tuple[str | None, str | None]]:
    """Wire fake LISTEN + defer; record the connector name passed to defer."""
    captured: list[tuple[str | None, str | None]] = []

    def fake_listen(_db_url: str, call_id: str) -> Any:
        return _fake_listen(payload)

    async def fake_defer(
        *, call_id: str, connector: str | None = None, instance: str | None = None
    ) -> None:
        captured.append((call_id, connector))

    monkeypatch.setattr("aios.api.connector_rpc.listen_for_connector_result", fake_listen)
    monkeypatch.setattr("aios.api.routers.connections.defer_connector_status", fake_defer)
    return captured


class TestAssertAccountInSnapshot:
    async def test_account_present_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aios.api.routers.connections import _assert_account_in_snapshot

        envelope = {
            "connectors": [
                {
                    "connector": "echo",
                    "instance": "echo",
                    "status": "running",
                    "accounts": [{"id": "acct-1", "display_name": "A1"}],
                }
            ]
        }
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        # Should not raise.
        await _assert_account_in_snapshot("postgresql://x", connector="echo", account="acct-1")

    async def test_account_present_on_one_of_many_instances_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-instance: account on any running instance counts as present."""
        from aios.api.routers.connections import _assert_account_in_snapshot

        envelope = {
            "connectors": [
                {
                    "connector": "telegram",
                    "instance": "support",
                    "status": "running",
                    "accounts": [{"id": "111", "display_name": "Support"}],
                },
                {
                    "connector": "telegram",
                    "instance": "alerts",
                    "status": "running",
                    "accounts": [{"id": "222", "display_name": "Alerts"}],
                },
            ]
        }
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        await _assert_account_in_snapshot("postgresql://x", connector="telegram", account="222")

    async def test_account_absent_raises_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aios.api.routers.connections import _assert_account_in_snapshot
        from aios.errors import AccountDriftError

        envelope = {
            "connectors": [
                {
                    "connector": "echo",
                    "instance": "echo",
                    "status": "running",
                    "accounts": [{"id": "acct-1", "display_name": "A1"}],
                }
            ]
        }
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        with pytest.raises(AccountDriftError) as exc_info:
            await _assert_account_in_snapshot(
                "postgresql://x", connector="echo", account="vanished"
            )
        # 409 with the right error type and detail bag the operator can branch on.
        assert exc_info.value.status_code == 409
        assert exc_info.value.error_type == "account_drift"
        assert exc_info.value.detail == {"connector": "echo", "account": "vanished"}

    async def test_running_with_empty_snapshot_raises_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Boot window: ``status == "running"`` but accounts haven't been pushed yet.

        Mirrors the inbound handler's empty-snapshot exemption — without it,
        an attach placed in this window would false-positive as drift.
        """
        from fastapi import HTTPException

        from aios.api.routers.connections import _assert_account_in_snapshot

        envelope = {
            "connectors": [
                {
                    "connector": "echo",
                    "instance": "echo",
                    "status": "running",
                    "accounts": [],
                }
            ]
        }
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        with pytest.raises(HTTPException) as exc_info:
            await _assert_account_in_snapshot("postgresql://x", connector="echo", account="acct-1")
        assert exc_info.value.status_code == 503
        assert "not yet populated" in exc_info.value.detail

    async def test_no_running_instances_raises_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Snapshot unavailable while every instance is starting / restarting."""
        from fastapi import HTTPException

        from aios.api.routers.connections import _assert_account_in_snapshot

        envelope = {
            "connectors": [
                {
                    "connector": "echo",
                    "instance": "echo",
                    "status": "starting",
                    "accounts": [],
                }
            ]
        }
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        with pytest.raises(HTTPException) as exc_info:
            await _assert_account_in_snapshot("postgresql://x", connector="echo", account="acct-1")
        assert exc_info.value.status_code == 503
        assert "no running instances" in exc_info.value.detail
        assert "starting" in exc_info.value.detail

    async def test_worker_error_envelope_raises_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``not_enabled`` / ``not_ready`` envelopes from the worker → 503."""
        from fastapi import HTTPException

        from aios.api.routers.connections import _assert_account_in_snapshot

        envelope = {"error": "connector 'echo' not enabled", "code": "not_enabled"}
        _patch_rpc(monkeypatch, payload=json.dumps(envelope))
        with pytest.raises(HTTPException) as exc_info:
            await _assert_account_in_snapshot("postgresql://x", connector="echo", account="acct-1")
        assert exc_info.value.status_code == 503
        assert "snapshot unavailable" in exc_info.value.detail

    async def test_timeout_raises_408(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No NOTIFY within the snapshot-check window → 408 from the shared helper."""
        from fastapi import HTTPException

        from aios.api.routers.connections import _assert_account_in_snapshot

        _patch_rpc(monkeypatch, payload=None)  # queue stays empty

        # Tighten the timeout so the test doesn't actually wait 10 seconds.
        monkeypatch.setattr("aios.api.routers.connections._DRIFT_CHECK_TIMEOUT_S", 0.05)

        with pytest.raises(HTTPException) as exc_info:
            await _assert_account_in_snapshot("postgresql://x", connector="echo", account="acct-1")
        assert exc_info.value.status_code == 408
        assert "did not respond" in exc_info.value.detail
